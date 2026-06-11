"""Run a policy in MuJoCo with proper physics simulation.

This script:
1. Generates a morphology (nominal or random)
2. Compiles it to a MuJoCo model via the ARIEL blueprint backend
3. Either:
   - "physics" mode (default): Runs the policy in MuJoCo with actuator
     commands and forward dynamics (mj_step)
   - "replay" mode: Rolls out in MinimalQuadSim and replays the trajectory
     in MuJoCo (state-only, no physics)

The MuJoCo model has one site-attached thrust actuator per motor.
ctrl[i] ∈ [0, 1] maps to thrust ∈ [0, max_thrust] N along motor's local +Z.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Any

import numpy as np

# ARIEL imports
from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint

# Generalist imports
from generalist.envs import DroneTrackingEnv
from generalist.envs.observation import DT, T_REF, build_observation, build_reference_window
from generalist.envs.sim_minimal import SimConfig
from generalist.expert import GeometricExpert
from generalist.morphology.allocation import ctbr_to_rotors, compute_allocation_matrix
from generalist.morphology.genome import (
    F_MAX_PER_ROTOR,
    N_ROTORS,
    NOMINAL_PHI,
    sample_valid_morphology,
)
from generalist.trajectories import generate_circle, generate_figure8


# CTBR scaling — must match drone_tracking_env.py
CTBR_RATE_MAX = 6.0


def _phi_to_ariel_genome(phi: np.ndarray) -> np.ndarray:
    """Convert our φ format to ARIEL's spherical-angular genome (4 rotors × 6 params).

    Our φ: 4 rotors × 4 params [arm_len, azimuth, elevation, spin_dir]
    ARIEL: 4 rotors × 6 params [arm_len, azimuth, elevation, motor_disc_azimuth,
                                 motor_disc_pitch, spin_dir]
    """
    phi_mat = phi.reshape(4, 4)
    ariel_arms = np.zeros((4, 6), dtype=np.float64)
    for i in range(4):
        ariel_arms[i, 0] = float(phi_mat[i, 0])  # arm_len
        ariel_arms[i, 1] = float(phi_mat[i, 1])  # azimuth
        ariel_arms[i, 2] = float(phi_mat[i, 2])  # elevation
        ariel_arms[i, 3] = 0.0                   # motor_disc_azimuth (no tilt)
        ariel_arms[i, 4] = 0.0                   # motor_disc_pitch
        ariel_arms[i, 5] = 1 if int(phi_mat[i, 3]) > 0 else 0  # spin_dir (0=CCW, 1=CW)
    return ariel_arms


def _build_mjmodel(
    phi: np.ndarray,
    max_thrust: float,
    total_mass: float = 0.5,
    motor_mass: float = 0.05,
    arm_mass: float = 0.01,
):
    """Build a MuJoCo model from our φ via ARIEL blueprint backend.

    Sets core_mass_override so the total drone mass matches `total_mass`
    (the value used by MinimalQuadSim's controller assumptions).

    Returns:
        (model, data, blueprint)
    """
    import mujoco

    ariel_arms = _phi_to_ariel_genome(phi)
    blueprint = spherical_angular_to_blueprint(ariel_arms, propsize=2)

    # Compute core mass so the total matches our SimConfig.mass
    # (4 arms + 4 motors + 1 core; rotors are mass=0)
    other_mass = N_ROTORS * (motor_mass + arm_mass)
    core_mass = max(0.01, total_mass - other_mass)

    spec = blueprint_to_mjspec(
        blueprint,
        body_name="drone",
        max_thrust=max_thrust,
        motor_mass=motor_mass,
        arm_mass=arm_mass,
        core_mass_override=core_mass,
    )

    # Add a small visible origin marker for reference
    spec.worldbody.add_body(name="world_origin").add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.05], rgba=[1, 0, 0, 1]
    )

    # Add a floor for visual reference
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[10.0, 10.0, 0.1],
        rgba=[0.7, 0.7, 0.8, 1.0],
    )

    # Add freejoint to drone body so it can fly
    drone_body = spec.worldbody.find_child("drone")
    if drone_body:
        try:
            drone_body.add_freejoint()
        except Exception:
            pass  # already has one

    model = spec.compile()
    data = mujoco.MjData(model)
    return model, data, blueprint


def _mujoco_state_to_dict(data) -> dict:
    """Extract drone state from MuJoCo MjData in the format MinimalQuadSim uses.

    Returns:
        dict with keys: pos, vel, quat (wxyz), quat_xyzw, omega, yaw
    """
    pos = data.qpos[0:3].copy()
    quat_wxyz = data.qpos[3:7].copy()  # MuJoCo: wxyz
    w, x, y, z = quat_wxyz
    quat_xyzw = np.array([x, y, z, w])

    vel = data.qvel[0:3].copy()
    omega = data.qvel[3:6].copy()  # body-frame angular velocity
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return {
        "pos": pos,
        "vel": vel,
        "quat": quat_wxyz,
        "quat_xyzw": quat_xyzw,
        "omega": omega,
        "yaw": float(yaw),
    }


def _ctbr_action_to_ctrls(
    action: np.ndarray,
    phi: np.ndarray,
    omega: np.ndarray,
    inertia_diag: tuple,
    dt: float,
    f_max_per_rotor: float,
) -> np.ndarray:
    """Convert CTBR action [-1,1]^4 to per-motor ctrl values [0,1]^4.

    Mirrors DroneTrackingEnv._action_to_rotor_thrusts() exactly.
    """
    action = np.clip(action, -1.0, 1.0).astype(np.float64)
    c_norm = (action[0] + 1.0) * 0.5
    Fz_body = c_norm * f_max_per_rotor * N_ROTORS  # total thrust in N

    # Inner attitude rate loop (feedback-linearized)
    rate_des = action[1:] * CTBR_RATE_MAX
    I = np.diag(inertia_diag)
    torque_demand = I @ (rate_des - omega) / dt + np.cross(omega, I @ omega)

    wrench = np.concatenate([[Fz_body], torque_demand])  # [Fz, τx, τy, τz]
    M = compute_allocation_matrix(phi)
    f_per_rotor = ctbr_to_rotors(wrench, M, f_min=0.0, f_max=f_max_per_rotor)

    # Normalize to [0, 1] for MuJoCo actuator ctrl
    return (f_per_rotor / f_max_per_rotor).astype(np.float64)


def _reset_drone_state(data, init_pos: np.ndarray, init_yaw: float) -> None:
    """Set the drone's initial position, orientation, and zero velocities."""
    data.qpos[0:3] = init_pos
    # Yaw-only quaternion: (cos(yaw/2), 0, 0, sin(yaw/2)) in wxyz
    data.qpos[3] = np.cos(init_yaw / 2.0)
    data.qpos[4] = 0.0
    data.qpos[5] = 0.0
    data.qpos[6] = np.sin(init_yaw / 2.0)
    data.qvel[:] = 0.0


def run_physics_simulation(
    phi: np.ndarray,
    reference: np.ndarray,
    policy: Any,
    expert: Any,
    sim_cfg: SimConfig,
    render: bool,
) -> list[dict]:
    """Run the policy in MuJoCo with real physics (mj_step).

    The drone state evolves via MuJoCo's forward dynamics; actuator ctrl
    values are computed from the policy's CTBR action via the same
    allocation pipeline as DroneTrackingEnv.

    Returns:
        Trajectory as list of dicts (one per simulation step).
    """
    import mujoco
    import mujoco.viewer
    import torch

    # Build model with matching max_thrust
    model, data, _ = _build_mjmodel(
        phi, max_thrust=sim_cfg.f_max_per_rotor, total_mass=sim_cfg.mass
    )
    print(f"MuJoCo model: {model.nbody} bodies, {model.nq} qpos, {model.nu} actuators")

    # Use small MuJoCo timestep for stable physics integration.
    # Run multiple substeps per control step to keep ctrl rate at 50Hz.
    mujoco_dt = 0.002  # 500Hz physics
    n_substeps = max(1, int(round(sim_cfg.dt / mujoco_dt)))
    model.opt.timestep = mujoco_dt

    # Read actual inertia from MuJoCo for the drone root body
    drone_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone")
    actual_inertia = tuple(model.body_inertia[drone_body_id].copy())
    actual_mass = float(model.body_subtreemass[drone_body_id])
    print(f"MuJoCo drone: mass={actual_mass:.3f} kg, inertia={actual_inertia}")
    print(f"Physics: {mujoco_dt}s ({n_substeps} substeps per control)")

    # Initialize drone at reference start
    init_pos = reference[0, :3].copy()
    init_yaw = float(reference[0, 3])
    _reset_drone_state(data, init_pos, init_yaw)
    mujoco.mj_forward(model, data)

    trajectory: list[dict] = []
    lstm_states = None
    prev_action = np.zeros(4, dtype=np.float64)
    prev_action[0] = -1.0

    def _maybe_step_viewer(viewer):
        if viewer is not None:
            viewer.sync()

    def _step_loop(viewer=None):
        nonlocal lstm_states, prev_action
        for t in range(len(reference) - 1):
            state = _mujoco_state_to_dict(data)

            # ---- Build observation for policy ----
            if policy is not None:
                ref_window = build_reference_window(
                    reference, t, state["pos"], state["yaw"], T_REF
                )
                obs = build_observation(state, ref_window, phi, prev_action)
                obs_t = torch.from_numpy(obs[np.newaxis, :].astype(np.float32))
                with torch.no_grad():
                    action_mean, _, lstm_states = policy(obs_t, lstm_states=lstm_states)
                    action = np.tanh(action_mean.squeeze(0).numpy()).astype(np.float64)
            elif expert is not None:
                ref_pos = reference[t, :3]
                ref_vel = (reference[t + 1, :3] - reference[t, :3]) / sim_cfg.dt
                ref_yaw = float(reference[t, 3])
                action = expert.get_action(state, ref_pos, ref_vel, ref_yaw)
            else:
                action = np.random.uniform(-1, 1, 4)

            # ---- Convert CTBR to per-motor ctrl ----
            # Use MuJoCo's actual inertia so the feedback-linearized
            # rate controller matches the simulated dynamics.
            ctrls = _ctbr_action_to_ctrls(
                action, phi, state["omega"],
                actual_inertia, sim_cfg.dt, sim_cfg.f_max_per_rotor,
            )
            data.ctrl[0:4] = ctrls

            # ---- Step MuJoCo physics (multiple substeps for stability) ----
            for _ in range(n_substeps):
                mujoco.mj_step(model, data)

            # ---- Record ----
            error = float(np.linalg.norm(data.qpos[0:3] - reference[t + 1, :3]))
            trajectory.append({
                "pos": data.qpos[0:3].copy(),
                "vel": data.qvel[0:3].copy(),
                "quat": data.qpos[3:7].copy(),
                "omega": data.qvel[3:6].copy(),
                "action": action.copy(),
                "ctrl": ctrls.copy(),
                "error": error,
                "ref": reference[t + 1, :3].copy(),
            })

            prev_action = action

            # Render
            _maybe_step_viewer(viewer)

            # Termination: crash (below ground) or huge tracking error
            if data.qpos[2] < 0.05:
                print(f"Crashed (below ground) at step {t}")
                break
            if error > 10.0:
                print(f"Lost (error > 10m) at step {t}")
                break
            if viewer is not None and not viewer.is_running():
                break

    if render:
        print("Launching MuJoCo viewer (physics mode)...")
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 5.0
                viewer.cam.lookat[:] = init_pos
                _step_loop(viewer)
            print("✓ Visualization complete")
        except Exception as e:
            print(f"⚠ Viewer launch failed: {type(e).__name__}: {str(e)[:80]}")
            print(f"  (Falling back to headless physics simulation)")
            # Re-init state and run without viewer
            _reset_drone_state(data, init_pos, init_yaw)
            mujoco.mj_forward(model, data)
            trajectory.clear()
            lstm_states = None
            prev_action = np.zeros(4); prev_action[0] = -1.0
            _step_loop(None)
    else:
        _step_loop(None)

    return trajectory


def run_replay_visualization(
    phi: np.ndarray,
    reference: np.ndarray,
    policy: Any,
    expert: Any,
    sim_cfg: SimConfig,
    render: bool,
) -> list[dict]:
    """Run policy in MinimalQuadSim, then replay state in MuJoCo viewer.

    Useful for visualizing trajectories without depending on MuJoCo physics
    parameters matching MinimalQuadSim's.

    Returns:
        Trajectory as list of dicts.
    """
    import mujoco
    import mujoco.viewer
    import torch

    # ---- Setup env (uses MinimalQuadSim internally) ----
    env = DroneTrackingEnv(phi=phi, seed=0)
    obs, _ = env.reset(options={"reference": reference})

    trajectory: list[dict] = []
    lstm_states = None
    for t in range(len(reference) - 1):
        state = env.sim.get_state()

        if policy is not None:
            obs_t = torch.from_numpy(obs[np.newaxis, :]).float()
            with torch.no_grad():
                action_mean, _, lstm_states = policy(obs_t, lstm_states=lstm_states)
                action = np.tanh(action_mean.squeeze(0).numpy())
        elif expert is not None:
            ref_pos = reference[t, :3]
            ref_vel = (reference[t + 1, :3] - reference[t, :3]) / sim_cfg.dt
            ref_yaw = float(reference[t, 3])
            action = expert.get_action(state, ref_pos, ref_vel, ref_yaw)
        else:
            action = np.random.uniform(-1, 1, 4)

        obs, _, term, trunc, info = env.step(action.astype(np.float32))

        trajectory.append({
            "pos": state["pos"].copy(),
            "vel": state["vel"].copy(),
            "quat": state["quat"].copy(),
            "omega": state["omega"].copy(),
            "action": action.copy(),
            "error": info.get("pos_error", 0.0),
            "ref": reference[t, :3].copy(),
        })

        if term or trunc:
            print(f"Episode ended at step {t}")
            break

    print(f"Trajectory: {len(trajectory)} steps")

    # ---- Replay in MuJoCo viewer ----
    if render:
        model, data, _ = _build_mjmodel(phi, max_thrust=sim_cfg.f_max_per_rotor)
        print(f"MuJoCo model: {model.nbody} bodies, {model.nq} qpos, {model.nu} actuators")
        print("Launching MuJoCo viewer (replay mode)...")
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 5.0
                viewer.cam.lookat[:] = trajectory[0]["pos"] if trajectory else [0, 0, 0]

                import time
                for state in trajectory:
                    if not viewer.is_running():
                        break

                    data.qpos[0:3] = state["pos"]
                    data.qpos[3:7] = state["quat"]
                    data.qvel[0:3] = state["vel"]
                    data.qvel[3:6] = state["omega"]
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(sim_cfg.dt)  # real-time playback

            print("✓ Visualization complete")
        except Exception as e:
            print(f"⚠ Viewer launch failed: {type(e).__name__}: {str(e)[:80]}")

    return trajectory


def run_and_visualize(
    duration: float = 20.0,
    task: str = "circle",
    morphology: str = "nominal",
    use_expert: bool = True,
    rl_checkpoint: str | None = None,
    render: bool = True,
    mode: str = "physics",
) -> None:
    """Run policy in MuJoCo with full physics simulation."""
    # Suppress GLFW warnings (common in headless/SSH)
    warnings.filterwarnings("ignore", message=".*GLFWError.*")

    # ---- Setup morphology ----
    if morphology == "nominal":
        phi = NOMINAL_PHI
    elif morphology == "random":
        rng = np.random.default_rng(0)
        phi = sample_valid_morphology(rng)
    else:
        raise ValueError(f"unknown morphology: {morphology}")

    print(f"Morphology: {morphology}, φ shape: {phi.shape}")

    # ---- Setup task ----
    if task == "circle":
        reference = generate_circle(radius=2.0, speed=2.0, altitude=1.5, dt=DT, T_total=duration)
    elif task == "figure8":
        reference = generate_figure8(scale=2.0, speed=2.0, altitude=1.5, dt=DT, T_total=duration)
    else:
        raise ValueError(f"unknown task: {task}")

    print(f"Task: {task}, Duration: {duration}s, Reference steps: {len(reference)}")

    # ---- Setup policy ----
    policy = None
    expert = None
    if rl_checkpoint:
        import torch
        print(f"Loading RL policy from {rl_checkpoint}")
        from generalist.policy import GeneralistDronePolicy
        policy = GeneralistDronePolicy()
        policy.load_state_dict(torch.load(rl_checkpoint, map_location="cpu"))
        policy.eval()
    elif use_expert:
        expert = GeometricExpert(phi)
        print("Using geometric expert controller")
    else:
        print("Using random actions")

    # ---- Run ----
    sim_cfg = SimConfig(dt=DT, f_max_per_rotor=F_MAX_PER_ROTOR)
    print(f"Simulation: dt={sim_cfg.dt}, f_max={sim_cfg.f_max_per_rotor} N/rotor")
    print(f"Mode: {mode}")

    if mode == "physics":
        trajectory = run_physics_simulation(phi, reference, policy, expert, sim_cfg, render)
    elif mode == "replay":
        trajectory = run_replay_visualization(phi, reference, policy, expert, sim_cfg, render)
    else:
        raise ValueError(f"unknown mode: {mode}")

    # ---- Summary ----
    if not trajectory:
        print("No trajectory recorded.")
        return

    errors = np.array([s["error"] for s in trajectory])
    print(f"\nTrajectory Summary ({mode} mode):")
    print(f"  Steps:      {len(trajectory)} / {len(reference) - 1}")
    print(f"  Mean error: {errors.mean():.3f} m")
    print(f"  Max error:  {errors.max():.3f} m")
    print(f"  RMS error:  {np.sqrt(np.mean(errors ** 2)):.3f} m")
    final_pos = trajectory[-1]["pos"]
    print(f"  Final pos:  [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and visualize drone policy in MuJoCo")
    parser.add_argument("--duration", type=float, default=20.0, help="Simulation duration (s)")
    parser.add_argument(
        "--task", choices=["circle", "figure8"], default="circle", help="Reference task"
    )
    parser.add_argument(
        "--morphology", choices=["nominal", "random"], default="nominal", help="Morphology type"
    )
    parser.add_argument(
        "--no-expert", action="store_true", help="Use random actions instead of expert"
    )
    parser.add_argument(
        "--rl-checkpoint", type=str, default=None, help="Path to RL policy checkpoint"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Skip MuJoCo viewer"
    )
    parser.add_argument(
        "--mode", choices=["physics", "replay"], default="physics",
        help="physics: MuJoCo forward dynamics with actuator commands (default); "
             "replay: rollout in MinimalQuadSim then replay state in MuJoCo viewer"
    )
    args = parser.parse_args()

    run_and_visualize(
        duration=args.duration,
        task=args.task,
        morphology=args.morphology,
        use_expert=not args.no_expert,
        rl_checkpoint=args.rl_checkpoint,
        render=not args.no_render,
        mode=args.mode,
    )
