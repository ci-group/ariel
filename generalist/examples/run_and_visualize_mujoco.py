"""Run a longer policy simulation and visualize in MuJoCo.

This script:
1. Generates a random morphology (or uses nominal)
2. Rolls out the geometric expert (or an RL policy) on a trajectory
3. Converts the morphology to a DroneBlueprint
4. Compiles it to MuJoCo mjSpec
5. Replays the trajectory with MuJoCo visualization
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ARIEL imports
from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint

# Generalist imports
from generalist.envs import DroneTrackingEnv
from generalist.envs.observation import DT
from generalist.expert import GeometricExpert
from generalist.morphology.genome import NOMINAL_PHI, sample_valid_morphology
from generalist.trajectories import generate_circle, generate_figure8


def _phi_to_ariel_genome(phi: np.ndarray) -> dict:
    """Convert our φ format to ARIEL's SphericalAngularDroneGenomeHandler format.

    Our φ: 4 rotors × 4 params [arm_len, azimuth, elevation, spin_dir]
    ARIEL: 4 rotors × 6 params [arm_len, azimuth, elevation, motor_disc_azimuth,
                                 motor_disc_pitch, spin_dir]

    For simplicity, set motor_disc angles to 0 (no motor tilt).
    """
    phi_mat = phi.reshape(4, 4)
    ariel_arms = []
    for i in range(4):
        arm_len = float(phi_mat[i, 0])
        azimuth = float(phi_mat[i, 1])
        elevation = float(phi_mat[i, 2])
        spin_dir_sign = int(phi_mat[i, 3])
        spin_dir = 1 if spin_dir_sign > 0 else 0  # ARIEL: 0=CCW, 1=CW
        ariel_arms.append([
            arm_len, azimuth, elevation,
            0.0,  # motor_disc_azimuth
            0.0,  # motor_disc_pitch
            spin_dir,
        ])
    return {"arms": np.array(ariel_arms, dtype=np.float64)}


def run_and_visualize(
    duration: float = 20.0,
    task: str = "circle",
    morphology: str = "nominal",
    use_expert: bool = True,
    rl_checkpoint: str | None = None,
    render: bool = True,
    output_video: str | None = None,
) -> None:
    """Roll out a policy and visualize in MuJoCo.

    Args:
        duration: rollout duration (seconds)
        task: "circle" or "figure8"
        morphology: "nominal" or "random"
        use_expert: if True, use geometric expert
        rl_checkpoint: if set, load RL policy from checkpoint (overrides use_expert)
        render: if True, display MuJoCo viewer
        output_video: if set, save video to this path
    """
    import mujoco
    import mujoco.viewer
    import torch

    # ---- Setup morphology ----
    if morphology == "nominal":
        phi = NOMINAL_PHI
    elif morphology == "random":
        rng = np.random.default_rng(0)
        phi = sample_valid_morphology(rng)
    else:
        raise ValueError(f"unknown morphology: {morphology}")

    print(f"Morphology: {morphology}")
    print(f"φ shape: {phi.shape}")

    # ---- Setup environment ----
    env = DroneTrackingEnv(phi=phi, seed=0)
    if task == "circle":
        reference = generate_circle(radius=2.0, speed=2.0, altitude=1.5, dt=DT, T_total=duration)
    elif task == "figure8":
        reference = generate_figure8(scale=2.0, speed=2.0, altitude=1.5, dt=DT, T_total=duration)
    else:
        raise ValueError(f"unknown task: {task}")

    print(f"Task: {task}, Duration: {duration}s, Steps: {len(reference)}")

    # ---- Setup policy (RL or expert) ----
    policy = None
    lstm_states = None
    if rl_checkpoint:
        print(f"Loading RL policy from {rl_checkpoint}")
        from generalist.policy import GeneralistDronePolicy
        policy = GeneralistDronePolicy()
        policy.load_state_dict(torch.load(rl_checkpoint, map_location="cpu"))
        policy.eval()
        expert = None
    else:
        expert = GeometricExpert(phi) if use_expert else None
        policy = None

    # ---- Rollout ----
    obs, _ = env.reset(options={"reference": reference})
    trajectory = []
    for t in range(len(reference) - 1):
        state = env.sim.get_state()

        if policy:
            # Use RL policy with LSTM state management
            obs_t = torch.from_numpy(obs[np.newaxis, :]).float()
            with torch.no_grad():
                action_mean, _, lstm_states = policy(obs_t, lstm_states=lstm_states)
                action = np.tanh(action_mean.squeeze(0).numpy())
        elif expert:
            ref_pos = reference[t, :3]
            ref_vel = (reference[t + 1, :3] - reference[t, :3]) / DT
            ref_yaw = float(reference[t, 3])
            action = expert.get_action(state, ref_pos, ref_vel, ref_yaw)
        else:
            action = np.random.uniform(-1, 1, 4)

        obs, reward, term, trunc, info = env.step(action.astype(np.float32))

        trajectory.append({
            "pos": state["pos"].copy(),
            "vel": state["vel"].copy(),
            "quat": state["quat"].copy(),
            "omega": state["omega"].copy(),
            "action": action.copy(),
            "error": info.get("pos_error", 0.0),
        })

        if term or trunc:
            print(f"Episode ended at step {t}")
            lstm_states = None  # reset LSTM at episode boundary
            break

    print(f"Trajectory: {len(trajectory)} steps")

    # ---- Convert morphology to ARIEL blueprint ----
    ariel_genome = _phi_to_ariel_genome(phi)
    ariel_arms = ariel_genome["arms"]
    print(f"ARIEL arms shape: {ariel_arms.shape}")

    blueprint = spherical_angular_to_blueprint(ariel_arms, propsize=2)
    print(f"Blueprint: {blueprint.summary()}")

    # ---- Compile to MuJoCo ----
    spec = blueprint_to_mjspec(blueprint, body_name="drone")
    spec.worldbody.add_body(name="world_origin").add_geom(
        type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1], rgba=[1, 0, 0, 1]
    )

    # Add a free joint to the drone body for free-floating dynamics
    drone_body = spec.worldbody.find_child("drone")
    if drone_body:
        try:
            drone_body.add_freejoint()
            print("Added freejoint to drone body")
        except Exception:
            pass  # freejoint may already exist

    try:
        model = spec.compile()
    except Exception as e:
        print(f"Warning: could not compile spec: {e}")
        print("Proceeding without visualization")
        return
    data = mujoco.MjData(model)

    print(f"MuJoCo model compiled: {model.nbody} bodies, {model.nq} q dims")

    # ---- Render with viewer ----
    if render:
        print("Launching MuJoCo viewer for trajectory playback...")
        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # Configure camera
                viewer.cam.distance = 5.0
                viewer.cam.lookat[:] = [0, 0, 0]

                for i, state in enumerate(trajectory):
                    if viewer.is_running() is False:
                        break

                    # Set drone state (freejoint format: qpos=[x,y,z,qw,qx,qy,qz])
                    data.qpos[0:3] = state["pos"]
                    data.qpos[3:7] = state["quat"]
                    data.qvel[0:3] = state["vel"]
                    data.qvel[3:6] = state["omega"]

                    # Update simulation state
                    mujoco.mj_forward(model, data)

                    # Let viewer render
                    viewer.sync()

            print("Visualization complete")
        except Exception as e:
            print(f"Could not launch viewer: {e}")
            print("(This is expected in headless/SSH environments)")
    else:
        print("Skipping visualization (render=False)")

    # Summary
    errors = [s["error"] for s in trajectory]
    print(f"\nTrajectory Summary:")
    print(f"  Mean error: {np.mean(errors):.3f} m")
    print(f"  Max error: {np.max(errors):.3f} m")
    print(f"  RMS error: {np.sqrt(np.mean(np.array(errors) ** 2)):.3f} m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run and visualize drone trajectory in MuJoCo")
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
        "--no-render", action="store_true", help="Skip MuJoCo visualization"
    )
    parser.add_argument(
        "--output-video", type=str, default=None, help="Save trajectory video to file"
    )
    args = parser.parse_args()

    run_and_visualize(
        duration=args.duration,
        task=args.task,
        morphology=args.morphology,
        use_expert=not args.no_expert,
        rl_checkpoint=args.rl_checkpoint,
        render=not args.no_render,
        output_video=args.output_video,
    )
