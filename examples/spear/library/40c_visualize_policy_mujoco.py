"""Visualize the trained hover PPO policy on a sequence of morphologies in MuJoCo.

For each morph in a configured sequence (canonical, +5°, +15°, +30° on
arm 0, a random σ=15° draw, etc.), this script:

  1. Decodes the perturbed genome to a DroneBlueprint.
  2. Rolls the trained policy through ResidualDroneEnv(hover) to get
     positions + orientations (NED).
  3. Builds a MuJoCo scene with that blueprint (real geometry — arms,
     motors, propellers), plus a green sphere at the hover target.
  4. Kinematically plays back the rollout.

Two output modes:

  * `--view`     : interactive passive viewer, one morph at a time.
                   Close the window to advance to the next morph.
  * default      : write a single MP4 with title cards between morphs.

Usage:
    uv run examples/spear/library/40c_visualize_policy_mujoco.py           # MP4
    uv run examples/spear/library/40c_visualize_policy_mujoco.py --view    # interactive
"""

import argparse
import importlib.util
import math
import sys
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec, blueprint_to_propellers,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder
from stable_baselines3 import PPO

sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location(
    "mba", str(Path(__file__).parent / "40_morph_break_analysis.py"),
)
mba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mba)

OUT = Path(__file__).parent / "morph_break_out"
POLICY = OUT / "hover_policy.zip"
HOVER_TARGET_NED = np.array([0.0, 0.0, -1.5], dtype=np.float64)


# ---------------------------------------------------------------- sequence

def morph_sequence(mode: str = "az"):
    """List of (label, offsets_rad_vector) to visualize in order.

    Chosen to span the observed break regime: canonical → mild → break-point
    → severe drift → all-arm noise.
    """
    def _single(arm, deg): v = np.zeros(6, np.float32); v[arm] = math.radians(deg); return v
    rng = np.random.RandomState(1000)
    all_arm = rng.normal(0.0, math.radians(15.0), size=6).astype(np.float32)
    axis = "az" if mode == "az" else "pitch (tilt)"
    return [
        (f"canonical (perfect hex)",             np.zeros(6, np.float32)),
        (f"arm 0 {axis} +5 deg (mild)",          _single(0, 5)),
        (f"arm 0 {axis} +15 deg (past break)",   _single(0, 15)),
        (f"arm 0 {axis} +30 deg (severe)",       _single(0, 30)),
        (f"all arms, {axis} sigma=15 deg noise", all_arm),
    ]


# ---------------------------------------------------------------- rollout

def rollout_policy(model, morph: dict, seed: int = 42):
    """Return pos_ned (T,3), euler_ned (T,3) roll/pitch/yaw."""
    env = mba.make_env(morph, num_envs=1, seed=seed,
                       frozen_features=None, device="cpu")
    obs = env.reset()
    pos, euler = [], []
    for _ in range(mba.EPISODE_STEPS):
        action, _ = model.predict(obs, deterministic=True)
        env.step_async(action)
        obs, r, dones, _info = env.step_wait()
        pos.append(env.world_states[0, 0:3].cpu().numpy().copy())
        euler.append(env.world_states[0, 6:9].cpu().numpy().copy())
        if bool(dones[0]):
            break
    return np.asarray(pos, dtype=np.float64), np.asarray(euler, dtype=np.float64)


# ---------------------------------------------------------------- frame math

def _euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw→pitch→roll) intrinsic Euler → quaternion (w,x,y,z)."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float64)


def _ned_to_enu(pos_ned: np.ndarray, euler_ned: np.ndarray):
    """Convert NED trajectory to ENU + quaternions for MuJoCo.

    ENU has x_ENU = y_NED, y_ENU = x_NED, z_ENU = -z_NED.
    Roll/pitch/yaw signs are adjusted accordingly.
    """
    pos_enu = np.empty_like(pos_ned)
    pos_enu[:, 0] = pos_ned[:, 1]
    pos_enu[:, 1] = pos_ned[:, 0]
    pos_enu[:, 2] = -pos_ned[:, 2]
    quat_enu = np.empty((len(pos_ned), 4), dtype=np.float64)
    for i, (r, p, y) in enumerate(euler_ned):
        # ENU yaw = pi/2 - yaw_NED, pitch and roll swap and negate accordingly
        quat_enu[i] = _euler_zyx_to_quat(p, r, math.pi / 2.0 - y)
    return pos_enu, quat_enu


# ---------------------------------------------------------------- scene

def build_scene(morph: dict, initial_pos_enu):
    bp = morph["_blueprint"]
    mean_arm = float(np.mean(
        [np.linalg.norm(np.asarray(p["loc"])[:2]) for p in morph["propellers"]]
    ))
    drone_spec = blueprint_to_mjspec(
        bp, motor_mass=0.01, arm_mass=0.034 * mean_arm, body_name="drone",
    )
    world_mj = SimpleFlatWorld()
    world_mj.spawn(
        drone_spec,
        position=(float(initial_pos_enu[0]),
                  float(initial_pos_enu[1]),
                  float(initial_pos_enu[2])),
        correct_collision_with_floor=False,
    )
    # Hover target marker (green sphere) in ENU
    target_enu = [HOVER_TARGET_NED[1], HOVER_TARGET_NED[0], -HOVER_TARGET_NED[2]]
    tgt_body = world_mj.spec.worldbody.add_body(
        name="hover_target", pos=target_enu, quat=[1, 0, 0, 0],
    )
    tgt_body.add_geom(
        name="hover_target_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.08, 0, 0], rgba=(0.1, 0.9, 0.1, 0.85),
        contype=0, conaffinity=0,
    )
    model_mj = world_mj.spec.compile()
    data_mj = mujoco.MjData(model_mj)
    return model_mj, data_mj


# ---------------------------------------------------------------- morph helper

def build_morph_with_bp(base_genome, offsets, mode: str = "az"):
    g = mba.perturbed_genome(base_genome, offsets, mode=mode)
    bp = spherical_angular_to_blueprint(
        g, core_mass=mba.CORE_MASS, propsize=mba.PROP_SIZE,
    )
    props = blueprint_to_propellers(bp, convention="ned")
    from ariel.simulation.drone.drone_configuration import DroneConfiguration
    from ariel.simulation.drone.dynamics_params import derive_reference_params
    from ariel.simulation.drone import GRAVITY
    from morphology_features import morph_features, _compute_twr
    from prior_controller import N_GAINS
    cfg = DroneConfiguration(props)
    mass = float(cfg.mass); inertia = np.asarray(cfg.inertia_matrix, dtype=np.float64)
    params = derive_reference_params(
        propellers=props, mass=mass, inertia=inertia,
        prop_size=mba.PROP_SIZE, gravity=GRAVITY,
    )
    return {
        "propellers": props, "mass": mass, "inertia": inertia,
        "prop_size": mba.PROP_SIZE,
        "twr": float(_compute_twr(params, mba.N_MOTORS, mass, GRAVITY)),
        "cmaes_params": np.zeros(mba.N_MOTORS + N_GAINS, dtype=np.float32),
        "morph_features": morph_features(
            props, mass=mass, inertia=inertia, prop_size=mba.PROP_SIZE,
        ).astype(np.float32),
        "_blueprint": bp,
    }


# ---------------------------------------------------------------- title card

def title_card(text: str, subtitle: str, width: int, height: int) -> np.ndarray:
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_facecolor("black")
    ax.text(0.5, 0.6, text, color="white", ha="center", va="center",
            fontsize=22, weight="bold")
    ax.text(0.5, 0.42, subtitle, color="#bbbbbb", ha="center", va="center",
            fontsize=13)
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.renderer.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


# ---------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", action="store_true",
                        help="Interactive MuJoCo viewer, one morph at a time")
    parser.add_argument("--mode", choices=["az", "pitch"], default="az",
                        help="Perturbation axis for the sequence")
    parser.add_argument("--out", type=Path, default=None,
                        help="MP4 path (default: morph_break_out/<mode>/policy_across_morphs.mp4)")
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--title-seconds", type=float, default=1.2)
    args = parser.parse_args()
    mode_dir = OUT / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if args.out is None:
        args.out = mode_dir / "policy_across_morphs.mp4"

    if not POLICY.exists():
        raise FileNotFoundError(f"Trained policy missing: {POLICY}. "
                                "Run 40_morph_break_analysis.py train first.")
    print(f"[viz] loading policy {POLICY}")
    model = PPO.load(str(POLICY), device="cpu")

    base_g = mba.canonical_genome()
    sequence = morph_sequence(mode=args.mode)

    print(f"[viz] mode={args.mode}  rolling out policy per morph …")
    rollouts = []
    for name, offsets in sequence:
        morph = build_morph_with_bp(base_g, offsets, mode=args.mode)
        pos_ned, euler_ned = rollout_policy(model, morph)
        drift = float(np.linalg.norm(pos_ned[-1] - HOVER_TARGET_NED))
        pos_enu, quat_enu = _ned_to_enu(pos_ned, euler_ned)
        rollouts.append((name, morph, pos_enu, quat_enu, drift))
        print(f"    {name:35s}  T={len(pos_ned):4d}  final_drift={drift:.3f} m  "
              f"mass={morph['mass']:.3f} kg  twr={morph['twr']:.2f}")

    if args.view:
        from mujoco import viewer as _mj_viewer
        for name, morph, pos_enu, quat_enu, drift in rollouts:
            print(f"\n[viewer] {name}   (close window to advance)")
            model_mj, data_mj = build_scene(morph, pos_enu[0])
            with _mj_viewer.launch_passive(model_mj, data_mj) as viewer:
                idx = 0
                while viewer.is_running() and idx < len(pos_enu):
                    t0 = _time.time()
                    data_mj.qpos[0:3] = pos_enu[idx]
                    data_mj.qpos[3:7] = quat_enu[idx]
                    mujoco.mj_forward(model_mj, data_mj)
                    viewer.sync()
                    slack = args.dt - (_time.time() - t0)
                    if slack > 0:
                        _time.sleep(slack)
                    idx += 1
        print("Done.")
        return

    # MP4 mode
    args.out.parent.mkdir(parents=True, exist_ok=True)
    recorder = VideoRecorder(
        file_name=args.out.stem, output_folder=args.out.parent,
        width=args.width, height=args.height, fps=args.fps,
    )
    steps_per_frame = max(1, int(round(1.0 / (args.fps * args.dt))))
    title_frames = int(round(args.title_seconds * args.fps))

    t_render = _time.time()
    for name, morph, pos_enu, quat_enu, drift in rollouts:
        subtitle = (f"mass={morph['mass']:.3f} kg   twr={morph['twr']:.2f}   "
                    f"final drift = {drift:.2f} m")
        card = title_card(name, subtitle, args.width, args.height)
        for _ in range(title_frames):
            recorder.write(frame=card)

        model_mj, data_mj = build_scene(morph, pos_enu[0])
        with mujoco.Renderer(model_mj,
                             width=args.width, height=args.height) as renderer:
            for idx in range(0, len(pos_enu), steps_per_frame):
                data_mj.qpos[0:3] = pos_enu[idx]
                data_mj.qpos[3:7] = quat_enu[idx]
                mujoco.mj_forward(model_mj, data_mj)
                renderer.update_scene(data_mj)
                recorder.write(frame=renderer.render())
    recorder.release()
    print(f"[viz] rendered in {_time.time() - t_render:.1f}s")
    print(f"[viz] mp4 -> {args.out}")


if __name__ == "__main__":
    main()
