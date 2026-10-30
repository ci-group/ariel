"""Blueprint trajectory pipeline, step 3/3 — MuJoCo replay of the trained policy.

Loads a run directory from 42b_train_blueprint_traj.py (policy + VecNormalize
stats + the drawn track), rolls the policy out deterministically from the fixed
start in the torch dynamics env, then kinematically replays the trajectory on
the blueprint-built hex inside a MuJoCo scene (same pattern as
40c_visualize_policy_mujoco.py). The drawn path and its waypoints are rendered
in the scene: grey spheres = raw drawing, orange spheres = waypoint gates,
green sphere = start.

Usage:
    uv run examples/spear/library/42c_visualize_blueprint_traj.py             # latest run, MP4
    uv run examples/spear/library/42c_visualize_blueprint_traj.py --view      # interactive
    uv run examples/spear/library/42c_visualize_blueprint_traj.py --run <dir>
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder

sys.path.insert(0, str(Path(__file__).parent))
_spec = importlib.util.spec_from_file_location(
    "t42b", str(Path(__file__).parent / "42b_train_blueprint_traj.py"),
)
t42b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(t42b)

RUNS_ROOT = t42b.DATA_ROOT / "runs"


# ---------------------------------------------------------------------------
# NED -> ENU conversion (copied from 40c_visualize_policy_mujoco.py)
# ---------------------------------------------------------------------------

def _euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX (yaw->pitch->roll) intrinsic Euler -> quaternion (w,x,y,z)."""
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
    """ENU: x_ENU = y_NED, y_ENU = x_NED, z_ENU = -z_NED."""
    pos_enu = np.empty_like(pos_ned)
    pos_enu[:, 0] = pos_ned[:, 1]
    pos_enu[:, 1] = pos_ned[:, 0]
    pos_enu[:, 2] = -pos_ned[:, 2]
    quat_enu = np.empty((len(pos_ned), 4), dtype=np.float64)
    for i, (r, p, y) in enumerate(euler_ned):
        quat_enu[i] = _euler_zyx_to_quat(p, r, math.pi / 2.0 - y)
    return pos_enu, quat_enu


def _pt_ned_to_enu(p_ned) -> list[float]:
    return [float(p_ned[1]), float(p_ned[0]), float(-p_ned[2])]


# ---------------------------------------------------------------------------
# Rollout in the torch env
# ---------------------------------------------------------------------------

def rollout(run_dir: Path, device: str = "cpu"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    cfg = json.loads((run_dir / "config.json").read_text())
    track = t42b.load_track(run_dir / "trajectory.npz")

    raw = t42b.make_env(
        track, num_envs=1, seed=42, device=device,
        gates_ahead=int(cfg["gates_ahead"]),
        max_steps=int(cfg["max_steps"]),
        random_init=False,
    )
    env = VecNormalize.load(str(run_dir / "vecnormalize.pkl"), raw)
    env.training = False
    env.norm_reward = False
    model = PPO.load(str(run_dir / "policy.zip"), device=device)

    obs = env.reset()
    pos, euler = [], []
    gates = 0
    for _ in range(int(cfg["max_steps"])):
        action, _ = model.predict(obs, deterministic=True)
        env.step_async(action)
        obs, _r, dones, infos = env.step_wait()
        pos.append(raw.world_states[0, 0:3].cpu().numpy().copy())
        euler.append(raw.world_states[0, 6:9].cpu().numpy().copy())
        if bool(dones[0]):
            gates = int(infos[0]["num_gates_passed"][0])
            break
    else:
        gates = int(raw.num_gates_passed[0])
    return (np.asarray(pos, dtype=np.float64),
            np.asarray(euler, dtype=np.float64), gates, track)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def build_scene(track: dict, initial_pos_enu, max_path_markers: int = 150):
    bp, props = t42b.build_blueprint_and_propellers()
    mean_arm = float(np.mean(
        [np.linalg.norm(np.asarray(p["loc"])[:2]) for p in props]
    ))
    drone_spec = blueprint_to_mjspec(
        bp, motor_mass=0.01, arm_mass=0.034 * mean_arm, body_name="drone",
    )
    world = SimpleFlatWorld()
    world.spawn(
        drone_spec,
        position=(float(initial_pos_enu[0]),
                  float(initial_pos_enu[1]),
                  float(initial_pos_enu[2])),
        correct_collision_with_floor=False,
    )

    def _marker(name, pos_enu, size, rgba):
        body = world.spec.worldbody.add_body(
            name=name, pos=pos_enu, quat=[1, 0, 0, 0],
        )
        body.add_geom(
            name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0], rgba=rgba, contype=0, conaffinity=0,
        )

    # Raw drawing as a breadcrumb line of small grey spheres.
    raw_xy = track["raw_xy"]
    stride = max(1, len(raw_xy) // max_path_markers)
    for i, (x, y) in enumerate(raw_xy[::stride]):
        _marker(f"path_{i}", _pt_ned_to_enu([x, y, -track["altitude"]]),
                0.02, (0.6, 0.6, 0.6, 0.5))

    # Waypoint gates (orange) + start (green).
    for i, g in enumerate(track["gates_pos"]):
        _marker(f"wp_{i}", _pt_ned_to_enu(g), 0.06, (0.95, 0.55, 0.1, 0.9))
    _marker("start", _pt_ned_to_enu(track["start_pos"]),
            0.08, (0.1, 0.9, 0.1, 0.9))

    model_mj = world.spec.compile()
    data_mj = mujoco.MjData(model_mj)
    return model_mj, data_mj


def _track_camera(track: dict) -> mujoco.MjvCamera:
    g_enu = np.array([_pt_ned_to_enu(g) for g in track["gates_pos"]])
    center = g_enu.mean(axis=0)
    extent = float(np.linalg.norm(g_enu[:, :2] - center[:2], axis=1).max())
    cam = mujoco.MjvCamera()
    cam.lookat = center
    cam.distance = 2.0 * extent + 3.0
    cam.elevation = -40.0
    cam.azimuth = 90.0
    return cam


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _latest_run() -> Path:
    runs = sorted(d for d in RUNS_ROOT.iterdir()
                  if d.is_dir() and (d / "policy.zip").exists())
    if not runs:
        raise SystemExit(f"no runs with a policy.zip under {RUNS_ROOT} — "
                         "train first with 42b_train_blueprint_traj.py")
    return runs[-1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=None,
                   help="run directory (default: latest under runs/)")
    p.add_argument("--view", action="store_true",
                   help="interactive passive viewer instead of MP4")
    p.add_argument("--out", type=Path, default=None,
                   help="MP4 path (default: <run>/replay.mp4)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    args = p.parse_args()

    run_dir = args.run if args.run is not None else _latest_run()
    print(f"[viz] run: {run_dir}")

    pos_ned, euler_ned, gates, track = rollout(run_dir, device=args.device)
    n_g = len(track["gates_pos"])
    print(f"[viz] rollout: T={len(pos_ned)} steps  "
          f"waypoints passed {gates}/{n_g} ({100 * gates / n_g:.0f}%)")

    pos_enu, quat_enu = _ned_to_enu(pos_ned, euler_ned)
    model_mj, data_mj = build_scene(track, pos_enu[0])
    cam = _track_camera(track)

    if args.view:
        from mujoco import viewer as _mj_viewer
        print("[viz] interactive viewer (close window to exit)")
        with _mj_viewer.launch_passive(model_mj, data_mj) as viewer:
            idx = 0
            while viewer.is_running():
                data_mj.qpos[0:3] = pos_enu[idx]
                data_mj.qpos[3:7] = quat_enu[idx]
                mujoco.mj_forward(model_mj, data_mj)
                viewer.sync()
                time.sleep(args.dt)
                idx = (idx + 1) % len(pos_enu)   # loop the replay
        return

    out = args.out if args.out is not None else run_dir / "replay.mp4"
    recorder = VideoRecorder(
        file_name=out.stem, output_folder=out.parent,
        width=args.width, height=args.height, fps=args.fps,
    )
    steps_per_frame = max(1, int(round(1.0 / (args.fps * args.dt))))
    with mujoco.Renderer(model_mj, width=args.width,
                         height=args.height) as renderer:
        for idx in range(0, len(pos_enu), steps_per_frame):
            data_mj.qpos[0:3] = pos_enu[idx]
            data_mj.qpos[3:7] = quat_enu[idx]
            mujoco.mj_forward(model_mj, data_mj)
            renderer.update_scene(data_mj, camera=cam)
            recorder.write(frame=renderer.render())
    recorder.release()
    # VideoRecorder appends a UTC timestamp to the filename; rename to the
    # requested path so the printed location is accurate.
    written = sorted(out.parent.glob(f"{out.stem}_*{out.suffix}"))
    if written:
        written[-1].rename(out)
    print(f"[viz] mp4 -> {out}")


if __name__ == "__main__":
    main()
