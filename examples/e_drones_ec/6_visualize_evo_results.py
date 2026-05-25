"""Visualise results from 5_drone_evo_rl_figure8.py.

Three outputs:
  1. Fitness-over-generations plot (best & mean per generation).
  2. DroneVisualizer 4-panel rendering of the best blueprint.
  3. MuJoCo video of the best individual flying the figure-8 with its PPO policy.

Usage
-----
Point at the directory produced by script 5 (auto-detects all files):

    uv run examples/e_drones_ec/6_visualize_evo_results.py \\
        --run-dir __data__/drone_evo_rl_figure8

Or supply explicit paths:

    uv run examples/e_drones_ec/6_visualize_evo_results.py \\
        --db        __data__/drone_evo_rl_figure8/database_20260525_145131.db \\
        --blueprint __data__/drone_evo_rl_figure8/best_blueprint_20260525_145131.json \\
        --policy    __data__/drone_evo_rl_figure8/best_policy_20260525_145131.zip

Add --no-show to save matplotlib figures without opening GUI windows.
Add --view to open a MuJoCo passive viewer instead of writing an MP4.
Add --no-video to skip the video step entirely.
"""
from __future__ import annotations

import argparse
import sqlite3
import time as _time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Visualise drone evo + PPO results")
parser.add_argument("--run-dir", default=None,
                    help="Directory written by script 5 (auto-detects DB and blueprint).")
parser.add_argument("--db", default=None, help="Explicit path to SQLite DB.")
parser.add_argument("--blueprint", default=None, help="Explicit path to blueprint JSON.")
parser.add_argument("--policy", default=None,
                    help="Explicit path to best_policy_*.zip (SB3 format).")
parser.add_argument("--out-dir", default=None,
                    help="Where to save figures and video (default: same dir as DB).")
parser.add_argument("--rollout-time", type=float, default=15.0,
                    help="PPO rollout duration for video in seconds (default 15).")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--device", default="cpu", help="Torch device for PPO (default cpu).")
parser.add_argument("--no-show", action="store_true",
                    help="Save figures without opening GUI windows.")
parser.add_argument("--no-video", action="store_true",
                    help="Skip the MuJoCo video step.")
parser.add_argument("--view", action="store_true",
                    help="Open MuJoCo passive viewer instead of writing MP4.")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

# Resolve paths ---------------------------------------------------------------

def _latest(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None

run_dir = Path(args.run_dir) if args.run_dir else None

db_path = Path(args.db) if args.db else (
    _latest(run_dir, "database_*.db") if run_dir else None
)
bp_path = Path(args.blueprint) if args.blueprint else (
    _latest(run_dir, "best_blueprint_*.json") if run_dir else None
)
policy_path = Path(args.policy) if args.policy else (
    _latest(run_dir, "best_policy_*.zip") if run_dir else None
)

if db_path is None or not db_path.exists():
    raise SystemExit(f"DB not found. Pass --db or --run-dir. (resolved: {db_path})")
if bp_path is None or not bp_path.exists():
    raise SystemExit(f"Blueprint not found. Pass --blueprint or --run-dir. (resolved: {bp_path})")

out_dir = Path(args.out_dir) if args.out_dir else db_path.parent
out_dir.mkdir(parents=True, exist_ok=True)

run_tag = db_path.stem.replace("database_", "")
print(f"DB         : {db_path}")
print(f"Blueprint  : {bp_path}")
print(f"Policy     : {policy_path or '(not found — video step will be skipped)'}")
print(f"Output dir : {out_dir}")

# ---------------------------------------------------------------------------
# 1. Fitness-over-generations plot
# ---------------------------------------------------------------------------

conn = sqlite3.connect(db_path)
rows = conn.execute(
    "SELECT time_of_birth, fitness_ FROM individual "
    "WHERE fitness_ IS NOT NULL ORDER BY time_of_birth"
).fetchall()
conn.close()

from collections import defaultdict
gen_fits: dict[int, list[float]] = defaultdict(list)
for gen, fit in rows:
    gen_fits[int(gen)].append(float(fit))

gens = sorted(gen_fits)
best_fit = [max(gen_fits[g]) for g in gens]
mean_fit = [float(np.mean(gen_fits[g])) for g in gens]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(gens, best_fit, "o-", color="steelblue", label="best")
ax.plot(gens, mean_fit, "s--", color="tomato", label="mean")
ax.set_xlabel("Generation")
ax.set_ylabel("Fitness (mean episode reward)")
ax.set_title("Drone morphology evolution — fitness history")
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()

fitness_png = out_dir / f"fitness_history_{run_tag}.png"
fig.savefig(fitness_png, dpi=150, bbox_inches="tight")
print(f"Saved: {fitness_png}")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# 2. Blueprint 4-panel visualisation
# ---------------------------------------------------------------------------

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer


def blueprint_to_cartesian_array(bp: DroneBlueprint) -> np.ndarray:
    rows: list[list[float]] = []
    for prop in blueprint_to_propellers(bp, convention="z_up"):
        x, y, z = prop["loc"]
        nx, ny, nz, spin = prop["dir"]
        roll = float(np.arcsin(np.clip(-ny, -1.0, 1.0)))
        pitch = float(np.arctan2(nx, nz))
        direction = 1.0 if spin == "cw" else 0.0
        rows.append([float(x), float(y), float(z), roll, pitch, 0.0, direction])
    return np.asarray(rows, dtype=float)


bp = DroneBlueprint.load_json(bp_path)
arr = blueprint_to_cartesian_array(bp)
print(bp.summary())

visualizer = DroneVisualizer()

# 4-panel blueprint view
fig, _ = visualizer.plot_blueprint(arr, title=f"Best drone blueprint ({run_tag})")
blueprint_png = out_dir / f"best_blueprint_4panel_{run_tag}.png"
fig.savefig(blueprint_png, dpi=150, bbox_inches="tight")
print(f"Saved: {blueprint_png}")
if not args.no_show:
    plt.show()
plt.close(fig)

# 3-D view
fig, _ = visualizer.plot_3d(arr, title=f"Best drone — 3D view ({run_tag})")
view3d_png = out_dir / f"best_blueprint_3d_{run_tag}.png"
fig.savefig(view3d_png, dpi=150, bbox_inches="tight")
print(f"Saved: {view3d_png}")
if not args.no_show:
    plt.show()
plt.close(fig)

print("Done (plots).")

# ---------------------------------------------------------------------------
# 3. MuJoCo video of the best individual flying the figure-8
# ---------------------------------------------------------------------------

if args.no_video:
    print("Video step skipped (--no-video).")
elif policy_path is None or not policy_path.exists():
    print("Video step skipped: no policy ZIP found.")
else:
    import mujoco
    from stable_baselines3 import PPO

    from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
    from ariel.simulation.environments import SimpleFlatWorld
    from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
    from ariel.utils.video_recorder import VideoRecorder

    print(f"\nLoading policy from {policy_path} …")
    propellers = blueprint_to_propellers(bp, convention="ned")

    env = DroneGateEnv(
        propellers=propellers,
        num_envs=1,
        device=args.device,
        dt=args.dt,
        seed=0,
    )
    model = PPO.load(str(policy_path), env=env, device=args.device)
    print("Policy loaded.")

    # -- Rollout --------------------------------------------------------------
    N = int(args.rollout_time / args.dt) + 1
    pos_ned = np.zeros((N, 3), dtype=np.float32)
    euler_log = np.zeros((N, 3), dtype=np.float32)

    obs = env.reset()
    for i in range(N):
        pos_ned[i] = env.world_states[0, 0:3]
        euler_log[i] = env.world_states[0, 6:9]
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        if dones[0]:
            N = i + 1
            pos_ned = pos_ned[:N]
            euler_log = euler_log[:N]
            break

    print(
        f"Rollout done: {N} steps ({N * args.dt:.1f}s)  "
        f"alt range (NED z): [{pos_ned[:, 2].min():.2f}, {pos_ned[:, 2].max():.2f}]"
    )

    # NED → ENU
    pos_enu = pos_ned.copy()
    pos_enu[:, 2] = -pos_enu[:, 2]

    # Euler → quaternion (w, x, y, z), then approximate NED→ENU correction
    phi, theta, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    quat_enu = np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        -(cr * sp * cy + sr * cp * sy),   # flip y component for NED→ENU
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)

    # -- Build MuJoCo model ---------------------------------------------------
    arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
    mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06

    drone_spec = blueprint_to_mjspec(
        bp,
        motor_mass=0.01,
        arm_mass=0.034 * mean_arm_len,
        body_name="evolved_quad",
    )
    world_mj = SimpleFlatWorld()
    world_mj.spawn(
        drone_spec,
        position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
        correct_collision_with_floor=False,
    )
    model_mj = world_mj.spec.compile()
    data_mj = mujoco.MjData(model_mj)
    model_mj.opt.timestep = args.dt

    def _set_pose(idx: int) -> None:
        p = pos_enu[idx]
        q = quat_enu[idx]
        data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
        data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        data_mj.qvel[:] = 0.0
        mujoco.mj_forward(model_mj, data_mj)

    # -- Render ---------------------------------------------------------------
    if args.view:
        import mujoco.viewer
        print("Launching MuJoCo passive viewer …")
        _set_pose(0)
        with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
            idx = 0
            while viewer.is_running() and idx < len(pos_enu):
                step_start = _time.time()
                _set_pose(idx)
                viewer.sync()
                slack = args.dt - (_time.time() - step_start)
                if slack > 0:
                    _time.sleep(slack)
                idx += 1
    else:
        mp4 = out_dir / f"evolved_figure8_{run_tag}.mp4"
        recorder = VideoRecorder(
            file_name=mp4.stem, output_folder=mp4.parent,
            width=720, height=540, fps=30,
        )
        steps_per_frame = max(1, int(round(1.0 / (recorder.fps * args.dt))))
        t_render = _time.time()
        with mujoco.Renderer(model_mj, width=recorder.width, height=recorder.height) as renderer:
            for idx in range(0, len(pos_enu), steps_per_frame):
                _set_pose(idx)
                renderer.update_scene(data_mj)
                recorder.write(frame=renderer.render())
        recorder.release()
        print(f"Rendered in {_time.time() - t_render:.2f}s")
        print(f"Video: {mp4}")

print("\nDone.")
