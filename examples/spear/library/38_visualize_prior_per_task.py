"""Visualize the analytical hover prior alone on all 5 tasks (no PPO).

Runs `ResidualDroneEnv` with α=0 (residual ignored) for each task so the
prior is the *only* controller. Replays the resulting trajectory in
MuJoCo (viewer or rendered video). This is the "what does the prior get
us for free" baseline that the PPO residual is supposed to *improve* on.

Run:
    # viewer, all tasks in sequence:
    uv run examples/spear/library/38_visualize_prior_per_task.py --view

    # one specific task:
    uv run examples/spear/library/38_visualize_prior_per_task.py \\
        --task figure8 --view

    # video, all tasks, to a folder:
    uv run examples/spear/library/38_visualize_prior_per_task.py \\
        --task all --out __data__/prior_baseline/
"""

from __future__ import annotations

import argparse
import sys
import time as _time
from pathlib import Path

import mujoco
import numpy as np
import torch

from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec, blueprint_to_propellers,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.environments import SimpleFlatWorld

sys.path.insert(0, str(Path(__file__).parent))
from envs.residual_drone_env import ResidualDroneEnv, TASK_NAMES  # noqa: E402
from hex_sampler import sample_feasible  # noqa: E402

GRAVITY = 9.81


# ─────────────────────────────────────────────────────────────────────────────
# Morph + prior rollout
# ─────────────────────────────────────────────────────────────────────────────

def _load_morph(library_path: Path, idx: int) -> dict:
    """Pair a library row with its sampler counterpart to recover propellers."""
    d = np.load(library_path)
    seed = int(d["morph_seed"][idx])
    sampled = sample_feasible(200, seed=42, stratify=True)
    by_seed = {m.seed: m for m in sampled}
    if seed not in by_seed:
        raise RuntimeError(f"library seed {seed} not in re-sampled set")
    m = by_seed[seed]
    return {
        "propellers":     m.propellers,
        "mass":           float(m.mass),
        "inertia":        m.inertia,
        "prop_size":      int(m.prop_size),
        "twr":            float(m.twr),
        "cmaes_params":   d["cmaes_params"][idx].astype(np.float32),
        "morph_features": d["morph_features"][idx].astype(np.float32),
        "genome":         d["genome"][idx],
    }


def _rollout_prior(env: ResidualDroneEnv, n_steps: int) -> dict:
    """Roll out with zero residual (prior-only). Log per-step state."""
    env.reset()
    N = env.num_motors
    zero_res = np.zeros((1, N), dtype=np.float32)
    pos_ned = np.zeros((n_steps + 1, 3), dtype=np.float32)
    euler = np.zeros((n_steps + 1, 3), dtype=np.float32)
    rewards = np.zeros(n_steps, dtype=np.float32)
    dones = np.zeros(n_steps, dtype=bool)
    gates_passed = np.zeros(n_steps, dtype=np.int64)
    n_resets = 0

    pos_ned[0] = env.world_states[0, 0:3].cpu().numpy()
    euler[0] = env.world_states[0, 6:9].cpu().numpy()
    for i in range(n_steps):
        env.step_async(zero_res)
        _obs, r, d, info = env.step_wait()
        pos_ned[i + 1] = env.world_states[0, 0:3].cpu().numpy()
        euler[i + 1] = env.world_states[0, 6:9].cpu().numpy()
        rewards[i] = float(r[0])
        dones[i] = bool(d[0])
        gp = info[0].get("num_gates_passed", None)
        if gp is not None:
            gates_passed[i] = int(np.asarray(gp)[0]) if hasattr(gp, "__len__") else int(gp)
        if bool(d[0]):
            n_resets += 1
    return dict(
        pos_ned=pos_ned, euler=euler, rewards=rewards, dones=dones,
        gates_passed=gates_passed, n_resets=n_resets,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo scene + replay
# ─────────────────────────────────────────────────────────────────────────────

def _ned_to_enu(pos_ned: np.ndarray, euler: np.ndarray):
    """Project NED state to MuJoCo's ENU world for visualization."""
    pos_enu = pos_ned.copy()
    pos_enu[:, 2] = -pos_enu[:, 2]
    phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    quat_enu = np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        -(cr * sp * cy + sr * cp * sy),
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)
    return pos_enu, quat_enu


def _build_scene(morph: dict, gate_pos_ned: np.ndarray, start_pos_ned: np.ndarray):
    """Build the MuJoCo model: drone + gate markers (green spheres)."""
    bp = spherical_angular_to_blueprint(morph["genome"])
    # Get arm length for spec sizing
    try:
        arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]
        mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.1
    except Exception:
        mean_arm_len = 0.15
    drone_spec = blueprint_to_mjspec(
        bp, motor_mass=0.01, arm_mass=0.034 * mean_arm_len,
        body_name="prior_drone",
    )
    world = SimpleFlatWorld()
    start_enu = (
        float(start_pos_ned[0]),
        float(start_pos_ned[1]),
        float(-start_pos_ned[2]),
    )
    world.spawn(drone_spec, position=start_enu,
                correct_collision_with_floor=False)
    # Gate markers
    for i, g in enumerate(gate_pos_ned):
        body = world.spec.worldbody.add_body(
            name=f"gate_{i}", pos=[float(g[0]), float(g[1]), float(-g[2])],
        )
        body.add_geom(
            name=f"gate_geom_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.08, 0.0, 0.0],
            rgba=(0.1, 0.9, 0.1, 0.7),
            contype=0, conaffinity=0,
        )
    return world.spec.compile()


def _replay(model, pos_enu, quat_enu, dt: float, view: bool,
            out_path: Path | None, label: str):
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    def _set_pose(idx):
        data.qpos[0:3] = [float(pos_enu[idx, 0]), float(pos_enu[idx, 1]),
                          float(pos_enu[idx, 2])]
        data.qpos[3:7] = [float(quat_enu[idx, 0]), float(quat_enu[idx, 1]),
                          float(quat_enu[idx, 2]), float(quat_enu[idx, 3])]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

    if view:
        from mujoco import viewer as mj_viewer
        print(f"[{label}] Launching MuJoCo viewer — close window to advance")
        _set_pose(0)
        with mj_viewer.launch_passive(model, data) as viewer:
            idx = 0
            while viewer.is_running() and idx < len(pos_enu):
                t0 = _time.time()
                _set_pose(idx)
                viewer.sync()
                slack = dt - (_time.time() - t0)
                if slack > 0:
                    _time.sleep(slack)
                idx += 1
        return
    # Video
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = 1280, 720, 60
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 4.0
    cam.elevation = -20.0
    cam.azimuth = 45.0
    cam.lookat[:] = [float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])]

    total_time = (len(pos_enu) - 1) * dt
    n_frames = max(1, int(round(total_time * FPS)) + 1)
    frame_times = np.linspace(0.0, total_time, n_frames)
    print(f"[{label}] Rendering {n_frames} frames → {out_path}")
    frames = []
    with mujoco.Renderer(model, width=W, height=H) as renderer:
        for t in frame_times:
            sim_idx = min(int(round(t / dt)), len(pos_enu) - 1)
            _set_pose(sim_idx)
            cam.lookat[:] = [float(pos_enu[sim_idx, 0]),
                             float(pos_enu[sim_idx, 1]),
                             float(pos_enu[sim_idx, 2])]
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    try:
        import imageio.v3 as iio
        iio.imwrite(
            str(out_path), np.stack(frames, axis=0), fps=FPS,
            codec="libx264", quality=8, pixelformat="yuv420p",
            macro_block_size=1, ffmpeg_log_level="error",
        )
        print(f"[{label}] saved → {out_path}")
    except ImportError:
        from ariel.utils.video_recorder import VideoRecorder
        rec = VideoRecorder(file_name=out_path.stem,
                            output_folder=out_path.parent,
                            width=W, height=H, fps=FPS)
        for f in frames:
            rec.write(frame=f)
        rec.release()
        print(f"[{label}] saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library",
                   default="__data__/hex_library/v1/library.npz")
    p.add_argument("--morph-idx", type=int, default=0,
                   help="Index into the library (0 = highest-scoring "
                        "according to library ordering)")
    p.add_argument("--task", default="all",
                   choices=("all",) + TASK_NAMES,
                   help="One task or 'all' to loop through every task")
    p.add_argument("--rollout-time", type=float, default=10.0,
                   help="Seconds of simulated flight per task")
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--view", action="store_true",
                   help="Open MuJoCo viewer (default: render to video)")
    p.add_argument("--out", default="__data__/prior_baseline",
                   help="Output dir for videos (ignored if --view)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    morph = _load_morph(Path(args.library), args.morph_idx)
    print(f"morph idx={args.morph_idx}  seed={morph['cmaes_params'].shape}"
          f"  prop={morph['prop_size']}  twr={morph['twr']:.1f}  "
          f"mass={morph['mass']:.3f}kg  motors={len(morph['propellers'])}")

    task_list = list(TASK_NAMES) if args.task == "all" else [args.task]
    out_root = Path(args.out)
    n_steps = int(args.rollout_time / args.dt)

    for task in task_list:
        # Build env with α=0: residual is ignored, prior alone drives the
        # drone. ResidualDroneEnv copies the right GATE_CONFIGS + reward
        # shaping for this task from 27_v4, so what you see is exactly
        # what the training env feeds PPO before any residual is added.
        env = ResidualDroneEnv(
            morph, task=task, alpha=0.0, num_envs=1,
            max_steps=n_steps + 50, device=args.device,
        )
        log = _rollout_prior(env, n_steps)
        # Compact per-task readout
        n_gates = int(log["gates_passed"].max())
        n_dones = int(log["dones"].sum())
        total_r = float(log["rewards"].sum())
        z_min = float(log["pos_ned"][:, 2].min())
        z_max = float(log["pos_ned"][:, 2].max())
        print(
            f"\n=== {task} (prior-only, α=0) ===\n"
            f"  flight time:  {args.rollout_time:.1f}s "
            f"({n_steps} steps)\n"
            f"  episodes:     {n_dones} (resets/episode early-terminations)\n"
            f"  gates passed: {n_gates} (cumulative across episodes)\n"
            f"  total reward: {total_r:+.2f}\n"
            f"  altitude NED z: [{z_min:.2f}, {z_max:.2f}]  "
            f"(target -1.5 for hover)"
        )

        # Build scene & replay
        gates_ned = env.gate_pos_t.cpu().numpy()
        start_ned = env.start_pos_t.cpu().numpy()
        model = _build_scene(morph, gates_ned, start_ned)
        pos_enu, quat_enu = _ned_to_enu(log["pos_ned"], log["euler"])
        out_path = out_root / f"prior_only_{task}.mp4"
        _replay(model, pos_enu, quat_enu, args.dt, args.view,
                out_path, label=task)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
