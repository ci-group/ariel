"""Visualize a trained PPO residual policy from a `37_train_residual_mtrl.py`
run, replayed per task in MuJoCo. Companion to `38_visualize_prior_per_task.py`
(which only shows the analytical prior with α=0); this one loads the saved
`policy.zip` + `vecnormalize.pkl` and rolls out the prior + α·residual that
PPO actually learned.

Run:
    # viewer, all tasks in sequence:
    uv run examples/spear/library/39_visualize_residual_policy.py \\
        --run-dir __data__/library_residual_mtrl/<TIMESTAMP> --view

    # one task, render to video:
    uv run examples/spear/library/39_visualize_residual_policy.py \\
        --run-dir __data__/library_residual_mtrl/<TIMESTAMP> \\
        --task figure8 --out __data__/residual_replay/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from envs.residual_drone_env import ResidualDroneEnv, TASK_NAMES  # noqa: E402

# Reuse helpers + classes from the sibling scripts so we render with exactly
# the same scene/NED→ENU pipeline as the prior-only baseline.
_38 = __import__("38_visualize_prior_per_task")
_37 = __import__("37_train_residual_mtrl")


def _rollout_policy(env: VecNormalize, raw_env: "_37.MorphRotatingVecEnv",
                    model: PPO, n_steps: int) -> dict:
    """Roll out the trained policy on one slot; log world state from the
    underlying ResidualDroneEnv (pre-normalization, NED frame)."""
    inner = raw_env.envs[0]
    obs = env.reset()
    pos_ned = np.zeros((n_steps + 1, 3), dtype=np.float32)
    euler = np.zeros((n_steps + 1, 3), dtype=np.float32)
    rewards = np.zeros(n_steps, dtype=np.float32)
    dones = np.zeros(n_steps, dtype=bool)
    gates_passed = np.zeros(n_steps, dtype=np.int64)
    n_resets = 0

    pos_ned[0] = inner.world_states[0, 0:3].cpu().numpy()
    euler[0] = inner.world_states[0, 6:9].cpu().numpy()
    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rews, ds, infos = env.step(action)
        pos_ned[i + 1] = inner.world_states[0, 0:3].cpu().numpy()
        euler[i + 1] = inner.world_states[0, 6:9].cpu().numpy()
        rewards[i] = float(rews[0])
        dones[i] = bool(ds[0])
        gp = infos[0].get("num_gates_passed", None)
        if gp is not None:
            gates_passed[i] = int(np.asarray(gp)[0]) if hasattr(gp, "__len__") else int(gp)
        if bool(ds[0]):
            n_resets += 1
    return dict(
        pos_ned=pos_ned, euler=euler, rewards=rewards, dones=dones,
        gates_passed=gates_passed, n_resets=n_resets,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="Output dir from 37_train_residual_mtrl.py "
                        "(must contain policy.zip + vecnormalize.pkl)")
    p.add_argument("--library", default="__data__/hex_library/v1/library.npz")
    p.add_argument("--morph-idx", type=int, default=0)
    p.add_argument("--task", default="all",
                   choices=("all",) + TASK_NAMES)
    p.add_argument("--rollout-time", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=None,
                   help="Override residual scale (default: per-task from env)")
    p.add_argument("--view", action="store_true")
    p.add_argument("--out", default="__data__/residual_replay")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    policy_path = run_dir / "policy.zip"
    vecnorm_path = run_dir / "vecnormalize.pkl"
    if not policy_path.exists() or not vecnorm_path.exists():
        raise FileNotFoundError(
            f"expected policy.zip + vecnormalize.pkl in {run_dir}"
        )

    morph = _38._load_morph(Path(args.library), args.morph_idx)
    print(f"morph idx={args.morph_idx}  prop={morph['prop_size']}  "
          f"twr={morph['twr']:.1f}  mass={morph['mass']:.3f}kg  "
          f"motors={len(morph['propellers'])}")

    task_list = list(TASK_NAMES) if args.task == "all" else [args.task]
    out_root = Path(args.out)
    n_steps = int(args.rollout_time / args.dt)

    for task in task_list:
        # One-slot VecEnv so the loaded VecNormalize stats (built on a 53d
        # obs space) match. Tasks are per-slot in MorphRotatingVecEnv so we
        # just build a fresh single-slot env per task.
        raw_env = _37.MorphRotatingVecEnv(
            morphs=[morph], tasks=[task], alpha=args.alpha,
            device=args.device, max_steps=n_steps + 50,
        )
        env = VecNormalize.load(str(vecnorm_path), raw_env)
        env.training = False
        env.norm_reward = False

        model = PPO.load(str(policy_path), device=args.device)

        log = _rollout_policy(env, raw_env, model, n_steps)

        n_gates = int(log["gates_passed"].max())
        n_dones = int(log["dones"].sum())
        total_r = float(log["rewards"].sum())
        z_min = float(log["pos_ned"][:, 2].min())
        z_max = float(log["pos_ned"][:, 2].max())
        print(
            f"\n=== {task} (PPO residual) ===\n"
            f"  flight time:  {args.rollout_time:.1f}s ({n_steps} steps)\n"
            f"  episodes:     {n_dones} (early terminations)\n"
            f"  gates passed: {n_gates}\n"
            f"  total reward: {total_r:+.2f} (post-VecNormalize scaling)\n"
            f"  altitude NED z: [{z_min:.2f}, {z_max:.2f}]  "
            f"(target -1.5 for hover)"
        )

        inner = raw_env.envs[0]
        gates_ned = inner.gate_pos_t.cpu().numpy()
        start_ned = inner.start_pos_t.cpu().numpy()
        model_mj = _38._build_scene(morph, gates_ned, start_ned)
        pos_enu, quat_enu = _38._ned_to_enu(log["pos_ned"], log["euler"])
        out_path = out_root / f"residual_{task}.mp4"
        _38._replay(model_mj, pos_enu, quat_enu, args.dt, args.view,
                    out_path, label=task)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
