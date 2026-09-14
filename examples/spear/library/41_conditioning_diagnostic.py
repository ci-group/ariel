"""Frozen-morph-features ablation on a trained residual-MTRL policy.

Loads an existing exp_NNN checkpoint (policy.zip + vecnormalize.pkl) built
by 37_train_residual_mtrl.py, then runs the standard 5-task eval TWICE:

  (a) TRUE   — obs[:, 31:53] contains the true morph_features for each env's
                assigned library morph (what training saw).
  (b) FROZEN — obs[:, 31:53] is overwritten with a single canonical vector
                (the mean over library morphs) for every env, every step.

If per-task rewards are ~identical between (a) and (b), the policy is not
using its morph-features input; the generalist plan needs an auxiliary loss
or a different conditioning mechanism.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper


def _import_37(worktree: Path):
    """Load the training module by absolute path so we reuse its
    MorphRotatingVecEnv, _load_morph_library, _select_held_out, _eval_per_task,
    and constants."""
    p = worktree / "examples/spear/library/37_train_residual_mtrl.py"
    spec = importlib.util.spec_from_file_location("mtrl37", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mtrl37"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FreezeMorphSlice(VecEnvWrapper):
    """Overrides morph_features slice [31:53] with a constant vector on every
    reset()/step_wait() return, BEFORE VecNormalize sees the obs."""
    def __init__(self, raw_env, mtrl37, frozen: np.ndarray):
        super().__init__(raw_env)
        self._start = mtrl37.BASE_OBS_DIM + mtrl37.NUM_TASKS  # 31
        self._end = self._start + mtrl37.MORPH_FEAT_DIM       # 53
        self._frozen = frozen.astype(np.float32)              # (22,)

    def _patch(self, obs):
        obs[:, self._start:self._end] = self._frozen
        return obs

    def reset(self):
        return self._patch(self.venv.reset())

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return self._patch(obs), rew, done, info


def build_raw(mtrl37, library: Path, num_envs: int, device: str, seed: int,
              held_out_seed: int, held_out: int, freeze_features: np.ndarray | None,
              hover_prior_only: bool):
    all_morphs = mtrl37._load_morph_library(library, n_morphs=10**9)
    held_idx = mtrl37._select_held_out(all_morphs, held_out, held_out_seed)
    train_morphs = [m for i, m in enumerate(all_morphs) if i not in set(held_idx)]
    morphs = [train_morphs[i % len(train_morphs)] for i in range(num_envs)]
    tasks = [mtrl37.TASK_NAMES[i % mtrl37.NUM_TASKS] for i in range(num_envs)]

    raw_env = mtrl37.MorphRotatingVecEnv(
        morphs=morphs, tasks=tasks, alpha=None,
        device=device, seed=seed, inner_batch=1,
        hover_prior_only=hover_prior_only,
    )
    if freeze_features is not None:
        raw_env = _FreezeMorphSlice(raw_env, mtrl37, freeze_features)
    return raw_env, train_morphs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worktree", default="/home/user/Desktop/EvoDevo/ariel/.claude/worktrees/autoresearch")
    p.add_argument("--policy", required=True, help="path to policy.zip")
    p.add_argument("--vn", required=True, help="path to vecnormalize.pkl")
    p.add_argument("--library", default="/home/user/Desktop/EvoDevo/ariel/__data__/hex_library/v1/library.npz")
    p.add_argument("--num-envs", type=int, default=20)
    p.add_argument("--eval-steps", type=int, default=3000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held-out", type=int, default=10)
    p.add_argument("--held-out-seed", type=int, default=0)
    p.add_argument("--no-hover-prior-only", action="store_true",
                   help="disable hover_prior_only (exp_071+ trained WITH it — leave off)")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    mtrl37 = _import_37(Path(args.worktree))

    # Compute the frozen canonical features vector = mean across library
    all_morphs = mtrl37._load_morph_library(Path(args.library), n_morphs=10**9)
    feats = np.stack([m["morph_features"] for m in all_morphs])  # (N, 22)
    frozen = feats.mean(axis=0).astype(np.float32)
    print(f"[freeze] canonical morph_features (mean over {len(all_morphs)}): "
          f"min={frozen.min():.3f} max={frozen.max():.3f} std_across_lib={feats.std(axis=0).mean():.3f}")

    for label, freeze in [("TRUE", None), ("FROZEN", frozen)]:
        print(f"\n{'='*70}\n[{label}]  morph_features = "
              f"{'true per-env' if freeze is None else 'mean canonical'}\n{'='*70}")
        raw_env, _ = build_raw(
            mtrl37, Path(args.library),
            num_envs=args.num_envs, device=args.device, seed=args.seed,
            held_out_seed=args.held_out_seed, held_out=args.held_out,
            freeze_features=freeze, hover_prior_only=(not args.no_hover_prior_only),
        )
        env = VecNormalize.load(args.vn, raw_env)
        env.training = False
        env.norm_reward = False
        model = PPO.load(args.policy, device=args.device)
        ep_r, ep_g, total_g, nspt, mar = mtrl37._eval_per_task(
            env, raw_env, model=model, n_steps=args.eval_steps,
        )
        print(mtrl37._format_eval(ep_r, ep_g, total_g, nspt, mar))
        m = lambda t: float(np.mean(ep_r[t])) if ep_r[t] else float("nan")
        metric = (m("hover") + 2*(m("figure8") + m("slalom") + m("shuttle-run") + m("circle"))) / 9
        print(f"weighted metric = {metric:.3f}  "
              f"[hover={m('hover'):.2f} fig8={m('figure8'):.2f} slalom={m('slalom'):.2f} "
              f"shuttle={m('shuttle-run'):.2f} circle={m('circle'):.2f}]")


if __name__ == "__main__":
    main()
