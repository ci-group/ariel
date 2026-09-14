"""P4 — Johannink et al. 2019, *Residual Reinforcement Learning for Robot Control*.

Ports two mechanisms from the paper:

1. **α ablation** over the residual-blend weight
   ``total = effort_to_action(prior_effort + α · residual)``. Sweep
   α ∈ {0.05, 0.10, 0.20, 0.40, 0.80, 1.0-no-prior} on the canonical hex
   for two tasks (hover, figure8) with a small fixed budget per cell to
   trace the reward-vs-α curve.
2. **Residual warm-up (α anneal 0 → target)**: for the first
   ``warmup_frac`` of training the residual is scaled down so the prior
   dominates early exploration, matching the paper's near-zero
   initialisation of the residual policy. Compared against fixed-α.

Wraps `ResidualDroneEnv` from `envs/residual_drone_env.py`; only introduces
a `NoPriorResidualEnv` subclass (α=1.0 with the prior zeroed) and an
`AlphaAnnealCallback`. No modifications to the base env or `37_*`.

Usage:
    uv run examples/spear/library/paper_impls/p4_johannink2019_residual.py --smoke
    uv run examples/spear/library/paper_impls/p4_johannink2019_residual.py \\
        --steps 300_000 --alphas 0.05 0.1 0.2 0.4 0.8 nop
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from common import (
    dump_config, dump_results, load_40, load_37, new_out_dir,
)

# ─────────────────────────────────────────────────────────────────────────────
# Environments
# ─────────────────────────────────────────────────────────────────────────────

# ResidualDroneEnv is loaded via 37 so envs.residual_drone_env is on sys.path.
_t37 = load_37()
ResidualDroneEnv = _t37.ResidualDroneEnv


class NoPriorResidualEnv(ResidualDroneEnv):
    """`ResidualDroneEnv` with the analytical + CMA prior zeroed out.

    The mixing becomes ``total = effort_to_action(α · residual)`` — plain
    PPO on the effort action space. This is the "no-prior" cell in the α
    sweep. Everything else (task setup, morph features, obs layout) is
    identical to the parent.
    """

    def step_async(self, residual_actions: np.ndarray) -> None:
        residual = torch.as_tensor(
            residual_actions, device=self.dev, dtype=self.dtype,
        )
        # Zero the prior; only the residual survives the mixing.
        effort_zero = torch.zeros_like(residual)
        total_action = self.prior.effort_to_action(effort_zero + self.alpha * residual)
        self.prev_actions_t = self.actions_t.clone()
        self.actions_t = total_action


# ─────────────────────────────────────────────────────────────────────────────
# Warm-up anneal
# ─────────────────────────────────────────────────────────────────────────────

class AlphaAnnealCallback(BaseCallback):
    """Linearly anneal env.alpha from 0 → target over `warmup_steps`.

    Applies to every VecEnv worker on each rollout end (cheap: one attribute
    write per worker). After warm-up completes, alpha stays at target.
    """

    def __init__(self, target_alpha: float, warmup_steps: int):
        super().__init__(verbose=0)
        self.target = float(target_alpha)
        self.warmup = max(1, int(warmup_steps))

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        frac = min(1.0, self.num_timesteps / self.warmup)
        current = frac * self.target
        for env in self.model.env.envs if hasattr(self.model.env, "envs") \
                else [self.model.env]:
            _set_alpha_recursive(env, current)


def _set_alpha_recursive(env, alpha: float) -> None:
    """Walk VecEnv wrappers to reach the underlying ResidualDroneEnv."""
    obj = env
    while hasattr(obj, "venv"):
        obj = obj.venv
    if hasattr(obj, "alpha"):
        obj.alpha = float(alpha)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical-hex morph (from 40_morph_break_analysis)
# ─────────────────────────────────────────────────────────────────────────────

def canonical_morph() -> dict:
    m40 = load_40()
    return m40.genome_to_morph(m40.canonical_genome())


# ─────────────────────────────────────────────────────────────────────────────
# Training + evaluation for one α cell
# ─────────────────────────────────────────────────────────────────────────────

def make_env(morph: dict, task: str, alpha: float, no_prior: bool,
             num_envs: int, seed: int, device: str, max_steps: int):
    cls = NoPriorResidualEnv if no_prior else ResidualDroneEnv
    return cls(
        morph, task=task, alpha=alpha,
        num_envs=num_envs, max_steps=max_steps,
        device=device, seed=seed,
    )


def eval_one_env(env, model, n_eval_steps: int) -> dict:
    """Roll out for n_eval_steps; return mean_ep_reward and mean gates_passed."""
    obs = env.reset()
    cur_r = np.zeros(env.num_envs, dtype=np.float64)
    prev_g = np.zeros(env.num_envs, dtype=np.int64)
    cur_g = np.zeros(env.num_envs, dtype=np.int64)
    ep_r, ep_g = [], []
    total_g = 0
    for _ in range(n_eval_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, dones, infos = env.step(action)
        cur_r += r
        for i, info in enumerate(infos):
            gp = info.get("num_gates_passed", None)
            if gp is not None:
                cur_g[i] = int(np.asarray(gp)[0])
            delta = cur_g[i] - prev_g[i]
            if delta > 0:
                total_g += delta
            prev_g[i] = cur_g[i]
        for i, done in enumerate(dones):
            if done:
                ep_r.append(float(cur_r[i])); ep_g.append(int(cur_g[i]))
                cur_r[i] = 0.0; cur_g[i] = 0; prev_g[i] = 0
    return {
        "mean_ep_reward": float(np.mean(ep_r)) if ep_r else float("nan"),
        "mean_ep_gates":  float(np.mean(ep_g)) if ep_g else 0.0,
        "n_ep": len(ep_r),
        "gates_per_sec":  total_g / max(n_eval_steps * 0.01, 1e-9),
    }


def train_cell(alpha, no_prior: bool, task: str, args, seed: int) -> dict:
    label = "nop" if no_prior else f"a{alpha:.2f}"
    print(f"\n=== [P4] task={task:>8}  alpha={alpha}  no_prior={no_prior}  seed={seed}  "
          f"warmup={'yes' if args.warmup_frac > 0 else 'no'} ===")
    morph = canonical_morph()

    raw = make_env(morph, task=task, alpha=alpha, no_prior=no_prior,
                   num_envs=args.num_envs, seed=seed,
                   device=args.device, max_steps=args.max_steps)
    env = VecNormalize(raw, norm_obs=True, norm_reward=True,
                       clip_obs=10.0, clip_reward=10.0, gamma=0.99)
    model = PPO(
        "MlpPolicy", env,
        policy_kwargs=dict(net_arch=[128, 128]),
        n_steps=args.n_steps,
        batch_size=max((args.n_steps * args.num_envs) // 8, 64),
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
        max_grad_norm=0.5, seed=seed, device=args.device, verbose=0,
    )
    callbacks = []
    if args.warmup_frac > 0.0 and not no_prior:
        callbacks.append(AlphaAnnealCallback(
            target_alpha=alpha,
            warmup_steps=int(args.warmup_frac * args.steps),
        ))
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks or None)
    train_s = time.time() - t0

    # Eval on a fresh env (same morph/task/alpha, deterministic).
    eval_raw = make_env(morph, task=task, alpha=alpha, no_prior=no_prior,
                        num_envs=4, seed=seed + 10_000,
                        device=args.device, max_steps=args.max_steps)
    eval_env = VecNormalize(eval_raw, training=False, norm_obs=env.norm_obs,
                            norm_reward=False, clip_obs=env.clip_obs)
    eval_env.obs_rms = env.obs_rms
    stats = eval_one_env(eval_env, model, n_eval_steps=args.eval_steps)
    stats.update({
        "task": task, "alpha": alpha, "no_prior": no_prior, "seed": seed,
        "label": label, "train_seconds": train_s,
        "warmup_frac": args.warmup_frac,
    })
    print(f"[eval {task} {label}] r={stats['mean_ep_reward']:+7.3f}  "
          f"gates/ep={stats['mean_ep_gates']:5.2f}  n_ep={stats['n_ep']}  "
          f"gps={stats['gates_per_sec']:.2f}  train={train_s:.0f}s")
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tasks", nargs="+", default=["hover", "figure8"],
                   choices=list(_t37.TASK_NAMES))
    p.add_argument("--alphas", nargs="+", default=["0.05", "0.1", "0.2", "0.4", "0.8", "nop"],
                   help="numeric α values plus 'nop' for the no-prior cell")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="if >0, anneal α from 0→target over that fraction of steps")
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--max-steps", type=int, default=1200)
    p.add_argument("--eval-steps", type=int, default=1500)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="cpu")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 20_000
        args.num_envs = 4
        args.n_steps = 256
        args.max_steps = 400
        args.eval_steps = 600
        args.alphas = ["0.1", "0.4", "nop"]
        args.tasks = ["hover"]

    cells = []
    for a in args.alphas:
        if a == "nop":
            cells.append((1.0, True, "nop"))
        else:
            cells.append((float(a), False, f"a{float(a):.2f}"))

    out_dir = new_out_dir("p4_johannink2019_residual")
    dump_config(out_dir, vars(args))
    print(f"[P4] out_dir: {out_dir}")

    all_rows: list[dict] = []
    for task in args.tasks:
        for alpha, no_prior, _ in cells:
            for seed in args.seeds:
                all_rows.append(train_cell(alpha, no_prior, task, args, seed))

    # Aggregate: per (task, alpha_label) mean/std over seeds.
    def _key(r):
        return (r["task"], "nop" if r["no_prior"] else f"a{r['alpha']:.2f}")
    agg: dict[tuple, dict] = {}
    for row in all_rows:
        k = _key(row)
        agg.setdefault(k, []).append(row)
    summary = {
        f"{t}/{lbl}": {
            "mean_reward": float(np.mean([r["mean_ep_reward"] for r in rows])),
            "std_reward":  float(np.std([r["mean_ep_reward"] for r in rows])),
            "mean_gates":  float(np.mean([r["mean_ep_gates"] for r in rows])),
            "n_seeds": len(rows),
        }
        for (t, lbl), rows in agg.items()
    }
    dump_results(out_dir, {"per_run": all_rows, "summary": summary})

    print("\n=== [P4] α curve ===")
    for task in args.tasks:
        print(f"  task={task}")
        for alpha, no_prior, lbl in cells:
            key = f"{task}/{lbl}"
            if key not in summary:
                continue
            s = summary[key]
            print(f"    α={lbl:>7}: reward={s['mean_reward']:+7.3f}±{s['std_reward']:.3f}  "
                  f"gates/ep={s['mean_gates']:5.2f}  (n={s['n_seeds']})")
    print(f"\n[P4] saved -> {out_dir}")


if __name__ == "__main__":
    main()
