"""P6 — Sodhani et al. 2021, *Multi-Task Reinforcement Learning with Context-based Representations* (CARE, ICML 2021).

Ports CARE's **context-conditioned attention over a mixture of task
encoders** into `MTRLActorCriticPolicy`. The current 37 policy hard-gates
one encoder by task one-hot:

    task_lat = Σ_k one_hot[k] · enc_k(task_obs)

CARE replaces one-hot with a learned softmax attention derived from a
task-context embedding:

    ctx      = context_encoder(task_one_hot)     # small MLP
    weights  = softmax(context_attention(ctx))   # (B, K)
    task_lat = Σ_k weights[..., k] · enc_k(task_obs)

The number of encoders K need not equal the number of tasks (default K =
NUM_TASKS = 5). Task-borrowing across encoders is the mechanism the paper
credits for the Meta-World gains — e.g. figure8 can learn to attend to
whichever encoder ends up specialising in banked flight, whether that's
its own or `circle`'s.

`--policy {onehot,care}` gives a paired comparison against the 37
baseline; K is `--n-experts`.

Usage:
    uv run examples/spear/library/paper_impls/p6_sodhani2021_care.py --smoke
    uv run examples/spear/library/paper_impls/p6_sodhani2021_care.py \\
        --policy care --n-experts 5 --steps 5_000_000 --num-envs 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from common import dump_config, dump_results, load_37, new_out_dir, standard_metric

_t37 = load_37()
MTRLActorCriticPolicy = _t37.MTRLActorCriticPolicy
BASE_OBS_DIM = _t37.BASE_OBS_DIM
TASK_OBS_DIM = _t37.TASK_OBS_DIM
NUM_TASKS = _t37.NUM_TASKS
TASK_NAMES = _t37.TASK_NAMES
ENCODER_LATENT = _t37.ENCODER_LATENT
ENCODER_HIDDEN = _t37.ENCODER_HIDDEN
CTX_DIM = 16   # small — one-hot embedding
CTX_HIDDEN = 32


# ─────────────────────────────────────────────────────────────────────────────
# CARE policy
# ─────────────────────────────────────────────────────────────────────────────

class CAREActorCriticPolicy(MTRLActorCriticPolicy):
    """Replaces one-hot gating with soft attention over `n_experts` encoders.

    Attention:
        ctx     = context_encoder(one_hot)                          # (B, CTX_DIM)
        logits  = context_attention(ctx)                            # (B, K)
        weights = softmax(logits / temperature)                     # (B, K)
        task_lat = Σ_k weights[..., k] · encoder_k(task_obs)        # (B, L)

    Critics: still per-task; gated by the ORIGINAL one-hot (Sodhani mixes
    encoders, not value heads — the paper's evaluation head is task-
    conditioned). This keeps the value function comparably calibrated to
    the baseline.
    """

    def __init__(self, *args, n_experts: int = NUM_TASKS,
                 attn_temperature: float = 1.0, **kwargs):
        # SB3 clones policy_kwargs into `self.policy_kwargs`; make sure
        # these two keys survive the constructor path.
        self.n_experts = int(n_experts)
        self.attn_temperature = float(attn_temperature)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Replace the fixed-count task_encoders with `n_experts` copies.
        self.task_encoders = nn.ModuleList([
            _t37._mlp(TASK_OBS_DIM, ENCODER_HIDDEN, ENCODER_LATENT, n_hidden=2)
            for _ in range(self.n_experts)
        ])
        # Small context MLP (input = task one-hot of size num_tasks).
        self.context_encoder = _t37._mlp(
            self.num_tasks, CTX_HIDDEN, CTX_DIM, n_hidden=1,
        )
        self.context_attention = nn.Linear(CTX_DIM, self.n_experts)
        # Refresh optimizer to include the new params.
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _attention_weights(self, one_hot: torch.Tensor) -> torch.Tensor:
        ctx = self.context_encoder(one_hot)
        logits = self.context_attention(ctx) / max(self.attn_temperature, 1e-6)
        return torch.softmax(logits, dim=-1)                             # (B, K)

    def _actor_latent(self, drone, task_obs, one_hot, morph):
        shared_lat = self.shared_encoder(torch.cat([drone, morph], dim=1))
        all_task_lats = torch.stack(
            [enc(task_obs) for enc in self.task_encoders], dim=1
        )                                                                # (B, K, L)
        weights = self._attention_weights(one_hot)                       # (B, K)
        task_lat = (all_task_lats * weights.unsqueeze(-1)).sum(dim=1)    # (B, L)
        return self.actor_trunk(torch.cat([shared_lat, task_lat], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Introspection helper — print the attention map per task once trained
# ─────────────────────────────────────────────────────────────────────────────

def print_attention_map(policy: CAREActorCriticPolicy) -> dict:
    """For each task, print softmax weights over the K encoders."""
    device = next(policy.parameters()).device
    weights_by_task = {}
    with torch.no_grad():
        for ti, t in enumerate(TASK_NAMES):
            oh = torch.zeros(1, NUM_TASKS, device=device)
            oh[0, ti] = 1.0
            w = policy._attention_weights(oh)[0].cpu().numpy()
            weights_by_task[t] = w.tolist()
            print(f"  {t:>12}: " + "  ".join(
                f"e{k}={w[k]:.2f}" for k in range(policy.n_experts)
            ))
    return weights_by_task


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="__data__/hex_library/v1/library.npz")
    p.add_argument("--policy", choices=["onehot", "care"], default="care")
    p.add_argument("--n-experts", type=int, default=NUM_TASKS)
    p.add_argument("--attn-temperature", type=float, default=1.0)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--ent-start", type=float, default=0.005)
    p.add_argument("--ent-end", type=float, default=1e-4)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--morph-seed", type=int, default=None)
    p.add_argument("--eval-steps", type=int, default=1500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 6_000
        args.num_envs = 5
        args.n_steps = 128
        args.eval_steps = 200

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = new_out_dir("p6_sodhani2021_care", args.policy)
    dump_config(out_dir, vars(args))
    print(f"[P6] out_dir: {out_dir}   policy={args.policy}  K={args.n_experts}")

    all_morphs = _t37._load_morph_library(Path(args.library), n_morphs=10**9)
    if args.morph_seed is not None:
        all_morphs = [m for m in all_morphs if m["morph_seed"] == args.morph_seed]
    morphs = [all_morphs[i % len(all_morphs)] for i in range(args.num_envs)]
    tasks = [TASK_NAMES[i % NUM_TASKS] for i in range(args.num_envs)]
    print("[P6] task distribution: "
          + ", ".join(f"{t}={tasks.count(t)}" for t in TASK_NAMES))

    raw = _t37.MorphRotatingVecEnv(
        morphs=morphs, tasks=tasks, alpha=None,
        device=args.device, seed=args.seed,
    )
    env = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if args.policy == "care":
        policy_cls = CAREActorCriticPolicy
        policy_kwargs = dict(
            log_std_init=args.log_std_init,
            n_experts=args.n_experts,
            attn_temperature=args.attn_temperature,
        )
    else:
        policy_cls = MTRLActorCriticPolicy
        policy_kwargs = dict(log_std_init=args.log_std_init)

    batch_size = (args.n_steps * args.num_envs) // 8
    model = PPO(
        policy_cls, env,
        policy_kwargs=policy_kwargs,
        n_steps=args.n_steps,
        batch_size=max(batch_size, 64),
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, ent_coef=args.ent_start, clip_range=0.2,
        max_grad_norm=0.5, device=args.device, seed=args.seed, verbose=1,
    )
    callbacks = [
        _t37.EntCoefAnneal(args.ent_start, args.ent_end, args.steps),
        CheckpointCallback(
            save_freq=max(args.n_steps, 250_000 // env.num_envs),
            save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo", save_vecnormalize=True,
        ),
    ]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    print(f"[P6] trained {args.steps:,} steps in {elapsed:.0f}s")

    print(f"\n[P6 after training] rollout ({args.eval_steps} steps):")
    ep_r, ep_g, tg, nspt, mar = _t37._eval_per_task(
        env, raw, model=model, n_steps=args.eval_steps,
    )
    print(_t37._format_eval(ep_r, ep_g, tg, nspt, mar))
    means = {t: (float(np.mean(ep_r[t])) if ep_r[t] else 0.0) for t in TASK_NAMES}
    metric = standard_metric(means)
    print(f"[P6] metric={metric:+.3f}  policy={args.policy}")

    attn_map = {}
    if args.policy == "care":
        print("\n[P6] learned attention (per task, over K encoders):")
        attn_map = print_attention_map(model.policy)

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    dump_results(out_dir, {
        "metric": metric,
        "mean_reward_per_task": means,
        "policy": args.policy,
        "n_experts": args.n_experts,
        "attention_map": attn_map,
        "train_seconds": elapsed,
    })
    print(f"[P6] saved -> {out_dir}")


if __name__ == "__main__":
    main()
