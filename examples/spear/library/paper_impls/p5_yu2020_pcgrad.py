"""P5 — Yu et al. 2020, *Gradient Surgery for Multi-Task Learning* (PCGrad, NeurIPS 2020).

Ports PCGrad's actor-gradient projection into the PPO update used by
`37_train_residual_mtrl.py`. Per minibatch:

    1. Split the minibatch by task one-hot (obs layout documented in `37_*`).
    2. Compute a per-task PPO clipped-surrogate loss on the actor params
       (shared_encoder + task_encoders + actor_trunk + action_mean +
       log_std) with `torch.autograd.grad` — one backward per active task.
    3. For each ordered pair (i, j), if <g_i, g_j> < 0, project
       g_i ← g_i − (<g_i, g_j> / ||g_j||²) · g_j — order shuffled per
       minibatch, per Yu 2020 §3.
    4. Sum the projected per-task gradients as the actor gradient.
    5. Value + entropy losses on the WHOLE minibatch (surgery is actor-only,
       matching the paper's Meta-World/RL experiments); their gradients
       flow through the critics + shared paths normally.
    6. Optimizer step.

The gradient-cosine logger in `37_*` (`GradientCosineCallback`) triggered
this: it observed 5-task conflict statistics and pre-registered PCGrad as
the escalation. Toggle with `--pcgrad {on,off}`; a paired baseline run
(same seeds, `off`) is the intended comparison.

Usage:
    uv run examples/spear/library/paper_impls/p5_yu2020_pcgrad.py --smoke
    uv run examples/spear/library/paper_impls/p5_yu2020_pcgrad.py \\
        --pcgrad on --steps 5_000_000 --num-envs 20
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
BASE_OBS_DIM = _t37.BASE_OBS_DIM
NUM_TASKS = _t37.NUM_TASKS
TASK_NAMES = _t37.TASK_NAMES


def _extract_task_ids(obs: torch.Tensor) -> torch.Tensor:
    """The task one-hot lives at obs[:, BASE_OBS_DIM : BASE_OBS_DIM + NUM_TASKS]."""
    oh = obs[:, BASE_OBS_DIM:BASE_OBS_DIM + NUM_TASKS]
    return oh.argmax(dim=1)


def _actor_params(policy) -> list[torch.nn.Parameter]:
    return (
        list(policy.shared_encoder.parameters())
        + list(policy.task_encoders.parameters())
        + list(policy.actor_trunk.parameters())
        + list(policy.action_mean.parameters())
        + [policy.log_std]
    )


def _flatten_grads(grads: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.reshape(-1) for g in grads])


def _pcgrad_project(task_grads: dict[int, torch.Tensor],
                    rng: np.random.Generator) -> tuple[torch.Tensor, int, int]:
    """Yu 2020 §3 gradient surgery.

    For each task i, iterate the other tasks j in a shuffled order and, if
    <g_i, g_j> < 0, project g_i onto the normal plane of g_j:
        g_i ← g_i − <g_i, g_j>/||g_j||² · g_j.
    Returns the summed projected gradient, plus (num_conflicts, num_pairs).
    """
    if not task_grads:
        return None, 0, 0
    task_ids = list(task_grads.keys())
    projected = {i: task_grads[i].clone() for i in task_ids}
    n_conflict = 0
    n_pairs = 0
    for i in task_ids:
        order = list(task_ids)
        rng.shuffle(order)
        for j in order:
            if j == i:
                continue
            n_pairs += 1
            g_j = task_grads[j]
            dot = torch.dot(projected[i], g_j)
            if dot.item() < 0.0:
                n_conflict += 1
                projected[i] = projected[i] - dot / (g_j.pow(2).sum() + 1e-12) * g_j
    summed = torch.stack(list(projected.values()), dim=0).sum(dim=0)
    # Normalize by number of tasks so magnitude matches the unmodified sum
    # (paper does this to keep the effective step size comparable).
    summed = summed / len(task_ids)
    return summed, n_conflict, n_pairs


def _unflatten_and_set(params: list[torch.nn.Parameter], flat: torch.Tensor):
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = flat[idx:idx + n].reshape_as(p).clone()
        idx += n


# ─────────────────────────────────────────────────────────────────────────────
# PCGrad PPO
# ─────────────────────────────────────────────────────────────────────────────

class PCGradPPO(PPO):
    """PPO whose actor update uses PCGrad across task minibatches."""

    def __init__(self, *args, pcgrad: bool = True, log_every: int = 250_000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.pcgrad = bool(pcgrad)
        self._pc_rng = np.random.default_rng(int(kwargs.get("seed", 0) or 0))
        self._pc_log_every = int(log_every)
        self._pc_next_log = int(log_every)
        self._pc_ep_conflicts = 0
        self._pc_ep_pairs = 0

    def train(self) -> None:
        if not self.pcgrad:
            return super().train()

        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        clip_range_vf = (self.clip_range_vf(self._current_progress_remaining)
                         if self.clip_range_vf is not None else None)

        actor_ps = _actor_params(self.policy)
        pg_losses, v_losses, ent_losses, kls, clip_fracs = [], [], [], [], []
        n_conflict = 0
        n_pairs = 0

        for _ in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                obs = rollout_data.observations
                actions = rollout_data.actions
                old_log_prob = rollout_data.old_log_prob
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                returns = rollout_data.returns
                old_values = rollout_data.old_values

                # ── per-task actor gradients on THIS minibatch ────────────
                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
                values_flat = values.flatten()
                ratio = torch.exp(log_prob - old_log_prob)
                pl1 = advantages * ratio
                pl2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss_full = -torch.min(pl1, pl2)         # (B,)

                task_ids = _extract_task_ids(obs).detach().cpu().numpy()
                task_grads: dict[int, torch.Tensor] = {}
                for ti in np.unique(task_ids):
                    mask = torch.as_tensor(task_ids == ti, device=obs.device)
                    if mask.sum() < 2:
                        continue
                    l_i = policy_loss_full[mask].mean()
                    grads = torch.autograd.grad(
                        l_i, actor_ps, retain_graph=True, allow_unused=True,
                    )
                    zeros_like = lambda p: torch.zeros_like(p)
                    grads = [g if g is not None else zeros_like(p)
                             for g, p in zip(grads, actor_ps)]
                    task_grads[int(ti)] = _flatten_grads(grads).detach()

                projected, nc, np_ = _pcgrad_project(task_grads, self._pc_rng)
                n_conflict += nc
                n_pairs += np_

                # ── value + entropy on the full minibatch (unsurgeried) ──
                self.policy.optimizer.zero_grad()
                # Re-evaluate with fresh graph to avoid stale detached values
                # from the actor gradient pass above.
                values2, log_prob2, entropy2 = self.policy.evaluate_actions(obs, actions)
                values2_flat = values2.flatten()
                if clip_range_vf is None:
                    v_pred = values2_flat
                else:
                    v_pred = old_values + torch.clamp(
                        values2_flat - old_values, -clip_range_vf, clip_range_vf,
                    )
                value_loss = nn.functional.mse_loss(returns, v_pred)
                entropy_loss = -(entropy2.mean() if entropy2 is not None
                                 else -log_prob2.mean())
                loss_vc = self.vf_coef * value_loss + self.ent_coef * entropy_loss
                loss_vc.backward()

                # Overwrite actor grads with the projected ones. `.grad` was
                # populated by loss_vc.backward() — for actor params those
                # partial contributions (entropy on actor's log_std) are
                # replaced; for shared / critic paths they persist.
                if projected is not None:
                    _unflatten_and_set(actor_ps, projected)

                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm,
                )
                self.policy.optimizer.step()

                pg_losses.append(float(policy_loss_full.mean().item()))
                v_losses.append(float(value_loss.item()))
                ent_losses.append(float(entropy_loss.item()))
                with torch.no_grad():
                    lr = log_prob - old_log_prob
                    kls.append(float(((torch.exp(lr) - 1) - lr).mean().item()))
                    clip_fracs.append(float((torch.abs(ratio - 1) > clip_range).float().mean().item()))

        self.logger.record("train/policy_gradient_loss", float(np.mean(pg_losses)))
        self.logger.record("train/value_loss", float(np.mean(v_losses)))
        self.logger.record("train/entropy_loss", float(np.mean(ent_losses)))
        self.logger.record("train/approx_kl", float(np.mean(kls)))
        self.logger.record("train/clip_fraction", float(np.mean(clip_fracs)))
        self.logger.record("train/pcgrad_conflict_frac",
                            n_conflict / max(n_pairs, 1))
        self._pc_ep_conflicts += n_conflict
        self._pc_ep_pairs += n_pairs
        if self.num_timesteps >= self._pc_next_log:
            frac = self._pc_ep_conflicts / max(self._pc_ep_pairs, 1)
            print(f"[pcgrad @ {self.num_timesteps:,}] conflict "
                  f"fraction = {frac:.3f}  ({self._pc_ep_conflicts}/{self._pc_ep_pairs})",
                  flush=True)
            self._pc_ep_conflicts = 0
            self._pc_ep_pairs = 0
            self._pc_next_log += self._pc_log_every


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

def _load_morphs(library_path: Path, single_seed: int | None = None) -> list[dict]:
    all_morphs = _t37._load_morph_library(library_path, n_morphs=10**9)
    if single_seed is not None:
        all_morphs = [m for m in all_morphs if m["morph_seed"] == single_seed]
        if not all_morphs:
            raise SystemExit(f"morph_seed {single_seed} not found")
    return all_morphs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library", default="__data__/hex_library/v1/library.npz")
    p.add_argument("--pcgrad", choices=["on", "off"], default="on")
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--morph-seed", type=int, default=None,
                   help="restrict to a single morph_seed (debug)")
    p.add_argument("--ent-start", type=float, default=0.005)
    p.add_argument("--ent-end", type=float, default=1e-4)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-steps", type=int, default=1500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        # PCGrad is ~2x slower than baseline PPO (per-task backward inside
        # every minibatch). Keep smoke tiny.
        args.steps = 6_000
        args.num_envs = 5
        args.n_steps = 128
        args.eval_steps = 200

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = new_out_dir(f"p5_yu2020_pcgrad", args.pcgrad)
    dump_config(out_dir, vars(args))
    print(f"[P5] out_dir: {out_dir}   pcgrad={args.pcgrad}")

    all_morphs = _load_morphs(Path(args.library), args.morph_seed)
    print(f"[P5] loaded {len(all_morphs)} morphs")
    morphs = [all_morphs[i % len(all_morphs)] for i in range(args.num_envs)]
    tasks = [TASK_NAMES[i % NUM_TASKS] for i in range(args.num_envs)]
    print("[P5] task distribution: "
          + ", ".join(f"{t}={tasks.count(t)}" for t in TASK_NAMES))

    raw = _t37.MorphRotatingVecEnv(
        morphs=morphs, tasks=tasks, alpha=None,
        device=args.device, seed=args.seed,
    )
    env = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)

    batch_size = (args.n_steps * args.num_envs) // 8
    model = PCGradPPO(
        _t37.MTRLActorCriticPolicy, env,
        pcgrad=(args.pcgrad == "on"),
        policy_kwargs=dict(log_std_init=args.log_std_init),
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
    print(f"[P5] trained {args.steps:,} steps in {elapsed:.0f}s")

    print(f"\n[P5 after training] rollout ({args.eval_steps} steps):")
    ep_r, ep_g, tg, nspt, mar = _t37._eval_per_task(
        env, raw, model=model, n_steps=args.eval_steps,
    )
    print(_t37._format_eval(ep_r, ep_g, tg, nspt, mar))
    means = {t: (float(np.mean(ep_r[t])) if ep_r[t] else 0.0) for t in TASK_NAMES}
    metric = standard_metric(means)
    print(f"[P5] metric={metric:+.3f}  pcgrad={args.pcgrad}")

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    dump_results(out_dir, {
        "metric": metric,
        "mean_reward_per_task": means,
        "pcgrad": args.pcgrad,
        "train_seconds": elapsed,
    })
    print(f"[P5] saved -> {out_dir}")


if __name__ == "__main__":
    main()
