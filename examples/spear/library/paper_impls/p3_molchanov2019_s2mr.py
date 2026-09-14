"""P3 — Molchanov et al. 2019, *Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors*.

Ports **morphology/dynamics randomization** for the residual controller.
Each VecEnv worker gets a freshly perturbed canonical hex with axis-aware σ
matched to the break-threshold study (`40_morph_break_analysis.py`):

    σ_az    ∈ {1°, 2°}         (single-arm cliff at ±2.5°)
    σ_pitch ∈ {5°, 10°, 15°}   (gradual; +22.5° single-arm tolerated)
    mass_scale ~ U(0.9, 1.1)   (Molchanov's dynamics-randomization list)
    tau_scale  ~ U(0.7, 1.3)   (motor lag)

Mass/τ randomization is applied post-construction by patching
`drone_sim.params` and rebuilding the compiled dynamics closure — same
approach as `_build_torch_dynamics` in `TorchDroneGateEnv`.

After training, run `40_morph_break_analysis.py sweep --suffix _p3` on the
saved policy to quantify how much the azimuth/pitch tolerance bands widen
vs the baseline `hover_policy_tuned.zip`.

Usage:
    uv run examples/spear/library/paper_impls/p3_molchanov2019_s2mr.py --smoke
    uv run examples/spear/library/paper_impls/p3_molchanov2019_s2mr.py \\
        --steps 2_000_000 --num-envs 12 --sigma-az-deg 2 --sigma-pitch-deg 10
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from common import dump_config, dump_results, load_37, load_40, new_out_dir

_t37 = load_37()
_t40 = load_40()

ResidualDroneEnv = _t37.ResidualDroneEnv
TASK_NAMES = _t37.TASK_NAMES


# ─────────────────────────────────────────────────────────────────────────────
# Sample a randomised morph
# ─────────────────────────────────────────────────────────────────────────────

def sample_perturbed_morph(rng: np.random.RandomState,
                           sigma_az_deg: float, sigma_pitch_deg: float) -> dict:
    """Return a `ResidualDroneEnv`-ready morph dict for a canonical hex with
    per-arm azimuth and pitch Gaussian noise. Feature vector is recomputed
    from the perturbed geometry (never frozen — the policy sees the morph)."""
    base = _t40.canonical_genome()
    az_offsets = rng.normal(
        0.0, math.radians(sigma_az_deg), size=_t40.N_MOTORS,
    ).astype(np.float32)
    pitch_offsets = rng.normal(
        0.0, math.radians(sigma_pitch_deg), size=_t40.N_MOTORS,
    ).astype(np.float32)
    genome = _t40.perturbed_genome(base, az_offsets, mode="az")
    genome = _t40.perturbed_genome(genome, pitch_offsets, mode="pitch")
    return _t40.genome_to_morph(genome)


def apply_dynamics_randomization(env: ResidualDroneEnv,
                                  mass_scale: float, tau_scale: float) -> None:
    """Post-construction perturbation of mass & motor-lag τ.

    Multiplies the compiled dynamics params (`k_w = k_f/m`, `tau`) in place
    and rebuilds the closure. Prior is left untouched — a mismatched prior
    is part of Molchanov's real-world transfer challenge and the residual
    should compensate.
    """
    p = env.drone_sim.params
    p["k_w"] = float(p["k_w"]) / float(mass_scale)     # heavier → less accel
    p["tau"] = float(p["tau"]) * float(tau_scale)      # laggier motors
    from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics
    env._dynamics = _build_torch_dynamics(
        env.drone_sim.params, env.num_motors,
        env.drone_sim.g, env.dev, env.dtype,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Randomizing VecEnv
# ─────────────────────────────────────────────────────────────────────────────

class RandomizedMorphVecEnv(_t37.MorphRotatingVecEnv):
    """`MorphRotatingVecEnv` with a fresh perturbed morph per slot + mass/τ
    randomization. Task assignment is round-robin like the parent."""

    def __init__(self, num_slots: int, tasks: list[str], sigma_az_deg: float,
                 sigma_pitch_deg: float, mass_range: tuple[float, float],
                 tau_range: tuple[float, float], seed: int = 0,
                 device: str = "cpu", max_steps: int = 1200):
        rng = np.random.RandomState(seed)
        morphs = [sample_perturbed_morph(rng, sigma_az_deg, sigma_pitch_deg)
                  for _ in range(num_slots)]
        self._rand_meta = []
        for i, m in enumerate(morphs):
            m_scale = float(rng.uniform(*mass_range))
            t_scale = float(rng.uniform(*tau_range))
            self._rand_meta.append({
                "mass_scale": m_scale, "tau_scale": t_scale,
                "twr": float(m["twr"]), "mass": float(m["mass"]),
                "min_az_gap_deg": _t40.min_azimuth_gap_deg(m),
                "max_pitch_deg": _t40.max_pitch_deg(m),
            })
        super().__init__(morphs=morphs, tasks=tasks, alpha=None,
                         device=device, seed=seed, max_steps=max_steps)
        for env, meta in zip(self.envs, self._rand_meta):
            apply_dynamics_randomization(env, meta["mass_scale"], meta["tau_scale"])


# ─────────────────────────────────────────────────────────────────────────────
# Train + eval
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--sigma-az-deg", type=float, default=2.0,
                   help="azimuth noise σ per arm (deg) — cliff at ±2.5°")
    p.add_argument("--sigma-pitch-deg", type=float, default=10.0,
                   help="pitch noise σ per arm (deg) — gradual to ±22.5°")
    p.add_argument("--mass-range", type=float, nargs=2, default=(0.9, 1.1))
    p.add_argument("--tau-range", type=float, nargs=2, default=(0.7, 1.3))
    p.add_argument("--ent-start", type=float, default=0.005)
    p.add_argument("--ent-end", type=float, default=1e-4)
    p.add_argument("--log-std-init", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--eval-steps", type=int, default=1500)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.steps = 40_000
        args.num_envs = 5
        args.n_steps = 256
        args.eval_steps = 400

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = new_out_dir("p3_molchanov2019_s2mr")
    dump_config(out_dir, vars(args))
    print(f"[P3] out_dir: {out_dir}")

    # Round-robin task assignment across all 5 tasks.
    tasks = [TASK_NAMES[i % len(TASK_NAMES)] for i in range(args.num_envs)]
    print("[P3] task distribution: "
          + ", ".join(f"{t}={tasks.count(t)}" for t in TASK_NAMES))

    raw = RandomizedMorphVecEnv(
        num_slots=args.num_envs, tasks=tasks,
        sigma_az_deg=args.sigma_az_deg,
        sigma_pitch_deg=args.sigma_pitch_deg,
        mass_range=tuple(args.mass_range),
        tau_range=tuple(args.tau_range),
        seed=args.seed, device=args.device,
    )
    print("[P3] sampled morphs:")
    for i, (t, meta) in enumerate(zip(tasks, raw._rand_meta)):
        print(f"  slot {i}: task={t:>8} mass={meta['mass']:.3f}kg×{meta['mass_scale']:.2f}  "
              f"τ×{meta['tau_scale']:.2f}  twr={meta['twr']:.2f}  "
              f"min_az_gap={meta['min_az_gap_deg']:.1f}°  "
              f"max_pitch={meta['max_pitch_deg']:.1f}°")

    env = VecNormalize(raw, norm_obs=True, norm_reward=False, clip_obs=10.0)
    print(f"\n[P3 before training] random-residual rollout ({args.eval_steps} steps):")
    ep_r, ep_g, tg, nspt, mar = _t37._eval_per_task(
        env, raw, model=None, n_steps=args.eval_steps,
    )
    print(_t37._format_eval(ep_r, ep_g, tg, nspt, mar))

    batch_size = (args.n_steps * args.num_envs) // 8
    model = PPO(
        _t37.MTRLActorCriticPolicy, env,
        policy_kwargs=dict(log_std_init=args.log_std_init),
        n_steps=args.n_steps,
        batch_size=max(batch_size, 64),
        n_epochs=10, gamma=0.99, gae_lambda=0.95,
        learning_rate=3e-4, ent_coef=args.ent_start, clip_range=0.2,
        max_grad_norm=0.5, device=args.device, seed=args.seed, verbose=1,
    )
    ckpt_freq = max(args.n_steps, 250_000 // env.num_envs)
    callbacks = [
        _t37.EntCoefAnneal(args.ent_start, args.ent_end, args.steps),
        CheckpointCallback(
            save_freq=ckpt_freq, save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo", save_vecnormalize=True,
        ),
    ]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    print(f"[P3] trained {args.steps:,} steps in {elapsed:.0f}s "
          f"({args.steps/max(elapsed, 1e-9):.0f} sps)")

    print(f"\n[P3 after training] trained-policy rollout ({args.eval_steps} steps):")
    ep_r, ep_g, tg, nspt, mar = _t37._eval_per_task(
        env, raw, model=model, n_steps=args.eval_steps,
    )
    print(_t37._format_eval(ep_r, ep_g, tg, nspt, mar))

    mean_by_task = {t: (float(np.mean(ep_r[t])) if ep_r[t] else 0.0) for t in TASK_NAMES}
    from common import standard_metric
    metric = standard_metric(mean_by_task)
    print(f"\n[P3] standard metric = {metric:+.3f}")

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    dump_results(out_dir, {
        "metric": metric,
        "mean_reward_per_task": mean_by_task,
        "sampled_morphs": raw._rand_meta,
        "train_seconds": elapsed,
    })
    print(f"[P3] saved -> {out_dir}\n"
          f"[P3] follow-up: `40_morph_break_analysis.py sweep` reusing this policy\n"
          f"     to quantify break-threshold widening vs baseline.")


if __name__ == "__main__":
    main()
