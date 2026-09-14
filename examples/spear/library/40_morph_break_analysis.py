"""Morphology break-threshold analysis for a PPO hover specialist.

Question: how far can the arm azimuths of a standard (regular) hexacopter
be perturbed before a hover policy trained on the unperturbed morph breaks?

Protocol
--------
1. TRAIN: PPO hover specialist on the canonical hex (even 60-degree azimuths,
   uniform arm length, planar arms, vertical motors, alternating spin) using
   ResidualDroneEnv(task="hover", use_prior=False) — pure RL, no prior.
2. SWEEP A (single arm): rotate one arm's azimuth in fixed steps up to +/-45
   degrees; the motor azimuth follows the arm (collinear). Mass/inertia are
   recomputed for every perturbed morph, so dynamics are physically consistent.
3. SWEEP B (all arms): Gaussian-perturb all six azimuths with increasing
   sigma, several independent draws per sigma.
4. For each perturbed morph run N deterministic rollouts; an episode is a
   SUCCESS if it survives 600 steps AND ends within 0.5 m of the hover target.
   Break threshold = perturbation where success rate crosses 50%.

Fairness notes
--------------
- The policy's 22-d morph-feature obs slice is FROZEN at the canonical
  morph's values by default (--update-features to flip): the policy is blind
  to the perturbation, so the curves measure pure dynamics-mismatch
  tolerance, not obs-shift effects.
- Env max_steps is set above the 600-step probe so the final position can be
  read before any auto-reset corrupts it.

Run:
    uv run examples/spear/library/40_morph_break_analysis.py all --train-steps 2000000
    uv run examples/spear/library/40_morph_break_analysis.py sweep   # reuse saved policy
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.drone import GRAVITY

from morphology_features import morph_features, _compute_twr  # noqa: E402
from prior_controller import N_GAINS  # noqa: E402
from envs.residual_drone_env import ResidualDroneEnv  # noqa: E402

N_MOTORS = 6
ARM_LENGTH = 0.15
CORE_MASS = 0.5
PROP_SIZE = 2
EPISODE_STEPS = 600
SUCCESS_DRIFT_M = 0.5

OUT_DIR_DEFAULT = Path(__file__).parent / "morph_break_out"


# ---------------------------------------------------------------- morphs

def canonical_genome() -> np.ndarray:
    """Regular hex: even azimuths, planar arms, vertical motors, 3ccw+3cw."""
    az = np.radians(np.arange(N_MOTORS) * 60.0).astype(np.float32)
    g = np.zeros((N_MOTORS, 6), dtype=np.float32)
    g[:, 0] = ARM_LENGTH          # magnitude
    g[:, 1] = az                  # arm_az
    g[:, 2] = 0.0                 # arm_pitch (planar)
    g[:, 3] = az                  # motor_az (collinear)
    g[:, 4] = 0.0                 # motor_pitch (vertical thrust)
    g[:, 5] = np.array([0, 1, 0, 1, 0, 1], dtype=np.float32)  # spin
    return g


def genome_to_morph(genome: np.ndarray) -> dict:
    """Decode a genome into the morph dict ResidualDroneEnv expects.

    cmaes_params are zeros — unused with use_prior=False.
    """
    bp = spherical_angular_to_blueprint(
        genome, core_mass=CORE_MASS, propsize=PROP_SIZE,
    )
    propellers = blueprint_to_propellers(bp, convention="ned")
    cfg = DroneConfiguration(propellers)
    mass = float(cfg.mass)
    inertia = np.asarray(cfg.inertia_matrix, dtype=np.float64)
    params = derive_reference_params(
        propellers=propellers, mass=mass, inertia=inertia,
        prop_size=PROP_SIZE, gravity=GRAVITY,
    )
    twr = float(_compute_twr(params, N_MOTORS, mass, GRAVITY))
    feats = morph_features(
        propellers, mass=mass, inertia=inertia, prop_size=PROP_SIZE,
    ).astype(np.float32)
    return {
        "propellers":     propellers,
        "mass":           mass,
        "inertia":        inertia,
        "prop_size":      PROP_SIZE,
        "twr":            twr,
        "cmaes_params":   np.zeros(N_MOTORS + N_GAINS, dtype=np.float32),
        "morph_features": feats,
    }


def perturbed_genome(base: np.ndarray, offsets_rad: np.ndarray,
                     mode: str = "az") -> np.ndarray:
    """Apply per-arm angular offsets.

    mode="az":    rotate arm azimuths (motor az follows the arm; thrust stays
                  vertical). Sweep A perturbs one arm at a time in the XY plane.
    mode="pitch": tilt arm elevation up/down; motor_pitch stays 0 so thrust
                  remains vertical (matches the codebase convention in
                  hex_sampler where motor_pitch is locked at 0). This changes
                  the propeller's XYZ location and therefore its moment arm
                  for roll/pitch control.
    """
    g = base.copy()
    if mode == "az":
        g[:, 1] = (g[:, 1] + offsets_rad) % (2 * math.pi)
        g[:, 3] = g[:, 1]
    elif mode == "pitch":
        g[:, 2] = np.clip(g[:, 2] + offsets_rad, -math.pi / 2 + 0.05,
                          math.pi / 2 - 0.05)
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'az' or 'pitch'")
    return g


def min_azimuth_gap_deg(morph: dict) -> float:
    locs = np.array([p["loc"] for p in morph["propellers"]], dtype=np.float32)
    az = np.sort(np.arctan2(locs[:, 1], locs[:, 0]) % (2 * math.pi))
    gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
    return float(np.degrees(gaps.min()))


def max_pitch_deg(morph: dict) -> float:
    """Largest per-arm tilt magnitude (deg) from horizontal — for pitch mode."""
    locs = np.array([p["loc"] for p in morph["propellers"]], dtype=np.float32)
    horiz = np.linalg.norm(locs[:, :2], axis=1)
    pitches = np.degrees(np.arctan2(locs[:, 2], np.maximum(horiz, 1e-6)))
    return float(np.abs(pitches).max())


# ---------------------------------------------------------------- envs

def make_env(morph: dict, num_envs: int, seed: int,
             frozen_features: np.ndarray | None,
             device: str = "cpu") -> ResidualDroneEnv:
    m = dict(morph)
    if frozen_features is not None:
        m["morph_features"] = frozen_features
    # alpha=1.0 with all-zero cmaes_params == pure RL: the prior's effort is
    # identically zero (zero trims, zero gains), so action = effort_to_action(residual).
    return ResidualDroneEnv(
        m, task="hover", alpha=1.0,
        num_envs=num_envs, max_steps=EPISODE_STEPS + 50,
        device=device, seed=seed,
    )


# ---------------------------------------------------------------- train

class HoverRewardShapingWrapper:
    """VecEnvWrapper adding a position-quadratic centering penalty to hover.

    The base env's telescoping distance reward has near-zero gradient at the
    target, letting the policy drift ~0.5 m without correction. Adding
    `-shape_coef * (||xy - xy_target||^2 + (z - z_target)^2)` per step gives
    a strong, dense signal that pulls the policy back to the target and
    turns the return surface into a proper bowl.
    """
    def __init__(self, env, shape_coef: float = 0.10):
        self._env = env
        self.shape_coef = float(shape_coef)
        self._target_ned = np.asarray(env.HOVER_TARGET_NED, dtype=np.float32)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        return self._env.reset()

    def step_async(self, actions):
        self._env.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self._env.step_wait()
        pos = self._env.world_states[:, 0:3].cpu().numpy()
        err = pos - self._target_ned
        # scale so a 1 m drift subtracts 0.1/step ~ +0.1*num_env_steps of loss
        penalty = self.shape_coef * (err ** 2).sum(axis=1)
        return obs, rewards - penalty.astype(np.float32), dones, infos


def train(args, out_dir: Path) -> Path:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecNormalize

    morph = genome_to_morph(canonical_genome())
    print(f"[train] canonical hex: mass={morph['mass']:.3f} kg  "
          f"twr={morph['twr']:.2f}  min_gap={min_azimuth_gap_deg(morph):.1f} deg")

    raw = make_env(morph, num_envs=args.num_envs, seed=0, frozen_features=None,
                   device=args.device)
    if args.shape_reward:
        raw = HoverRewardShapingWrapper(raw, shape_coef=args.shape_coef)
        print(f"[train] reward shaping ENABLED: -{args.shape_coef}*|pos-target|^2")
    env = VecNormalize(raw, norm_obs=False, norm_reward=True,
                       clip_reward=10.0, gamma=0.999)

    ppo_kwargs = dict(
        policy_kwargs=dict(net_arch=[256, 256]),
        n_steps=1024,
        batch_size=args.num_envs * 1024 // 8,
        gamma=0.999,
        device=args.device,
        seed=0,
        verbose=1,
        ent_coef=args.ent_coef,
    )
    if args.clip_vf is not None:
        ppo_kwargs["clip_range_vf"] = args.clip_vf
        print(f"[train] value clipping ENABLED: clip_range_vf={args.clip_vf}")
    if args.ent_coef > 0:
        print(f"[train] ent_coef={args.ent_coef} (default 0.0)")

    model = PPO("MlpPolicy", env, **ppo_kwargs)
    t0 = time.time()
    model.learn(total_timesteps=args.train_steps)
    print(f"[train] done in {time.time() - t0:.0f}s")

    filename = f"hover_policy{args.suffix}.zip"
    policy_path = args.out_dir / filename
    model.save(str(policy_path))
    print(f"[train] policy -> {policy_path}")
    return policy_path


# ---------------------------------------------------------------- eval

def evaluate(model, morph: dict, seed: int, num_envs: int,
             frozen_features: np.ndarray | None,
             device: str = "cpu") -> dict:
    """Deterministic rollouts; returns success/reward/survival statistics."""
    env = make_env(morph, num_envs=num_envs, seed=seed,
                   frozen_features=frozen_features, device=device)
    obs = env.reset()
    alive = np.ones(num_envs, dtype=bool)
    steps_alive = np.full(num_envs, EPISODE_STEPS, dtype=np.int64)
    total_reward = np.zeros(num_envs, dtype=np.float64)

    for t in range(EPISODE_STEPS):
        actions, _ = model.predict(obs, deterministic=True)
        env.step_async(actions)
        obs, r, dones, infos = env.step_wait()
        total_reward[alive] += r[alive]
        crashed = np.asarray(dones, dtype=bool) & alive
        # max_steps > EPISODE_STEPS, so any done here is a crash, not timeout
        steps_alive[crashed] = t + 1
        alive &= ~crashed

    pos = env.world_states[:, 0:3].cpu().numpy()
    target = np.asarray(env.HOVER_TARGET_NED, dtype=np.float64)
    drift = np.linalg.norm(pos - target, axis=1)
    success = alive & (drift < SUCCESS_DRIFT_M)
    return {
        "success_rate": float(success.mean()),
        "survival_rate": float(alive.mean()),
        "mean_reward": float(total_reward.mean()),
        "mean_steps_alive": float(steps_alive.mean()),
        "mean_final_drift": float(drift[alive].mean()) if alive.any() else float("nan"),
    }


def sweep(args, out_dir: Path) -> None:
    from stable_baselines3 import PPO

    # Policy lives at OUT_DIR_DEFAULT (or --out-dir); sweep results are
    # mode-scoped underneath.
    filename = f"hover_policy{args.suffix}.zip"
    for candidate in (args.out_dir / filename,
                      out_dir.parent / filename,
                      out_dir / filename):
        if candidate.exists():
            policy_path = candidate
            break
    else:
        raise FileNotFoundError(f"policy {filename} not found under {args.out_dir}")
    mode_dir = out_dir
    mode_dir.mkdir(parents=True, exist_ok=True)

    model = PPO.load(str(policy_path), device=args.device)
    base = canonical_genome()
    canon = genome_to_morph(base)
    frozen = None if args.update_features else canon["morph_features"]

    stats = evaluate(model, canon, seed=123, num_envs=args.eval_envs,
                     frozen_features=frozen, device=args.device)
    print(f"[gate] mode={args.mode}  canonical morph: "
          f"success={stats['success_rate']:.2f} "
          f"reward={stats['mean_reward']:.1f} drift={stats['mean_final_drift']:.3f}")
    if stats["success_rate"] < 0.8:
        print("[gate] WARNING: specialist success < 80% on its own morph — "
              "break thresholds below are not meaningful. Train longer.")

    arms = (list(range(N_MOTORS)) if args.arms == "all"
            else [int(a) for a in args.arms.split(",")])
    offsets_deg = np.arange(-45.0, 45.0 + 1e-9, 2.5)
    _health = (min_azimuth_gap_deg if args.mode == "az" else max_pitch_deg)
    _health_key = "min_gap_deg" if args.mode == "az" else "max_pitch_deg"

    # ---- Sweep A: single-arm perturbation ----------------------------
    rows_a = []
    for arm in arms:
        for off in offsets_deg:
            vec = np.zeros(N_MOTORS, dtype=np.float32)
            vec[arm] = math.radians(off)
            morph = genome_to_morph(perturbed_genome(base, vec, mode=args.mode))
            s = evaluate(model, morph, seed=123, num_envs=args.eval_envs,
                         frozen_features=frozen, device=args.device)
            rows_a.append({
                "arm": arm, "offset_deg": off,
                _health_key: round(_health(morph), 2), **s,
            })
            print(f"[A] arm={arm} off={off:+6.1f}  succ={s['success_rate']:.2f} "
                  f"rew={s['mean_reward']:7.1f} steps={s['mean_steps_alive']:5.0f}")
    _write_csv(mode_dir / "single_arm.csv", rows_a)

    # ---- Sweep B: all-arm Gaussian noise -----------------------------
    sigmas_deg = np.arange(2.5, 30.0 + 1e-9, 2.5)
    rows_b = []
    for sigma in sigmas_deg:
        for draw in range(args.sigma_draws):
            rng = np.random.RandomState(1000 + draw)
            vec = rng.normal(0.0, math.radians(sigma), size=N_MOTORS).astype(np.float32)
            morph = genome_to_morph(perturbed_genome(base, vec, mode=args.mode))
            s = evaluate(model, morph, seed=123, num_envs=args.eval_envs,
                         frozen_features=frozen, device=args.device)
            rows_b.append({
                "sigma_deg": sigma, "draw": draw,
                _health_key: round(_health(morph), 2), **s,
            })
        sub = [r for r in rows_b if r["sigma_deg"] == sigma]
        mean_succ = np.mean([r["success_rate"] for r in sub])
        print(f"[B] sigma={sigma:5.1f}  mean_succ={mean_succ:.2f}")
    _write_csv(mode_dir / "all_arm.csv", rows_b)

    _plot(rows_a, rows_b, arms, mode_dir, mode=args.mode)
    _print_thresholds(rows_a, rows_b, arms, mode=args.mode)


# ---------------------------------------------------------------- output

def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[out] {path} ({len(rows)} rows)")


def _plot(rows_a, rows_b, arms, out_dir: Path, mode: str = "az") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    axis_label = ("azimuth (deg)" if mode == "az"
                  else "elevation / arm-pitch (deg)")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for arm in arms:
        sub = sorted((r for r in rows_a if r["arm"] == arm),
                     key=lambda r: r["offset_deg"])
        axes[0].plot([r["offset_deg"] for r in sub],
                     [r["success_rate"] for r in sub],
                     marker="o", ms=3, label=f"arm {arm}")
    axes[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[0].set_xlabel(f"single-arm {axis_label} offset")
    axes[0].set_ylabel("success rate")
    axes[0].set_title(f"Sweep A: single-arm ({mode})")
    axes[0].legend(fontsize=8)

    sigmas = sorted({r["sigma_deg"] for r in rows_b})
    mean = [np.mean([r["success_rate"] for r in rows_b if r["sigma_deg"] == s])
            for s in sigmas]
    std = [np.std([r["success_rate"] for r in rows_b if r["sigma_deg"] == s])
           for s in sigmas]
    axes[1].errorbar(sigmas, mean, yerr=std, marker="o", ms=4, capsize=3)
    axes[1].axhline(0.5, color="gray", ls="--", lw=0.8)
    axes[1].set_xlabel(f"all-arm {axis_label} noise sigma")
    axes[1].set_ylabel("success rate")
    axes[1].set_title(f"Sweep B: all-arm ({mode})")

    fig.suptitle(f"PPO hover specialist: break thresholds — mode={mode}")
    fig.tight_layout()
    out = out_dir / "break_thresholds.png"
    fig.savefig(out, dpi=140)
    print(f"[out] {out}")


def _print_thresholds(rows_a, rows_b, arms, mode: str = "az") -> None:
    print(f"\n=== Break thresholds ({mode}, success >= 50%) ===")
    for arm in arms:
        sub = {r["offset_deg"]: r["success_rate"]
               for r in rows_a if r["arm"] == arm}
        ok_pos = [o for o in sorted(sub) if o >= 0 and sub[o] >= 0.5]
        ok_neg = [o for o in sorted(sub) if o <= 0 and sub[o] >= 0.5]
        pos_thr = max(ok_pos) if ok_pos else float("nan")
        neg_thr = min(ok_neg) if ok_neg else float("nan")
        print(f"  arm {arm}: holds within [{neg_thr:+.1f}, {pos_thr:+.1f}] deg")
    sigmas = sorted({r["sigma_deg"] for r in rows_b})
    ok = [s for s in sigmas
          if np.mean([r["success_rate"] for r in rows_b if r["sigma_deg"] == s]) >= 0.5]
    print(f"  all-arm: mean success holds up to sigma = "
          f"{max(ok) if ok else float('nan'):.1f} deg")


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["train", "sweep", "all"])
    p.add_argument("--train-steps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--eval-envs", type=int, default=16)
    p.add_argument("--sigma-draws", type=int, default=8)
    p.add_argument("--arms", type=str, default="0,1",
                   help='comma-separated arm indices or "all"')
    p.add_argument("--update-features", action="store_true",
                   help="let the policy see the perturbed morph features "
                        "(default: frozen at canonical values)")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mode", choices=["az", "pitch"], default="az",
                   help="which arm-angle axis to perturb in the sweep")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--suffix", type=str, default="",
                   help='policy filename suffix, e.g. "_tuned" writes '
                        'hover_policy_tuned.zip; sweeps read the same. '
                        'Empty = baseline hover_policy.zip.')
    p.add_argument("--shape-reward", action="store_true",
                   help="train with a position-quadratic hover penalty")
    p.add_argument("--shape-coef", type=float, default=0.10)
    p.add_argument("--ent-coef", type=float, default=0.0,
                   help="PPO entropy coefficient (default 0.0)")
    p.add_argument("--clip-vf", type=float, default=None,
                   help="If set, enables clip_range_vf (e.g. 0.2)")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    # For non-empty suffix, sweep outputs go under <out-dir>/<suffix_bare>/<mode>/
    root_for_mode = (args.out_dir if not args.suffix
                     else args.out_dir / args.suffix.lstrip("_"))
    mode_dir = root_for_mode / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in ("train", "all"):
        train(args, mode_dir)
    if args.phase in ("sweep", "all"):
        sweep(args, mode_dir)


if __name__ == "__main__":
    main()
