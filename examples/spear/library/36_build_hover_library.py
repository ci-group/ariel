"""Stage 1 — build the hover library.

For each of N sampled hexacopter morphologies:
  1. run a 35c-style CMA-ES hover optimisation
  2. record the resulting 11d controller parameters + hover score
  3. persist morph_features + cmaes_params + metadata to a library file

The resulting library is the input for Stage 2/3 (residual env + PPO MTRL).

Run:
    uv run examples/spear/library/36_build_hover_library.py \\
        --n 100 --seed 42 --out __data__/hex_library/v1/

Pass `--budget 100 --population 32` for a fast smoke test before the full run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import nevergrad as ng
import numpy as np
import torch
from rich.console import Console

from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics

sys.path.insert(0, str(Path(__file__).parent))
from hex_sampler import HexMorph, sample_feasible           # noqa: E402
from morphology_features import FEATURE_DIM, morph_features  # noqa: E402
from prior_controller import HoverPrior, N_GAINS             # noqa: E402

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Per-morph training (the 35c-equivalent inner loop, slimmed down)
# ─────────────────────────────────────────────────────────────────────────────

HOVER_TARGET_NED = (0.0, 0.0, -1.5)
GRAVITY = 9.81
DT = 0.01


@dataclass
class TrainResult:
    cmaes_params: np.ndarray   # (N+5,)
    score: float               # cumulative dense reward (max single-env)
    median_score: float        # median across final-gen population
    generations_run: int
    train_time_s: float


def _make_env(propellers, params_dict, num_envs, max_steps, device, action_scale=0.4, twr=None):
    """Build the batched hover env + prior. Mirrors BatchedHover in 35c."""
    dev = torch.device(device)
    dtype = torch.float32
    dyn = _build_torch_dynamics(params_dict, len(propellers), GRAVITY, dev, dtype)
    prior = HoverPrior(
        propellers=propellers, params=params_dict,
        target_ned=HOVER_TARGET_NED,
        gravity=GRAVITY, action_scale=action_scale, twr=twr,
        device=dev, dtype=dtype,
    )
    return dev, dtype, dyn, prior, num_envs, max_steps


@torch.no_grad()
def _rollout(params_batch, dev, dtype, dyn, prior, num_envs, max_steps):
    """One CMA-ES generation: roll out `num_envs` candidates for `max_steps`
    and return per-env cumulative reward. Same controller + reward as 35c."""
    # Reset with the same perturbations as 35c.
    n_mot = prior.n_motors
    state_dim = 12 + n_mot
    s = torch.zeros((num_envs, state_dim), device=dev, dtype=dtype)
    s[:, 0:3] = prior.target.unsqueeze(0)
    s[:, 0:3] += (torch.rand((num_envs, 3), device=dev, dtype=dtype) - 0.5) * 0.2
    s[:, 6:8] = (torch.rand((num_envs, 2), device=dev, dtype=dtype) - 0.5) * 0.1
    # Motor state w ∈ [-1, 1] maps via W = (w+1)·W_R/2 + w_lo to physical
    # rotor speed. The default w=0 places motors at mid-throttle, which
    # for high-TWR morphs is far above W_hover — the drone explodes
    # upward at t=0 before feedback can engage. Initialise motors to
    # the hover-equivalent normalised state instead.
    w_lo, w_hi = float(prior._w_min), float(prior._w_max)
    W_hover = math.sqrt(GRAVITY / (prior._k_w * n_mot))
    w_hover_norm = (2.0 * W_hover - (w_hi + w_lo)) / max(w_hi - w_lo, 1e-6)
    s[:, 12:12 + n_mot] = float(w_hover_norm)

    alive = torch.ones(num_envs, dtype=torch.bool, device=dev)
    cum_reward = torch.zeros(num_envs, dtype=dtype, device=dev)

    for _ in range(max_steps):
        action = prior.prior_action(s, params_batch)
        sd = dyn(s.T, action.T).T
        s = s + DT * sd

        pos = s[:, 0:3]
        tilt = torch.norm(s[:, 6:8], dim=1)
        yaw_rate = s[:, 11:12]
        dist = torch.norm(pos - prior.target.unsqueeze(0), dim=1)
        oob = (pos.abs() > 3.0).any(dim=1)
        flipped = tilt > (math.pi / 3)
        divg = ~torch.isfinite(s).all(dim=1)
        dead = oob | flipped | divg

        r_abs = yaw_rate.abs().squeeze(-1)
        reward = (torch.exp(-(dist / 0.4) ** 2)
                  * torch.exp(-(tilt / 0.25) ** 2)
                  * torch.exp(-(r_abs / 1.0) ** 2))
        cum_reward = torch.where(alive & ~dead, cum_reward + reward, cum_reward)
        alive = alive & ~dead
        if not alive.any():
            break

    return cum_reward.cpu().numpy()


def train_one_morph(
    morph: HexMorph, *,
    budget: int, population: int, max_steps: int,
    seed: int, device: str, sigma0: float = 0.4,
    zero_init: bool = False, action_scale: float = 0.4,
) -> TrainResult:
    """Run a 35c-equivalent CMA-ES on one morph; return best params + score."""
    params_dict = derive_reference_params(
        propellers=morph.propellers, mass=morph.mass,
        inertia=morph.inertia, prop_size=morph.prop_size,
        gravity=GRAVITY,
    )
    dev, dtype, dyn, prior, num_envs, _ = _make_env(
        morph.propellers, params_dict, population, max_steps, device,
        action_scale=action_scale, twr=morph.twr,
    )

    # Warm-start strategy:
    #   * Default: morph-conditional warm-start scaled by mass/inertia.
    #     Tends to converge fast on morphs similar to the reference,
    #     but can trap CMA in a wrong-sign local minimum on outliers.
    #   * --zero-init: start from all zeros + larger sigma. CMA has to
    #     find both signs and magnitudes, but doesn't bias.
    if zero_init:
        init = np.zeros(prior.param_dim, dtype=np.float32)
    else:
        init = prior.default_init_params(mass=morph.mass, inertia=morph.inertia)
    param = ng.p.Array(init=init)
    param.set_mutation(sigma=sigma0)
    optimizer = ng.optimizers.ParametrizedCMA(
        popsize=population, elitist=False, diagonal=False,
        inopts={
            "tolflatfitness": 50,
            "CMA_active": True,
            "CMA_diagonal": 20,
            "bounds": [-2.0, 2.0],
            "seed": int(seed),
            "verbose": -9,
        },
    )(parametrization=param, budget=budget * population, num_workers=population)

    t0 = time.time()
    best_score = 0.0
    best_vec = init.copy()
    last_median = 0.0

    for gen in range(budget):
        cands = [optimizer.ask() for _ in range(population)]
        P = np.stack([c.value for c in cands], axis=0).astype(np.float32)
        rewards = _rollout(
            torch.as_tensor(P, device=dev, dtype=dtype),
            dev, dtype, dyn, prior, num_envs, max_steps,
        )
        for c, r in zip(cands, rewards):
            optimizer.tell(c, float(-r))   # CMA minimises

        idx = int(np.argmax(rewards))
        if rewards[idx] > best_score:
            best_score = float(rewards[idx])
            best_vec = P[idx].astype(np.float32)
        last_median = float(np.median(rewards))
        if last_median >= max_steps - 5:
            break

    return TrainResult(
        cmaes_params=best_vec,
        score=best_score,
        median_score=last_median,
        generations_run=gen + 1,
        train_time_s=time.time() - t0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Library persistence
# ─────────────────────────────────────────────────────────────────────────────

def _save_library(out_dir: Path, rows: list[dict], manifest: dict) -> None:
    """Write `library.npz` + `manifest.json`. Atomic via tmp + rename."""
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "morph_seed":        np.asarray([r["morph_seed"] for r in rows], dtype=np.int64),
        "core_mass":         np.asarray([r["core_mass"] for r in rows], dtype=np.float32),
        "prop_size":         np.asarray([r["prop_size"] for r in rows], dtype=np.int32),
        "mass":              np.asarray([r["mass"] for r in rows], dtype=np.float32),
        "u_hover":           np.asarray([r["u_hover"] for r in rows], dtype=np.float32),
        "twr":               np.asarray([r["twr"] for r in rows], dtype=np.float32),
        "score":             np.asarray([r["score"] for r in rows], dtype=np.float32),
        "median_score":      np.asarray([r["median_score"] for r in rows], dtype=np.float32),
        "generations_run":   np.asarray([r["generations_run"] for r in rows], dtype=np.int32),
        "train_time_s":      np.asarray([r["train_time_s"] for r in rows], dtype=np.float32),
        "genome":            np.stack([r["genome"] for r in rows], axis=0).astype(np.float32),
        "morph_features":    np.stack([r["morph_features"] for r in rows], axis=0).astype(np.float32),
        "cmaes_params":      np.stack([r["cmaes_params"] for r in rows], axis=0).astype(np.float32),
    }
    # np.savez auto-appends `.npz`; write to a `.tmp` basename so the
    # resulting `.tmp.npz` is rename-able to the real path.
    npz_path = out_dir / "library.npz"
    tmp_basename = out_dir / "library.tmp"
    np.savez(str(tmp_basename), **arrays)
    (tmp_basename.parent / (tmp_basename.name + ".npz")).rename(npz_path)

    morph_ids = [r["morph_id"] for r in rows]
    (out_dir / "manifest.json").write_text(json.dumps({
        **manifest, "n_morphs": len(rows), "morph_ids": morph_ids,
    }, indent=2))


def _load_partial(out_dir: Path) -> list[dict]:
    """Return previously completed rows (for resume), or [] if none."""
    partial = out_dir / "library_partial.npz"
    manifest = out_dir / "manifest_partial.json"
    if not (partial.exists() and manifest.exists()):
        return []
    d = np.load(partial)
    m = json.loads(manifest.read_text())
    rows = []
    for i, mid in enumerate(m["morph_ids"]):
        rows.append({
            "morph_id":        mid,
            "morph_seed":      int(d["morph_seed"][i]),
            "core_mass":       float(d["core_mass"][i]),
            "prop_size":       int(d["prop_size"][i]),
            "mass":            float(d["mass"][i]),
            "u_hover":         float(d["u_hover"][i]),
            "twr":             float(d["twr"][i]),
            "score":           float(d["score"][i]),
            "median_score":    float(d["median_score"][i]),
            "generations_run": int(d["generations_run"][i]),
            "train_time_s":    float(d["train_time_s"][i]),
            "genome":          d["genome"][i],
            "morph_features":  d["morph_features"][i],
            "cmaes_params":    d["cmaes_params"][i],
        })
    return rows


def _save_partial(out_dir: Path, rows: list[dict], manifest: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_library(out_dir, rows, manifest)
    # Also leave a `_partial` copy so a separate final-only consumer can
    # distinguish in-progress from complete.
    (out_dir / "library_partial.npz").write_bytes(
        (out_dir / "library.npz").read_bytes(),
    )
    (out_dir / "manifest_partial.json").write_text(
        (out_dir / "manifest.json").read_text(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Build the hover library")
    p.add_argument("--n", type=int, default=100, help="Number of morphs (default 100)")
    p.add_argument("--seed", type=int, default=42, help="Sampler seed")
    p.add_argument("--budget", type=int, default=400, help="CMA generations per morph")
    p.add_argument("--population", type=int, default=128, help="CMA λ per morph")
    p.add_argument("--sigma0", type=float, default=0.4,
                   help="CMA initial σ. 0.4 matches the original 35c "
                        "warm-start; bump to 0.6–0.8 if many morphs get stuck "
                        "in a bad local minimum near the warm-start.")
    p.add_argument("--zero-init", action="store_true",
                   help="Use init=zeros instead of the morph-conditional "
                        "warm-start. Slower per-morph convergence, but "
                        "doesn't bias CMA into a wrong-sign local minimum "
                        "when the warm-start mismatches a particular morph.")
    p.add_argument("--action-scale", type=float, default=0.4,
                   help="HoverPrior.action_scale. 0.4 was calibrated for "
                        "small props; large props need a smaller value to "
                        "avoid over-correction. Try 0.15 if pass rate is "
                        "low on prop-6\"/7\" morphs.")
    p.add_argument("--episode-steps", type=int, default=600,
                   help="Sim steps per rollout (6s @ dt=0.01)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="__data__/hex_library/v1",
                   help="Output directory")
    p.add_argument("--no-stratify", action="store_true",
                   help="Skip stratification (faster, less coverage)")
    p.add_argument("--resume", action="store_true",
                   help="Skip morphs already present in <out>/library_partial.npz")
    p.add_argument("--save-every", type=int, default=10,
                   help="Persist progress every N completed morphs")
    args = p.parse_args()

    out_dir = Path(args.out)
    manifest = {
        "n_requested": args.n,
        "seed": args.seed,
        "budget": args.budget,
        "population": args.population,
        "episode_steps": args.episode_steps,
        "stratify": not args.no_stratify,
        "feature_dim": FEATURE_DIM,
        "cmaes_param_dim_offset": N_GAINS,
    }

    # Sample morphs first; doing all sampling up-front lets us know wall
    # time and lets resume work cleanly.
    console.rule(f"[bold blue]Sampling {args.n} feasible hex morphs")
    t_sample = time.time()
    morphs = sample_feasible(args.n, seed=args.seed,
                             stratify=not args.no_stratify)
    console.log(f"Sampled {len(morphs)} morphs in {time.time()-t_sample:.1f}s")

    # Resume support
    rows: list[dict] = []
    skip_ids: set[str] = set()
    if args.resume:
        rows = _load_partial(out_dir)
        skip_ids = {r["morph_id"] for r in rows}
        console.log(f"Resuming: {len(rows)} morphs already done; "
                    f"{args.n - len(rows)} remaining")

    todo = [m for m in morphs if m.morph_id not in skip_ids]
    console.rule(f"[bold blue]Training hover controllers for {len(todo)} morphs")
    t_all = time.time()

    for i, morph in enumerate(todo):
        t0 = time.time()
        # Per-morph seed derived from sampler seed so re-runs are
        # deterministic (and `--resume` matches what was there).
        train_seed = args.seed * 1_000_003 + morph.seed
        result = train_one_morph(
            morph,
            budget=args.budget, population=args.population,
            max_steps=args.episode_steps,
            seed=train_seed, device=args.device,
            sigma0=args.sigma0, zero_init=args.zero_init,
            action_scale=args.action_scale,
        )
        feat = morph_features(
            morph.propellers, mass=morph.mass,
            inertia=morph.inertia, prop_size=morph.prop_size,
        )
        rows.append({
            "morph_id":        morph.morph_id,
            "morph_seed":      morph.seed,
            "core_mass":       morph.core_mass,
            "prop_size":       morph.prop_size,
            "mass":            morph.mass,
            "u_hover":         morph.u_hover,
            "twr":             morph.twr,
            "score":           result.score,
            "median_score":    result.median_score,
            "generations_run": result.generations_run,
            "train_time_s":    result.train_time_s,
            "genome":          morph.genome,
            "morph_features":  feat,
            "cmaes_params":    result.cmaes_params,
        })

        elapsed = time.time() - t0
        total_done = len(rows)
        eta_s = (time.time() - t_all) / max(i + 1, 1) * (len(todo) - i - 1)
        console.log(
            f"[{total_done:>3}/{args.n}]  {morph.morph_id}  "
            f"score={result.score:>6.1f}/{args.episode_steps}  "
            f"gens={result.generations_run:>3}  "
            f"({elapsed:.1f}s, ETA {eta_s/60:.1f}min)"
        )

        if (i + 1) % args.save_every == 0:
            _save_partial(out_dir, rows, manifest)

    # Final write
    _save_library(out_dir, rows, manifest)
    # Clean up partial markers
    for pname in ("library_partial.npz", "manifest_partial.json"):
        f = out_dir / pname
        if f.exists():
            f.unlink()

    # Summary
    scores = np.array([r["score"] for r in rows], dtype=np.float32)
    console.rule("[bold green]Library complete")
    console.log(f"  n         : {len(rows)}")
    console.log(f"  score min : {scores.min():.1f}")
    console.log(f"  score max : {scores.max():.1f}")
    console.log(f"  score mean: {scores.mean():.1f}")
    console.log(f"  score std : {scores.std():.1f}")
    pass_rate = float((scores >= args.episode_steps * 0.66).mean())  # ≥400/600
    console.log(f"  score ≥ {args.episode_steps * 0.66:.0f}: {pass_rate:.1%}  (gate ≥90%)")
    console.log(f"  total wall: {(time.time() - t_all)/60:.1f} min")
    console.log(f"  written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
