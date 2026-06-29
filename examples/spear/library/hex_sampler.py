"""Constrained hexacopter morphology sampler with feasibility filter.

See `IMPLEMENTATION_PLAN.md` §"Hex sampler". Generates valid hexacopter
genomes in the `spherical_angular` encoding (6 active rows × 6 columns)
under physically meaningful constraints, then rejects any morph that
the analytical hover machinery in 35c can't fly.

CLI:
    uv run examples/spear/library/hex_sampler.py --n 100 --seed 42 \\
        --plot __data__/hex_library/v1/coverage.png
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.dynamics_params import derive_reference_params

sys.path.insert(0, str(Path(__file__).parent))
from morphology_features import _compute_u_hover, _compute_twr  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Sampler ranges (locked by IMPLEMENTATION_PLAN.md §Hex sampler)
# ─────────────────────────────────────────────────────────────────────────────

N_MOTORS = 6                # hex
# Arm-length lower floor pushed up from 0.08 so propeller blades don't
# physically overlap. Default rotor_radius in
# spherical_angular_to_blueprint is 0.0635 m; with 60° hex spacing,
# adjacent prop centres are separated by `arm_length`. At arm=0.10 m
# the blades just touch; below that, the morph isn't physical and the
# small moment arms also leave the prior with near-zero pitch/roll
# authority. 0.10 is the user-chosen minimum.
ARM_MAG_RANGE = (0.10, 0.20)        # m  (base radius, before per-motor jitter)
ARM_MAG_JITTER_PROB = 0.5           # probability of per-motor ±jitter
ARM_MAG_JITTER_FRAC = 0.30          # ±30% per-motor multiplier when jittered
MIN_AZIMUTH_GAP_DEG = 30.0          # enforced minimum gap between adjacent arms
TILTED_ARM_PROB = 0.20              # fraction of morphs with non-planar arms
TILTED_ARM_RANGE_DEG = (-10.0, 10.0)
CORE_MASS_RANGE = (0.20, 0.60)      # kg
PROP_SIZE_CHOICES = (4, 5, 6, 7)    # inch
GRAVITY = 9.81

# Feasibility thresholds
MIN_TWR = 1.5
MIN_GAP_DEG_HARD = 15.0             # absolute floor regardless of sampler config

# Stratification: 3 bins per axis × 3 axes = 27 cells. Sample
# uniformly across cells when stratification is enabled.
ARM_MEAN_BINS = [(0.10, 0.13), (0.13, 0.17), (0.17, 0.20)]    # m — match ARM_MAG_RANGE
PROP_SIZE_BINS = [4, 5, 6, 7]   # stratify on every prop choice (no overflow)
ASYM_BINS = [(0.0, 0.05), (0.05, 0.15), (0.15, 1.0)]          # arm-length std/mean


# ─────────────────────────────────────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HexMorph:
    """One sampled hexacopter, with everything downstream consumers need."""
    morph_id: str
    seed: int
    genome: np.ndarray         # (6, 6) spherical_angular rows
    core_mass: float
    prop_size: int
    # Cached on construction so we don't redecode repeatedly
    propellers: list[dict] = field(repr=False)
    mass: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)), repr=False)
    # Diagnostic metadata
    u_hover: float = 0.0
    twr: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_azimuths(rng: np.random.Generator, min_gap_deg: float) -> np.ndarray:
    """Sample 6 azimuths in [0, 2π) with enforced minimum gap.

    Uses rejection on the gap constraint, but biases the proposal toward
    even spacing so rejection is rare. Returns the sorted azimuths.
    """
    min_gap = math.radians(min_gap_deg)
    max_attempts = 200
    for _ in range(max_attempts):
        # Propose by jittering the regular hex pattern.
        base = np.linspace(0, 2 * math.pi, N_MOTORS, endpoint=False)
        # Per-motor angular jitter in (-jitter, +jitter); shrink jitter to
        # leave headroom for the min-gap constraint.
        max_jitter = max(0.0, (2 * math.pi / N_MOTORS - min_gap) / 2.0)
        jitter = rng.uniform(-max_jitter, max_jitter, size=N_MOTORS)
        az = np.sort(base + jitter) % (2 * math.pi)
        gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
        if gaps.min() >= min_gap:
            return az
    raise RuntimeError(
        f"_sample_azimuths: failed to satisfy min_gap={min_gap_deg}° in "
        f"{max_attempts} attempts — sampler parameters are inconsistent",
    )


def _sample_arm_magnitudes(rng: np.random.Generator) -> np.ndarray:
    """Per-motor arm lengths. Some morphs are uniform, some have jitter."""
    base = float(rng.uniform(*ARM_MAG_RANGE))
    mags = np.full(N_MOTORS, base, dtype=np.float32)
    if rng.uniform() < ARM_MAG_JITTER_PROB:
        jitter = rng.uniform(
            1.0 - ARM_MAG_JITTER_FRAC,
            1.0 + ARM_MAG_JITTER_FRAC,
            size=N_MOTORS,
        )
        mags = (base * jitter).astype(np.float32)
        # Clip back into the global range so the dynamics module doesn't choke
        mags = np.clip(mags, ARM_MAG_RANGE[0] * 0.8, ARM_MAG_RANGE[1] * 1.2)
    return mags


def _alternating_spin_pattern(rng: np.random.Generator) -> np.ndarray:
    """3 ccw + 3 cw, shuffled so the cw/ccw pattern around the body varies.
    Returns array of {0.0, 1.0} matching spherical_angular's `direction`
    column (0=CCW, 1=CW)."""
    pattern = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    rng.shuffle(pattern)
    return pattern


def _build_genome(rng: np.random.Generator) -> np.ndarray:
    """One spherical_angular genome row per motor.

    Columns: [magnitude, arm_az, arm_pitch, motor_az, motor_pitch, direction]
    """
    az = _sample_azimuths(rng, MIN_AZIMUTH_GAP_DEG)
    mags = _sample_arm_magnitudes(rng)

    if rng.uniform() < TILTED_ARM_PROB:
        arm_pitch = np.radians(
            rng.uniform(*TILTED_ARM_RANGE_DEG, size=N_MOTORS).astype(np.float32),
        )
    else:
        arm_pitch = np.zeros(N_MOTORS, dtype=np.float32)

    motor_pitch = np.zeros(N_MOTORS, dtype=np.float32)   # locked: vertical thrust
    motor_az = az.copy().astype(np.float32)              # collinear with arm

    directions = _alternating_spin_pattern(rng)

    return np.stack([
        mags,                  # magnitude
        az.astype(np.float32), # arm_az
        arm_pitch,             # arm_pitch
        motor_az,              # motor_az (collinear with arm)
        motor_pitch,           # motor_pitch (always 0)
        directions,            # direction
    ], axis=1)


def _check_min_gap(propellers: list[dict]) -> bool:
    """Recompute gap on the decoded propellers (defends against decoder
    rounding); reject if any adjacent gap < hard floor."""
    locs = np.array([p["loc"] for p in propellers], dtype=np.float32)
    az = np.sort(np.arctan2(locs[:, 1], locs[:, 0]) % (2 * math.pi))
    gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
    return bool(gaps.min() >= math.radians(MIN_GAP_DEG_HARD))


def _build_one_candidate(rng: np.random.Generator, morph_id: str, seed: int) -> HexMorph | None:
    """Generate one candidate morph and run all feasibility checks.

    Returns the HexMorph if feasible, None otherwise. Never raises on
    feasibility-related failures — only on programmer errors.
    """
    genome = _build_genome(rng)
    core_mass = float(rng.uniform(*CORE_MASS_RANGE))
    prop_size = int(rng.choice(PROP_SIZE_CHOICES))

    try:
        bp = spherical_angular_to_blueprint(
            genome, core_mass=core_mass, propsize=prop_size,
        )
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return None

    if len(propellers) != N_MOTORS or not _check_min_gap(propellers):
        return None

    # DroneConfiguration computes the actual integrated mass/inertia from
    # the decoded propeller list. The genome's core_mass only sets the
    # plate; arm mass is derived from arm length in DroneConfiguration.
    try:
        cfg = DroneConfiguration(propellers)
        mass = float(cfg.mass)
        inertia = np.asarray(cfg.inertia_matrix, dtype=np.float64)
        params = derive_reference_params(
            propellers=propellers,
            mass=mass,
            inertia=inertia,
            prop_size=prop_size,
            gravity=GRAVITY,
        )
    except Exception:
        return None

    # Flyability checks
    try:
        u_hover = _compute_u_hover(params, N_MOTORS, GRAVITY)
        twr = _compute_twr(params, N_MOTORS, mass, GRAVITY)
    except Exception:
        return None

    if not (-1.0 < u_hover < 1.0):
        return None
    if twr < MIN_TWR:
        return None
    if not np.isfinite(np.linalg.det(inertia)) or np.linalg.det(inertia) < 1e-9:
        return None
    if abs(genome[:, 5].sum() - 3.0) > 0.5:    # not 3 ccw + 3 cw
        return None

    return HexMorph(
        morph_id=morph_id, seed=seed,
        genome=genome, core_mass=core_mass, prop_size=prop_size,
        propellers=propellers, mass=mass, inertia=inertia,
        u_hover=u_hover, twr=twr,
    )


def _stratify_key(morph: HexMorph) -> tuple[int, int, int]:
    """Return the (arm_bin, prop_bin, asym_bin) cell this morph falls into."""
    locs = np.array([p["loc"] for p in morph.propellers], dtype=np.float32)
    radii = np.linalg.norm(locs[:, :2], axis=1)
    mean_arm = float(radii.mean())
    arm_bin = next(
        (i for i, (lo, hi) in enumerate(ARM_MEAN_BINS) if lo <= mean_arm < hi),
        len(ARM_MEAN_BINS) - 1,
    )
    prop_bin = (PROP_SIZE_BINS.index(morph.prop_size)
                if morph.prop_size in PROP_SIZE_BINS else len(PROP_SIZE_BINS) - 1)
    asym = float(radii.std() / max(mean_arm, 1e-6))
    asym_bin = next(
        (i for i, (lo, hi) in enumerate(ASYM_BINS) if lo <= asym < hi),
        len(ASYM_BINS) - 1,
    )
    return arm_bin, prop_bin, asym_bin


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def iter_candidates(seed: int) -> Iterator[HexMorph]:
    """Infinite stream of feasible candidates, deterministic given seed."""
    rng = np.random.default_rng(seed)
    counter = 0
    while True:
        morph_id = f"hex_{seed:08d}_{counter:06d}"
        m = _build_one_candidate(rng, morph_id, seed + counter)
        counter += 1
        if m is not None:
            yield m


def sample_feasible(
    n: int,
    seed: int = 0,
    *,
    stratify: bool = True,
    max_attempts_per_cell: int = 500,
) -> list[HexMorph]:
    """Return `n` feasible morphs.

    If `stratify=True`, distributes across the 27 stratification cells
    as evenly as possible (favouring underfilled cells once any are
    full); if False, takes the first `n` feasible candidates.

    `max_attempts_per_cell` bounds the rejection-sampling effort if a
    cell is hard to hit — the function returns what it has rather than
    looping forever.
    """
    if not stratify:
        return [m for _, m in zip(range(n), iter_candidates(seed))]

    n_cells = len(ARM_MEAN_BINS) * len(PROP_SIZE_BINS) * len(ASYM_BINS)
    target_per_cell = max(1, int(math.ceil(n / n_cells)))
    cells: dict[tuple[int, int, int], list[HexMorph]] = {}
    attempts = 0
    cap = n_cells * max_attempts_per_cell

    for m in iter_candidates(seed):
        attempts += 1
        if attempts > cap:
            break
        key = _stratify_key(m)
        bucket = cells.setdefault(key, [])
        if len(bucket) < target_per_cell:
            bucket.append(m)
        total = sum(len(b) for b in cells.values())
        if total >= n:
            break

    morphs = [m for bucket in cells.values() for m in bucket][:n]
    if len(morphs) < n:
        # Top up uniformly without stratification — better to return n
        # morphs with worse coverage than fewer with perfect coverage.
        for m in iter_candidates(seed + 100_000):
            morphs.append(m)
            if len(morphs) >= n:
                break
    return morphs


# ─────────────────────────────────────────────────────────────────────────────
# Coverage visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_coverage(morphs: list[HexMorph], out_path: str | Path) -> None:
    """Render a coverage report PNG.

    Three panels:
      (a) stratification heatmap: arm_bin × prop_bin, summed over asym
      (b) scatter of (mean arm length, TWR), colored by prop_size
      (c) histogram of azimuth-gap std (catches over-regular sampling)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_bin_counts = np.zeros((len(ARM_MEAN_BINS), len(PROP_SIZE_BINS)), dtype=int)
    mean_arms, twrs, sizes, gap_stds = [], [], [], []
    for m in morphs:
        ab, pb, _ = _stratify_key(m)
        arm_bin_counts[ab, pb] += 1
        locs = np.array([p["loc"] for p in m.propellers])
        radii = np.linalg.norm(locs[:, :2], axis=1)
        az = np.sort(np.arctan2(locs[:, 1], locs[:, 0]) % (2 * math.pi))
        gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
        mean_arms.append(radii.mean())
        twrs.append(m.twr)
        sizes.append(m.prop_size)
        gap_stds.append(gaps.std())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)

    ax = axes[0]
    im = ax.imshow(arm_bin_counts, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(PROP_SIZE_BINS)))
    ax.set_xticklabels([f"prop {s}\"" for s in PROP_SIZE_BINS])
    ax.set_yticks(range(len(ARM_MEAN_BINS)))
    ax.set_yticklabels([f"{lo:.2f}–{hi:.2f} m" for lo, hi in ARM_MEAN_BINS])
    ax.set_title("stratification: arm-length × prop-size")
    for i in range(arm_bin_counts.shape[0]):
        for j in range(arm_bin_counts.shape[1]):
            ax.text(j, i, str(arm_bin_counts[i, j]),
                    ha="center", va="center", color="white", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1]
    sc = ax.scatter(mean_arms, twrs, c=sizes, cmap="plasma",
                    s=30, alpha=0.85, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("mean arm length (m)")
    ax.set_ylabel("thrust-to-weight ratio")
    ax.set_title("flyability scatter")
    ax.axhline(MIN_TWR, color="red", lw=1, ls="--", label=f"TWR floor {MIN_TWR}")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.colorbar(sc, ax=ax, label="prop size (in)", fraction=0.046)

    ax = axes[2]
    ax.hist(np.degrees(gap_stds), bins=20, color="C2", edgecolor="white")
    ax.set_xlabel("azimuth-gap std (deg)")
    ax.set_ylabel("morphs")
    ax.set_title("angular-spacing diversity")
    ax.grid(alpha=0.3)

    fig.suptitle(f"hex sampler — {len(morphs)} morphs", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    p = argparse.ArgumentParser(description="Sample feasible hex morphologies")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-stratify", action="store_true")
    p.add_argument("--plot", default=None,
                   help="Write coverage PNG (skipped if omitted)")
    p.add_argument("--summary", action="store_true",
                   help="Print per-morph summary line")
    args = p.parse_args()

    import time
    t0 = time.time()
    morphs = sample_feasible(args.n, seed=args.seed, stratify=not args.no_stratify)
    dt = time.time() - t0
    print(f"Sampled {len(morphs)}/{args.n} feasible morphs in {dt:.1f}s "
          f"({args.n / max(dt, 1e-6):.1f} morphs/s)")

    if args.summary:
        for m in morphs[:10]:
            radii = np.linalg.norm([p["loc"] for p in m.propellers], axis=1)
            print(f"  {m.morph_id}: arm_mean={radii.mean():.3f}m  "
                  f"mass={m.mass:.3f}kg  prop={m.prop_size}\"  "
                  f"TWR={m.twr:.2f}  u_hover={m.u_hover:+.3f}")
        if len(morphs) > 10:
            print(f"  ... ({len(morphs) - 10} more)")

    if args.plot:
        plot_coverage(morphs, args.plot)
        print(f"Coverage plot → {args.plot}")


if __name__ == "__main__":
    _cli()
