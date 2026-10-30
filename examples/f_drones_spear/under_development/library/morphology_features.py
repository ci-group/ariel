"""Permutation-invariant morphology featurizer for the generalist
hex-drone library.

See `IMPLEMENTATION_PLAN.md` §"Morphology featurizer". Fixed-size 22d
vector describing a hex (or any rotor count) morphology to be consumed
by the residual policy in stage 3 training.

Design constraints:
  * Permutation-invariant in motor order — two physically identical
    drones differing only by propeller-list ordering must produce
    identical features.
  * Normalised to roughly O(1) so the residual MLP doesn't have to
    learn the scale.
  * 22d with a small zero-pad at the end, so we can add features later
    without breaking checkpoint shapes.
"""

from __future__ import annotations

import math

import numpy as np

from ariel.simulation.drone.dynamics_params import derive_reference_params

FEATURE_DIM = 22


def _spin_sign(direction: str) -> float:
    """+1 for ccw, -1 for cw. Matches dynamics_params._spin_sign."""
    if direction == "ccw":
        return 1.0
    if direction == "cw":
        return -1.0
    raise ValueError(f"unknown rotation direction: {direction!r}")


def _azimuth_gap_stats(azimuths: np.ndarray) -> tuple[float, float, float]:
    """Circular gap statistics for motor azimuths.

    Returns (mean_gap, max_gap, std_gap) in radians. Always non-negative;
    sums to 2π exactly. Order-invariant because azimuths are sorted first.
    """
    if azimuths.size == 0:
        return 0.0, 0.0, 0.0
    sorted_az = np.sort(azimuths % (2.0 * math.pi))
    gaps = np.diff(np.concatenate([sorted_az, sorted_az[:1] + 2.0 * math.pi]))
    return float(gaps.mean()), float(gaps.max()), float(gaps.std())


def _compute_u_hover(params: dict, n_motors: int, gravity: float) -> float:
    """Closed-form hover-throttle scalar. Same math as 35c."""
    k_w, k_sq = float(params["k_w"]), float(params["k"])
    w_min, w_max = float(params["w_min"]), float(params["w_max"])
    w_hover = math.sqrt(gravity / (k_w * n_motors))
    z = float(np.clip((w_hover - w_min) / (w_max - w_min), 0.0, 1.0))
    disc = (1.0 - k_sq) ** 2 + 4.0 * k_sq * z * z
    u_hover_raw = (-(1.0 - k_sq) + math.sqrt(max(disc, 0.0))) / (2.0 * k_sq)
    return float(np.clip(2.0 * u_hover_raw - 1.0, -1.0, 1.0))


def _compute_twr(params: dict, n_motors: int, mass: float, gravity: float) -> float:
    """Thrust-to-weight at full throttle.

    Per-motor max thrust = k_f · w_max² = (k_w · mass) · w_max².
    TWR = N · k_w · w_max² / g.
    """
    k_w = float(params["k_w"])
    w_max = float(params["w_max"])
    return float(n_motors * k_w * w_max * w_max / gravity)


def morph_features(
    propellers: list[dict],
    mass: float,
    inertia: np.ndarray,
    prop_size: int = 5,
    gravity: float = 9.81,
) -> np.ndarray:
    """Return a fixed-size 22d morphology descriptor.

    Parameters
    ----------
    propellers : list of dicts with keys "loc" (3-vec, body frame, m),
                 "dir" (sequence whose [3] entry is "ccw" or "cw"),
                 optional "propsize" (int).
    mass       : total drone mass (kg).
    inertia    : 3x3 body-frame inertia matrix (kg·m²).
    prop_size  : nominal prop size used to derive the dynamics
                 parameters when the propeller dicts don't carry one.
    gravity    : m/s². Default 9.81.

    Returns
    -------
    np.ndarray of shape (22,), dtype float32.
    """
    n = len(propellers)
    if n == 0:
        raise ValueError("morph_features: empty propeller list")

    # Per-motor primitives — kept as float32 for shape consistency.
    locs = np.array([p["loc"] for p in propellers], dtype=np.float32)        # (N, 3)
    spins = np.array(
        [_spin_sign(p["dir"][3]) for p in propellers], dtype=np.float32,
    )                                                                         # (N,)
    propsizes = np.array(
        [float(p.get("propsize", prop_size)) for p in propellers],
        dtype=np.float32,
    )                                                                         # (N,)
    radii = np.linalg.norm(locs[:, :2], axis=1)                               # (N,)
    azim = np.arctan2(locs[:, 1], locs[:, 0])                                 # (N,)

    # Dynamics-derived scalars (needs the same machinery as the prior).
    params = derive_reference_params(
        propellers=propellers,
        mass=float(mass),
        inertia=np.asarray(inertia),
        prop_size=int(propsizes.mean().round()),
        gravity=float(gravity),
    )
    u_hover = _compute_u_hover(params, n, float(gravity))
    twr = _compute_twr(params, n, float(mass), float(gravity))

    inertia_diag = np.diag(np.asarray(inertia, dtype=np.float32)) / float(mass)
    gap_mean, gap_max, gap_std = _azimuth_gap_stats(azim)

    out = np.concatenate([
        # 1 — normalised motor count (so 22d featurizer is rotor-count general)
        [n / 8.0],
        # 1 — mass (kg, expected O(0.3))
        [float(mass)],
        # 3 — Ixx/Iyy/Izz per kg, expected O(1e-3 .. 1e-1) — keep raw
        inertia_diag.astype(np.float32),
        # 3 — arm-length stats (m)
        [float(radii.mean()), float(radii.std()), float(radii.max())],
        # 1 — spin balance: 0 for perfectly balanced, ±1 for all-one-direction
        [float(spins.sum() / max(n, 1))],
        # 3 — azimuth-gap stats (rad): describe how evenly spaced motors are
        [gap_mean, gap_max, gap_std],
        # 4 — xy position summaries: catches front-heavy / left-heavy bodies
        [float(locs[:, 0].mean()), float(locs[:, 0].std()),
         float(locs[:, 1].mean()), float(locs[:, 1].std())],
        # 1 — analytical hover throttle
        [u_hover],
        # 1 — thrust-to-weight ratio at full throttle
        [twr],
        # 1 — mean prop size (proxy for absolute thrust capability)
        [float(propsizes.mean())],
        # 3 — zero padding for future features (don't break checkpoint shapes)
        [0.0, 0.0, 0.0],
    ]).astype(np.float32)

    assert out.shape == (FEATURE_DIM,), (
        f"morph_features produced {out.shape}, expected ({FEATURE_DIM},) — "
        f"if you added/removed a feature, update FEATURE_DIM and zero-padding."
    )
    return out


__all__ = ["morph_features", "FEATURE_DIM"]
