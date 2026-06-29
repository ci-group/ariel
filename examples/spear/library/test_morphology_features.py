"""Tests for the morphology featurizer.

Run:
    uv run pytest examples/spear/library/test_morphology_features.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# examples/ isn't a Python package; add the sibling source file directly.
sys.path.insert(0, str(Path(__file__).parent))
from morphology_features import (  # noqa: E402
    FEATURE_DIM,
    morph_features,
)

# Feature layout — keep in sync with morph_features() in
# morphology_features.py. Tests use these named indices so the assert
# messages stay readable if features get reordered.
IDX_COUNT       = 0
IDX_MASS        = 1
IDX_INERTIA     = slice(2, 5)
IDX_RADII       = slice(5, 8)        # mean, std, max
IDX_RADIUS_MEAN = 5
IDX_RADIUS_STD  = 6
IDX_RADIUS_MAX  = 7
IDX_SPIN_BAL    = 8
IDX_GAPS        = slice(9, 12)       # mean, max, std
IDX_GAP_MEAN    = 9
IDX_GAP_MAX     = 10
IDX_GAP_STD     = 11
IDX_XY_STATS    = slice(12, 16)
IDX_U_HOVER     = 16
IDX_TWR         = 17
IDX_PROPSIZE    = 18
IDX_PADDING     = slice(19, 22)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_hex(radius: float = 0.12, prop_size: int = 5) -> tuple[list[dict], float, np.ndarray]:
    """Build a regular hexacopter morphology: 6 motors evenly spaced
    at the given arm radius, alternating ccw/cw, flat in XY plane.
    Returns (propellers, mass, inertia).
    """
    propellers: list[dict] = []
    for i in range(6):
        phi = i * (math.pi / 3.0)            # 0, 60, 120, ..., 300°
        loc = (radius * math.cos(phi), radius * math.sin(phi), 0.0)
        direction = ("nan", "nan", "nan", "ccw" if i % 2 == 0 else "cw")
        propellers.append({"loc": loc, "dir": direction, "propsize": prop_size})

    mass = 0.4   # kg, plausible
    # Diagonal inertia tensor — sphere-ish core + thin arms approximation
    Ixx = Iyy = 0.5 * mass * radius * radius
    Izz = 1.0 * mass * radius * radius
    inertia = np.diag([Ixx, Iyy, Izz]).astype(np.float32)
    return propellers, mass, inertia


# ─────────────────────────────────────────────────────────────────────────────
# Shape and dtype
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape_and_dtype():
    props, mass, inertia = _make_hex()
    feat = morph_features(props, mass, inertia)
    assert feat.shape == (FEATURE_DIM,)
    assert feat.dtype == np.float32


def test_feature_dim_constant():
    """If FEATURE_DIM changes, callers and checkpoint shapes must follow."""
    assert FEATURE_DIM == 22


# ─────────────────────────────────────────────────────────────────────────────
# Permutation invariance — the load-bearing property
# ─────────────────────────────────────────────────────────────────────────────

def test_permutation_invariance_full_shuffle():
    """Same physical drone, propellers listed in a different order →
    identical features. This is the property the residual policy
    silently relies on."""
    props, mass, inertia = _make_hex()
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(props))
    shuffled = [props[i] for i in perm]

    f1 = morph_features(props, mass, inertia)
    f2 = morph_features(shuffled, mass, inertia)
    np.testing.assert_allclose(f1, f2, atol=1e-6)


def test_permutation_invariance_many_random_shuffles():
    """Stress: every random shuffle produces the same vector."""
    props, mass, inertia = _make_hex()
    f_ref = morph_features(props, mass, inertia)
    rng = np.random.default_rng(0)
    for _ in range(20):
        perm = rng.permutation(len(props))
        shuffled = [props[i] for i in perm]
        f = morph_features(shuffled, mass, inertia)
        np.testing.assert_allclose(f, f_ref, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Physical sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_balanced_hex_has_zero_spin_balance():
    """3 ccw + 3 cw → mean spin sign = 0."""
    props, mass, inertia = _make_hex()
    f = morph_features(props, mass, inertia)
    spin_balance = f[IDX_SPIN_BAL]
    assert abs(spin_balance) < 1e-6, f"balanced hex has spin balance {spin_balance}"


def test_all_ccw_has_positive_spin_balance():
    props, mass, inertia = _make_hex()
    for p in props:
        p["dir"] = (*p["dir"][:3], "ccw")
    f = morph_features(props, mass, inertia)
    assert f[IDX_SPIN_BAL] == pytest.approx(1.0, abs=1e-6)


def test_regular_hex_radii_stats():
    """All arms equal → mean = radius, std ≈ 0, max = radius."""
    radius = 0.12
    props, mass, inertia = _make_hex(radius=radius)
    f = morph_features(props, mass, inertia)
    assert f[IDX_RADIUS_MEAN] == pytest.approx(radius, rel=1e-5)
    assert f[IDX_RADIUS_STD] == pytest.approx(0.0, abs=1e-6)
    assert f[IDX_RADIUS_MAX] == pytest.approx(radius, rel=1e-5)


def test_regular_hex_evenly_spaced():
    """Regular hex → all azimuth gaps = 60° = π/3."""
    props, mass, inertia = _make_hex()
    f = morph_features(props, mass, inertia)
    expected_gap = math.pi / 3.0
    assert f[IDX_GAP_MEAN] == pytest.approx(expected_gap, rel=1e-5)
    assert f[IDX_GAP_MAX] == pytest.approx(expected_gap, rel=1e-5)
    assert f[IDX_GAP_STD] == pytest.approx(0.0, abs=1e-5)


def test_u_hover_in_valid_range():
    """Hover throttle must lie in (-1, 1) for any flyable morph."""
    props, mass, inertia = _make_hex()
    f = morph_features(props, mass, inertia)
    u_hover = f[IDX_U_HOVER]
    assert -1.0 < u_hover < 1.0, f"u_hover {u_hover} outside (-1, 1)"


def test_twr_above_one_for_flyable_hex():
    """A drone that cannot lift itself can't hover."""
    props, mass, inertia = _make_hex()
    f = morph_features(props, mass, inertia)
    twr = f[IDX_TWR]
    assert twr > 1.0, f"TWR {twr} ≤ 1 — drone cannot hover"


def test_padding_zeros():
    """Trailing 3 entries are reserved for future features and must stay zero."""
    props, mass, inertia = _make_hex()
    f = morph_features(props, mass, inertia)
    np.testing.assert_array_equal(f[-3:], np.zeros(3, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Empty / invalid inputs
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_propellers_raises():
    with pytest.raises(ValueError, match="empty propeller list"):
        morph_features([], mass=0.4, inertia=np.eye(3))


def test_unknown_direction_raises():
    props, mass, inertia = _make_hex()
    props[0]["dir"] = ("nan", "nan", "nan", "diagonal")
    with pytest.raises(ValueError, match="unknown rotation direction"):
        morph_features(props, mass, inertia)


# ─────────────────────────────────────────────────────────────────────────────
# Sensitivity — features should respond to physically meaningful changes
# ─────────────────────────────────────────────────────────────────────────────

def test_features_differ_when_radius_differs():
    """Two different arm lengths must produce different features
    (otherwise the descriptor can't possibly let the policy distinguish
    morphologies)."""
    a_props, a_mass, a_inertia = _make_hex(radius=0.10)
    b_props, b_mass, b_inertia = _make_hex(radius=0.18)
    fa = morph_features(a_props, a_mass, a_inertia)
    fb = morph_features(b_props, b_mass, b_inertia)
    assert not np.allclose(fa, fb), "different arm lengths → identical features (bug)"


def test_asymmetric_arms_have_nonzero_radius_std():
    """One leg longer than the others → radius std > 0."""
    props, mass, inertia = _make_hex(radius=0.12)
    # Lengthen one arm
    px, py, pz = props[0]["loc"]
    scale = 1.5
    props[0]["loc"] = (px * scale, py * scale, pz)
    f = morph_features(props, mass, inertia)
    radius_std = f[IDX_RADIUS_STD]
    assert radius_std > 0.01, f"asymmetric arms gave std={radius_std}"
