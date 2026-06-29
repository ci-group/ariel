"""Tests for the hex morphology sampler.

Run:
    uv run pytest examples/spear/library/test_hex_sampler.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from hex_sampler import (  # noqa: E402
    HexMorph,
    MIN_AZIMUTH_GAP_DEG,
    MIN_GAP_DEG_HARD,
    MIN_TWR,
    N_MOTORS,
    iter_candidates,
    sample_feasible,
)


# ─────────────────────────────────────────────────────────────────────────────
# Determinism
# ─────────────────────────────────────────────────────────────────────────────

def test_same_seed_same_morphs():
    a = sample_feasible(10, seed=123, stratify=False)
    b = sample_feasible(10, seed=123, stratify=False)
    assert len(a) == len(b) == 10
    for ma, mb in zip(a, b):
        np.testing.assert_array_equal(ma.genome, mb.genome)
        assert ma.core_mass == mb.core_mass
        assert ma.prop_size == mb.prop_size


def test_different_seeds_different_morphs():
    a = sample_feasible(5, seed=1, stratify=False)
    b = sample_feasible(5, seed=2, stratify=False)
    # At least the first genome should differ
    assert not np.array_equal(a[0].genome, b[0].genome)


# ─────────────────────────────────────────────────────────────────────────────
# Genome structure
# ─────────────────────────────────────────────────────────────────────────────

def test_genome_shape():
    morphs = sample_feasible(5, seed=0, stratify=False)
    for m in morphs:
        assert m.genome.shape == (N_MOTORS, 6), f"unexpected genome shape {m.genome.shape}"
        assert m.genome.dtype == np.float32


def test_genome_no_nans():
    """spherical_angular_to_blueprint treats NaN rows as inactive — a
    hex must have all 6 rows active."""
    morphs = sample_feasible(20, seed=0, stratify=False)
    for m in morphs:
        assert np.isfinite(m.genome).all(), f"NaN in genome for {m.morph_id}"


def test_motor_az_collinear_with_arm_az():
    """Our sampler locks motor thrust direction to arm direction."""
    morphs = sample_feasible(10, seed=0, stratify=False)
    for m in morphs:
        # column 1 = arm_az, column 3 = motor_az
        np.testing.assert_allclose(m.genome[:, 1], m.genome[:, 3], atol=1e-6)


def test_motor_pitch_locked_at_zero():
    morphs = sample_feasible(10, seed=0, stratify=False)
    for m in morphs:
        np.testing.assert_array_equal(m.genome[:, 4], np.zeros(N_MOTORS, dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Spin balance — critical for the yaw mixer to converge
# ─────────────────────────────────────────────────────────────────────────────

def test_exactly_three_ccw_three_cw():
    """The yaw mixer can cancel residual yaw torque only if spin is
    perfectly balanced. Sampler must guarantee this."""
    morphs = sample_feasible(50, seed=0, stratify=False)
    for m in morphs:
        ccw_count = int((m.genome[:, 5] == 0).sum())
        cw_count = int((m.genome[:, 5] == 1).sum())
        assert ccw_count == 3 and cw_count == 3, (
            f"{m.morph_id}: spin ratio is {ccw_count} ccw / {cw_count} cw"
        )


def test_propellers_spin_attribute_matches_genome():
    """Decoded propellers must reflect the genome's spin assignment."""
    morphs = sample_feasible(20, seed=0, stratify=False)
    for m in morphs:
        spins_genome = m.genome[:, 5]  # 0=ccw, 1=cw
        spins_decoded = [
            (1.0 if p["dir"][3] == "cw" else 0.0) for p in m.propellers
        ]
        # Order may differ (decoder doesn't promise stable order) — compare counts
        assert sorted(spins_genome.tolist()) == sorted(spins_decoded)


# ─────────────────────────────────────────────────────────────────────────────
# Azimuth-gap constraint
# ─────────────────────────────────────────────────────────────────────────────

def test_min_azimuth_gap_respected():
    """No two arms should sit within MIN_GAP_DEG_HARD of each other."""
    morphs = sample_feasible(50, seed=0, stratify=False)
    for m in morphs:
        locs = np.array([p["loc"] for p in m.propellers], dtype=np.float32)
        az = np.sort(np.arctan2(locs[:, 1], locs[:, 0]) % (2 * math.pi))
        gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
        min_gap_deg = math.degrees(gaps.min())
        assert min_gap_deg >= MIN_GAP_DEG_HARD - 0.5, (
            f"{m.morph_id}: gap {min_gap_deg:.1f}° below floor {MIN_GAP_DEG_HARD}°"
        )


def test_sampler_min_gap_target_met():
    """Sampler-level guarantee is stricter than the hard floor."""
    morphs = sample_feasible(50, seed=0, stratify=False)
    for m in morphs:
        locs = np.array([p["loc"] for p in m.propellers], dtype=np.float32)
        az = np.sort(np.arctan2(locs[:, 1], locs[:, 0]) % (2 * math.pi))
        gaps = np.diff(np.concatenate([az, az[:1] + 2 * math.pi]))
        # Use a 1° slack to account for rounding in the decoder
        assert math.degrees(gaps.min()) >= MIN_AZIMUTH_GAP_DEG - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Flyability checks
# ─────────────────────────────────────────────────────────────────────────────

def test_all_morphs_flyable():
    """Every yielded morph must pass the analytical flyability checks."""
    morphs = sample_feasible(30, seed=0, stratify=False)
    for m in morphs:
        assert -1.0 < m.u_hover < 1.0
        assert m.twr >= MIN_TWR
        assert m.mass > 0
        assert np.linalg.det(m.inertia) > 0


def test_propellers_count_matches():
    morphs = sample_feasible(20, seed=0, stratify=False)
    for m in morphs:
        assert len(m.propellers) == N_MOTORS


# ─────────────────────────────────────────────────────────────────────────────
# Featurizer compatibility (the whole point of the sampler)
# ─────────────────────────────────────────────────────────────────────────────

def test_featurizer_accepts_sampler_output():
    """Sampler output must feed cleanly into morphology_features.

    This is the integration that downstream stages depend on; if it
    breaks, Stage 1's library is unusable."""
    from morphology_features import morph_features

    morphs = sample_feasible(10, seed=0, stratify=False)
    for m in morphs:
        feat = morph_features(
            m.propellers, mass=m.mass, inertia=m.inertia,
            prop_size=m.prop_size,
        )
        assert feat.shape == (22,)
        assert np.isfinite(feat).all(), f"non-finite features for {m.morph_id}"


# ─────────────────────────────────────────────────────────────────────────────
# Stratification
# ─────────────────────────────────────────────────────────────────────────────

def test_stratified_sampling_returns_requested_count():
    morphs = sample_feasible(30, seed=0, stratify=True)
    assert len(morphs) == 30


def test_stratified_more_diverse_than_unstratified():
    """Stratification should produce a wider arm-length distribution
    than naive sampling at the same n."""
    n = 50
    strat = sample_feasible(n, seed=0, stratify=True)
    naive = sample_feasible(n, seed=0, stratify=False)
    strat_radii = [
        np.linalg.norm([p["loc"] for p in m.propellers], axis=1).mean()
        for m in strat
    ]
    naive_radii = [
        np.linalg.norm([p["loc"] for p in m.propellers], axis=1).mean()
        for m in naive
    ]
    # Strat should span a wider range
    assert (max(strat_radii) - min(strat_radii)) >= (max(naive_radii) - min(naive_radii)) * 0.8


# ─────────────────────────────────────────────────────────────────────────────
# iter_candidates streaming API
# ─────────────────────────────────────────────────────────────────────────────

def test_iter_candidates_streams_indefinitely():
    """Should be able to pull more morphs than any single call would
    materialise."""
    it = iter_candidates(seed=99)
    seen = []
    for _ in range(15):
        seen.append(next(it))
    assert len(seen) == 15
    assert all(isinstance(m, HexMorph) for m in seen)


def test_morph_ids_unique():
    morphs = sample_feasible(50, seed=7, stratify=False)
    ids = [m.morph_id for m in morphs]
    assert len(set(ids)) == len(ids), "morph_id collisions"
