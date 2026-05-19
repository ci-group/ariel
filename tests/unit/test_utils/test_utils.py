"""Test: utility classes and functions."""

# Standard library
from pathlib import Path

# Third-party libraries
import networkx as nx
import numpy as np
import pytest

# Local libraries
from ariel.utils.morphological_descriptor import MorphologicalMeasures
from ariel.utils.noise_gen import PerlinNoise


# ---------------------------------------------------------------------------
# PerlinNoise
# ---------------------------------------------------------------------------


def test_perlin_noise_initialization() -> None:
    """Simply instantiate the PerlinNoise class."""
    _ = PerlinNoise()


def test_perlin_noise_seeded_reproducible() -> None:
    """Two PerlinNoise instances with the same seed produce identical grids."""
    n1 = PerlinNoise(seed=42)
    n2 = PerlinNoise(seed=42)
    g1 = n1.as_grid(32, 32, scale=8.0)
    g2 = n2.as_grid(32, 32, scale=8.0)
    assert np.allclose(g1, g2)


def test_perlin_noise_different_seeds_differ() -> None:
    """Two PerlinNoise instances with different seeds produce different grids."""
    n1 = PerlinNoise(seed=1)
    n2 = PerlinNoise(seed=2)
    g1 = n1.as_grid(32, 32, scale=8.0)
    g2 = n2.as_grid(32, 32, scale=8.0)
    assert not np.allclose(g1, g2)


def test_perlin_noise_output_shape() -> None:
    """as_grid returns an array of shape (height, width)."""
    noise = PerlinNoise(seed=0)
    grid = noise.as_grid(width=64, height=32, scale=16.0)
    assert grid.shape == (32, 64)


def test_perlin_noise_normalize_linear_range() -> None:
    """Linear normalization clips output to approximately [0, 1]."""
    noise = PerlinNoise(seed=7)
    grid = noise.as_grid(64, 64, scale=16.0, normalize="linear")
    assert grid.min() >= 0.0 - 1e-6
    assert grid.max() <= 1.0 + 1e-6


def test_perlin_noise_normalize_clip_range() -> None:
    """Clip normalization keeps all values in [0, 1]."""
    noise = PerlinNoise(seed=7)
    grid = noise.as_grid(64, 64, scale=16.0, normalize="clip")
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_perlin_noise_invalid_width_raises() -> None:
    """as_grid raises ValueError for non-positive width."""
    noise = PerlinNoise()
    with pytest.raises(ValueError):
        noise.as_grid(width=0, height=32, scale=8.0)


def test_perlin_noise_invalid_height_raises() -> None:
    """as_grid raises ValueError for non-positive height."""
    noise = PerlinNoise()
    with pytest.raises(ValueError):
        noise.as_grid(width=32, height=0, scale=8.0)


def test_perlin_noise_invalid_scale_raises() -> None:
    """as_grid raises ValueError for non-positive scale."""
    noise = PerlinNoise()
    with pytest.raises(ValueError):
        noise.as_grid(width=32, height=32, scale=0.0)


# ---------------------------------------------------------------------------
# MorphologicalMeasures
# ---------------------------------------------------------------------------


def _make_single_core_graph() -> nx.DiGraph:
    """Minimal graph: only a CORE node."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    return g


def _make_spider_graph() -> nx.DiGraph:
    """Spider-like graph: core + 4 hinge arms."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    for i, face in enumerate(["FRONT", "BACK", "LEFT", "RIGHT"], start=1):
        g.add_node(i, type="HINGE", rotation="DEG_0")
        g.add_edge(0, i, face=face)
    return g


def _make_branching_graph() -> nx.DiGraph:
    """Core with two bricks attached and hinges on the bricks."""
    g = nx.DiGraph()
    g.add_node(0, type="CORE", rotation="DEG_0")
    g.add_node(1, type="BRICK", rotation="DEG_0")
    g.add_node(2, type="BRICK", rotation="DEG_0")
    g.add_node(3, type="HINGE", rotation="DEG_0")
    g.add_edge(0, 1, face="FRONT")
    g.add_edge(0, 2, face="BACK")
    g.add_edge(1, 3, face="FRONT")
    return g


def test_morphological_empty_graph_raises() -> None:
    """MorphologicalMeasures raises ValueError for an empty graph."""
    with pytest.raises(ValueError, match="empty"):
        MorphologicalMeasures(nx.DiGraph())


def test_morphological_single_core_num_modules() -> None:
    """A graph with only a core has num_modules == 1."""
    m = MorphologicalMeasures(_make_single_core_graph())
    assert m.num_modules == 1


def test_morphological_single_core_no_bricks_or_hinges() -> None:
    """A single-core graph has no bricks or active hinges."""
    m = MorphologicalMeasures(_make_single_core_graph())
    assert m.num_bricks == 0
    assert m.num_active_hinges == 0


def test_morphological_spider_graph_module_counts() -> None:
    """Spider graph has 4 hinges and 0 bricks."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert m.num_active_hinges == 4
    assert m.num_bricks == 0
    assert m.num_modules == 5


def test_morphological_branching_graph_brick_count() -> None:
    """Branching graph has 2 bricks and 1 hinge."""
    m = MorphologicalMeasures(_make_branching_graph())
    assert m.num_bricks == 2
    assert m.num_active_hinges == 1


def test_morphological_symmetry_in_range() -> None:
    """All symmetry scores are in [0, 1]."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert 0.0 <= m.xy_symmetry <= 1.0
    assert 0.0 <= m.xz_symmetry <= 1.0
    assert 0.0 <= m.yz_symmetry <= 1.0
    assert 0.0 <= m.symmetry <= 1.0


def test_morphological_bounding_box_positive() -> None:
    """Bounding box dimensions are all positive integers."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert m.bounding_box_depth >= 1
    assert m.bounding_box_width >= 1
    assert m.bounding_box_height >= 1


def test_morphological_coverage_in_range() -> None:
    """Coverage (C) is in (0, 1]."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert 0.0 < m.C <= 1.0


def test_morphological_joints_ratio_in_range() -> None:
    """Joint ratio J is in [0, 1]."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert 0.0 <= m.J <= 1.0


def test_morphological_limbs_in_range() -> None:
    """Limbs ratio L is in [0, 1]."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert 0.0 <= m.L <= 1.0


def test_morphological_proportion_in_range() -> None:
    """Proportion P is in (0, 1]."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert 0.0 < m.P <= 1.0


def test_morphological_is_2d_flag() -> None:
    """A flat robot with only 90-degree rotations is recognised as 2D."""
    m = MorphologicalMeasures(_make_spider_graph())
    assert isinstance(m.is_2d, bool)


def test_morphological_paper_aliases_consistent() -> None:
    """Paper-letter aliases agree with their named property counterparts."""
    m = MorphologicalMeasures(_make_branching_graph())
    assert m.B == pytest.approx(m.branching)
    assert m.L == pytest.approx(m.limbs)
    assert m.E == pytest.approx(m.length_of_limbs)
    assert m.C == pytest.approx(m.coverage)
    assert m.J == pytest.approx(m.joints)
    assert m.S == pytest.approx(m.symmetry)
    assert m.D == pytest.approx(m.module_diversity)
    assert m.P == pytest.approx(m.proportion)
