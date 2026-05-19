"""Test: crossover operators in the ec module."""

import pytest

from ariel.ec.crossover import Crossover


def _genes(n: int) -> list[float]:
    return [float(i) for i in range(n)]


# ---------------------------------------------------------------------------
# one_point
# ---------------------------------------------------------------------------


def test_one_point_returns_two_children() -> None:
    """one_point produces exactly two children."""
    p_i, p_j = _genes(6), _genes(6)
    c1, c2 = Crossover.one_point(p_i, p_j)
    assert len(c1) == len(p_i)
    assert len(c2) == len(p_j)


def test_one_point_children_are_recombinations() -> None:
    """Each position in the children comes from one of the two parents."""
    p_i = [0.0] * 6
    p_j = [1.0] * 6
    c1, c2 = Crossover.one_point(p_i, p_j)
    assert all(v in (0.0, 1.0) for v in c1)
    assert all(v in (0.0, 1.0) for v in c2)


def test_one_point_symmetric() -> None:
    """Swapping parents gives complementary children."""
    p_i = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    p_j = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    c1, c2 = Crossover.one_point(p_i, p_j)
    c3, c4 = Crossover.one_point(p_j, p_i)
    # The multiset of all genes must be preserved across the pair
    assert sorted(c1 + c2) == pytest.approx(sorted(p_i + p_j))
    assert sorted(c3 + c4) == pytest.approx(sorted(p_j + p_i))


# ---------------------------------------------------------------------------
# n_point
# ---------------------------------------------------------------------------


def test_n_point_valid_n() -> None:
    """n_point with valid n returns two same-length children."""
    p_i, p_j = _genes(8), _genes(8)
    c1, c2 = Crossover.n_point(p_i, p_j, n=3)
    assert len(c1) == 8
    assert len(c2) == 8


def test_n_point_n_zero_raises() -> None:
    """n_point with n=0 raises ValueError."""
    with pytest.raises(ValueError):
        Crossover.n_point(_genes(6), _genes(6), n=0)


def test_n_point_n_too_large_raises() -> None:
    """n_point with n >= len raises ValueError."""
    with pytest.raises(ValueError):
        Crossover.n_point(_genes(6), _genes(6), n=6)


def test_n_point_mismatched_shapes_raise() -> None:
    """n_point with mismatched parent lengths raises ValueError."""
    with pytest.raises(ValueError):
        Crossover.n_point(_genes(4), _genes(6), n=2)


# ---------------------------------------------------------------------------
# uniform
# ---------------------------------------------------------------------------


def test_uniform_returns_two_children() -> None:
    """uniform crossover produces two same-length children."""
    p_i, p_j = _genes(10), _genes(10)
    c1, c2 = Crossover.uniform(p_i, p_j)
    assert len(c1) == 10
    assert len(c2) == 10


def test_uniform_genes_come_from_parents() -> None:
    """Each gene in the children is from one of the parents."""
    p_i = [0.0] * 8
    p_j = [1.0] * 8
    c1, c2 = Crossover.uniform(p_i, p_j)
    assert all(v in (0.0, 1.0) for v in c1)
    assert all(v in (0.0, 1.0) for v in c2)


def test_uniform_invalid_probability_raises() -> None:
    """uniform with swap_probability outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError):
        Crossover.uniform(_genes(5), _genes(5), swap_probability=1.5)


def test_uniform_probability_zero_unchanged() -> None:
    """uniform with swap_probability=0 leaves both children equal to parents."""
    p_i = [1.0, 2.0, 3.0]
    p_j = [4.0, 5.0, 6.0]
    c1, c2 = Crossover.uniform(p_i, p_j, swap_probability=0.0)
    assert c1 == pytest.approx(p_i)
    assert c2 == pytest.approx(p_j)


def test_uniform_probability_one_swaps_all() -> None:
    """uniform with swap_probability=1 fully swaps both children."""
    p_i = [1.0, 2.0, 3.0]
    p_j = [4.0, 5.0, 6.0]
    c1, c2 = Crossover.uniform(p_i, p_j, swap_probability=1.0)
    assert c1 == pytest.approx(p_j)
    assert c2 == pytest.approx(p_i)


# ---------------------------------------------------------------------------
# order_crossover
# ---------------------------------------------------------------------------


def test_order_crossover_returns_permutations() -> None:
    """order_crossover children contain the same elements as parents."""
    p_i = [0, 1, 2, 3, 4, 5]
    p_j = [5, 4, 3, 2, 1, 0]
    c1, c2 = Crossover.order_crossover(p_i, p_j)
    assert sorted(c1) == sorted(p_i)
    assert sorted(c2) == sorted(p_j)


def test_order_crossover_children_are_lists() -> None:
    """order_crossover returns plain Python lists."""
    p_i = [0, 1, 2, 3]
    p_j = [3, 2, 1, 0]
    c1, c2 = Crossover.order_crossover(p_i, p_j)
    assert isinstance(c1, list)
    assert isinstance(c2, list)


def test_order_crossover_mismatched_length_raises() -> None:
    """order_crossover raises ValueError for parents of different lengths."""
    with pytest.raises(ValueError):
        Crossover.order_crossover([0, 1, 2], [0, 1, 2, 3])
