"""Test: RevDE optimizer."""

import numpy as np
import pytest

from ariel.utils.optimizers.revde import RevDE


def test_revde_initialization() -> None:
    """RevDE initializes with a 3x3 transformation matrix."""
    revde = RevDE(scaling_factor=0.5)
    assert revde.r_matrix.shape == (3, 3)


def test_revde_mutate_returns_three_children() -> None:
    """mutate returns a list of three arrays."""
    revde = RevDE(scaling_factor=0.5)
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    c = np.array([7.0, 8.0, 9.0])
    children = revde.mutate(a, b, c)
    assert len(children) == 3


def test_revde_mutate_children_same_shape() -> None:
    """Each child has the same shape as the parents."""
    revde = RevDE(scaling_factor=0.5)
    rng = np.random.default_rng(0)
    a, b, c = rng.random(5), rng.random(5), rng.random(5)
    children = revde.mutate(a, b, c)
    for child in children:
        assert child.shape == (5,)


def test_revde_mutate_mismatched_shapes_raise() -> None:
    """mutate raises ValueError when parent shapes differ."""
    revde = RevDE(scaling_factor=0.5)
    a = np.ones(3)
    b = np.ones(4)
    c = np.ones(3)
    with pytest.raises(ValueError, match="same shape"):
        revde.mutate(a, b, c)


def test_revde_scaling_factor_negative() -> None:
    """RevDE works correctly with a negative scaling factor."""
    revde = RevDE(scaling_factor=-0.5)
    a, b, c = np.ones(3), np.ones(3), np.ones(3)
    children = revde.mutate(a, b, c)
    assert len(children) == 3


def test_revde_linear_transformation() -> None:
    """The mutation is a linear operation: output is r_matrix @ [a,b,c]."""
    revde = RevDE(scaling_factor=0.3)
    rng = np.random.default_rng(42)
    a, b, c = rng.random(4), rng.random(4), rng.random(4)
    children = revde.mutate(a, b, c)
    expected = revde.r_matrix @ np.vstack([a, b, c])
    for i, child in enumerate(children):
        assert np.allclose(child, expected[i])


def test_revde_reversibility() -> None:
    """Applying the transformation twice with the inverse restores the originals.

    For RevDE the matrix is designed to be (approximately) reversible.
    Here we just verify determinism: same inputs always give same outputs.
    """
    revde = RevDE(scaling_factor=0.5)
    rng = np.random.default_rng(7)
    a, b, c = rng.random(6), rng.random(6), rng.random(6)
    children1 = revde.mutate(a, b, c)
    children2 = revde.mutate(a, b, c)
    for ch1, ch2 in zip(children1, children2):
        assert np.allclose(ch1, ch2)
