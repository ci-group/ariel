"""Test: CPPN-NEAT activation functions."""

import math

import pytest

from ariel.body_phenotypes.robogen_lite.cppn_neat.activations import (
    ACTIVATION_FUNCTIONS,
    DEFAULT_ACTIVATION,
    gaussian,
    relu,
    sigmoid,
    sin_act,
    tanh,
)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------


def test_sigmoid_zero() -> None:
    """sigmoid(0) == 0.5."""
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_large_positive() -> None:
    """sigmoid(large positive) approaches 1."""
    assert sigmoid(100) == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative() -> None:
    """sigmoid(large negative) approaches 0."""
    assert sigmoid(-100) == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_range() -> None:
    """sigmoid output is always in (0, 1)."""
    for x in [-10, -1, 0, 1, 10]:
        val = sigmoid(x)
        assert 0.0 < val < 1.0


# ---------------------------------------------------------------------------
# tanh
# ---------------------------------------------------------------------------


def test_tanh_zero() -> None:
    """tanh(0) == 0."""
    assert tanh(0) == pytest.approx(0.0)


def test_tanh_positive() -> None:
    """tanh(positive) is positive."""
    assert tanh(1) > 0


def test_tanh_negative() -> None:
    """tanh(negative) is negative."""
    assert tanh(-1) < 0


def test_tanh_range() -> None:
    """tanh output is always in (-1, 1)."""
    for x in [-10, -1, 0, 1, 10]:
        val = tanh(x)
        assert -1.0 < val < 1.0


# ---------------------------------------------------------------------------
# sin_act
# ---------------------------------------------------------------------------


def test_sin_act_zero() -> None:
    """sin(0) == 0."""
    assert sin_act(0) == pytest.approx(0.0)


def test_sin_act_pi_half() -> None:
    """sin(pi/2) == 1."""
    assert sin_act(math.pi / 2) == pytest.approx(1.0)


def test_sin_act_range() -> None:
    """sin output is always in [-1, 1]."""
    for x in [-10, -1, 0, 1, 10]:
        val = sin_act(x)
        assert -1.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------


def test_gaussian_zero() -> None:
    """gaussian(0) == 1 (peak of bell curve)."""
    assert gaussian(0) == pytest.approx(1.0)


def test_gaussian_positive() -> None:
    """gaussian output is always positive."""
    for x in [-5, -1, 0, 1, 5]:
        assert gaussian(x) > 0


def test_gaussian_symmetric() -> None:
    """gaussian is symmetric: gaussian(x) == gaussian(-x)."""
    for x in [1, 2, 3]:
        assert gaussian(x) == pytest.approx(gaussian(-x))


def test_gaussian_decays() -> None:
    """gaussian decays away from zero."""
    assert gaussian(1) < gaussian(0)
    assert gaussian(2) < gaussian(1)


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------


def test_relu_zero() -> None:
    """relu(0) == 0."""
    assert relu(0) == pytest.approx(0.0)


def test_relu_positive() -> None:
    """relu passes positive values unchanged."""
    assert relu(3.5) == pytest.approx(3.5)


def test_relu_negative() -> None:
    """relu clamps negative values to 0."""
    assert relu(-5.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# ACTIVATION_FUNCTIONS registry
# ---------------------------------------------------------------------------


def test_activation_functions_contains_all() -> None:
    """ACTIVATION_FUNCTIONS dict has all five activation names."""
    expected = {"sigmoid", "tanh", "sin", "gaussian", "relu"}
    assert set(ACTIVATION_FUNCTIONS.keys()) == expected


def test_activation_functions_all_callable() -> None:
    """Every entry in ACTIVATION_FUNCTIONS is callable."""
    for fn in ACTIVATION_FUNCTIONS.values():
        assert callable(fn)


def test_activation_functions_produce_floats() -> None:
    """All activation functions return a float for input 0."""
    for fn in ACTIVATION_FUNCTIONS.values():
        result = fn(0)
        assert isinstance(result, float)


def test_default_activation_in_registry() -> None:
    """DEFAULT_ACTIVATION key exists in ACTIVATION_FUNCTIONS."""
    assert DEFAULT_ACTIVATION in ACTIVATION_FUNCTIONS
