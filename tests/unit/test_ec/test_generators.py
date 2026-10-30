"""Test: generators and mutators in the ec module."""

import pytest

from ariel.ec.generators import (
    FloatMutator,
    FloatsGenerator,
    IntegerMutator,
    IntegersGenerator,
)


# ---------------------------------------------------------------------------
# IntegersGenerator
# ---------------------------------------------------------------------------


def test_integers_generator_returns_list() -> None:
    """IntegersGenerator.integers returns a list."""
    result = IntegersGenerator.integers(0, 10, size=5)
    assert isinstance(result, list)
    assert len(result) == 5


def test_integers_generator_in_range() -> None:
    """All generated integers fall within [low, high]."""
    result = IntegersGenerator.integers(0, 10, size=100)
    assert all(0 <= v <= 10 for v in result)


def test_integers_generator_choice_from_set() -> None:
    """choice samples from the supplied set."""
    value_set = [10, 20, 30]
    result = IntegersGenerator.choice(value_set, size=50)
    assert all(v in value_set for v in result)


def test_integers_generator_permutation_length() -> None:
    """permutation(n) returns a list of length n containing 0..n-1."""
    n = 8
    result = IntegersGenerator.permutation(n)
    assert sorted(result) == list(range(n))


# ---------------------------------------------------------------------------
# FloatsGenerator
# ---------------------------------------------------------------------------


def test_floats_generator_uniform_range() -> None:
    """FloatsGenerator.uniform samples from [low, high)."""
    result = FloatsGenerator.uniform(0.0, 1.0, size=100)
    assert isinstance(result, list)
    assert all(0.0 <= v < 1.0 for v in result)


def test_floats_generator_normal_size() -> None:
    """FloatsGenerator.normal returns the requested number of samples."""
    result = FloatsGenerator.normal(mean=0.0, std=1.0, size=20)
    assert len(result) == 20


def test_floats_generator_lognormal_positive() -> None:
    """Log-normal samples are all positive."""
    result = FloatsGenerator.lognormal(mean=0.0, sigma=1.0, size=50)
    assert all(v > 0 for v in result)


def test_floats_generator_linspace() -> None:
    """FloatsGenerator.linspace returns evenly spaced values."""
    result = FloatsGenerator.linspace(0.0, 1.0, 5)
    assert len(result) == 5
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# IntegerMutator
# ---------------------------------------------------------------------------


def test_integer_mutator_random_reset_same_length() -> None:
    """random_reset returns a list of the same length."""
    individual = [1, 2, 3, 4, 5]
    result = IntegerMutator.random_reset(individual, 0, 10, 0.5)
    assert len(result) == len(individual)


def test_integer_mutator_random_reset_prob_zero_unchanged() -> None:
    """random_reset with probability 0 never mutates."""
    individual = [5, 5, 5, 5, 5]
    result = IntegerMutator.random_reset(individual, 0, 10, 0.0)
    assert result == individual


def test_integer_mutator_creep_same_length() -> None:
    """integer_creep returns a list of the same length."""
    individual = [1, 2, 3]
    result = IntegerMutator.integer_creep(individual, span=2, mutation_probability=1.0)
    assert len(result) == len(individual)


def test_integer_mutator_swap_same_elements() -> None:
    """swap preserves the multiset of genes."""
    individual = [1, 2, 3, 4, 5]
    result = IntegerMutator.swap(individual, mutation_probability=1.0)
    assert sorted(result) == sorted(individual)


def test_integer_mutator_inversion_same_elements() -> None:
    """inversion preserves the multiset of genes."""
    individual = [1, 2, 3, 4, 5]
    result = IntegerMutator.inversion(individual, mutation_probability=1.0)
    assert sorted(result) == sorted(individual)


def test_integer_mutator_scramble_same_elements() -> None:
    """scramble preserves the multiset of genes."""
    individual = [10, 20, 30, 40, 50]
    result = IntegerMutator.scramble(individual, mutation_probability=1.0)
    assert sorted(result) == sorted(individual)


# ---------------------------------------------------------------------------
# FloatMutator
# ---------------------------------------------------------------------------


def test_float_mutator_gaussian_same_length() -> None:
    """gaussian mutation returns a list of the same length."""
    individual = [0.1, 0.2, 0.3]
    result = FloatMutator.gaussian(individual, std=0.01, mutation_probability=0.5)
    assert len(result) == len(individual)


def test_float_mutator_gaussian_respects_bounds() -> None:
    """gaussian mutation with bounds clips results."""
    individual = [0.5] * 20
    result = FloatMutator.gaussian(
        individual, std=100.0, mutation_probability=1.0,
        lower_bound=0.0, upper_bound=1.0,
    )
    assert all(0.0 <= v <= 1.0 for v in result)


def test_float_mutator_uniform_reset_same_length() -> None:
    """uniform_reset returns a list of the same length."""
    individual = [1.0, 2.0, 3.0]
    result = FloatMutator.uniform_reset(individual, 0.0, 5.0, 0.5)
    assert len(result) == len(individual)


def test_float_mutator_uniform_reset_prob_zero_unchanged() -> None:
    """uniform_reset with probability 0 never changes any gene."""
    individual = [0.5, 0.5, 0.5]
    result = FloatMutator.uniform_reset(individual, 0.0, 1.0, 0.0)
    assert result == pytest.approx(individual)


def test_float_mutator_boundary_only_extremes() -> None:
    """boundary mutation replaces genes with either low or high."""
    individual = [0.5] * 20
    result = FloatMutator.boundary(individual, low=0.0, high=1.0, mutation_probability=1.0)
    assert all(v == pytest.approx(0.0) or v == pytest.approx(1.0) for v in result)


def test_float_mutator_polynomial_within_bounds() -> None:
    """polynomial mutation keeps all genes within [low, high]."""
    individual = [0.5] * 10
    result = FloatMutator.polynomial(individual, low=0.0, high=1.0, mutation_probability=1.0)
    assert all(0.0 <= v <= 1.0 for v in result)


def test_float_mutator_swap_same_elements() -> None:
    """float swap preserves the multiset of genes."""
    individual = [1.0, 2.0, 3.0, 4.0]
    result = FloatMutator.swap(individual, mutation_probability=1.0)
    assert sorted(result) == pytest.approx(sorted(individual))


def test_float_mutator_inversion_same_elements() -> None:
    """float inversion preserves the multiset of genes."""
    individual = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = FloatMutator.inversion(individual, mutation_probability=1.0)
    assert sorted(result) == pytest.approx(sorted(individual))
