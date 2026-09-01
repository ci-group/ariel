"""Test: isolated fitness evaluation."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.evaluation import (
    directional_fitness,
    evaluate_individual,
    evaluate_population,
)
from ariel.spatial_ea.genetics import create_initial_hyperneat_genome
from ariel.spatial_ea.individual import SpatialIndividual


def _config(**overrides: object) -> SpatialEAConfig:
    """Build a fast evaluation config."""
    defaults = {
        "simulation_time": 0.2,
        "world_size": (10.0, 10.0),
        "save_results": False,
    }
    return SpatialEAConfig(**{**defaults, **overrides})


def _individual() -> SpatialIndividual:
    """Build an unevaluated individual with a random genome."""
    return SpatialIndividual(
        unique_id=0,
        genotype=create_initial_hyperneat_genome(),
    )


def test_directional_fitness_rewards_aim() -> None:
    """Equal distance should score higher when aimed at the target."""
    aimed = directional_fitness(
        total_distance=2.0,
        progress_toward_target=2.0,
        progress_weight=0.5,
    )
    sideways = directional_fitness(
        total_distance=2.0,
        progress_toward_target=0.0,
        progress_weight=0.5,
    )
    away = directional_fitness(
        total_distance=2.0,
        progress_toward_target=-2.0,
        progress_weight=0.5,
    )

    assert aimed == pytest.approx(3.0)
    assert sideways == pytest.approx(2.0)
    assert away == pytest.approx(1.0)
    assert aimed > sideways > away


def test_directional_fitness_of_a_stationary_robot_is_zero() -> None:
    """No movement means no score, and no division by zero."""
    assert directional_fitness(0.0, 0.0, 0.5) == 0.0


def test_directional_fitness_never_goes_negative() -> None:
    """A large backwards bonus must not push the score below zero."""
    assert directional_fitness(1.0, -1.0, 5.0) == 0.0


def test_evaluate_population_records_state() -> None:
    """Evaluation should fill in fitness, positions and the evaluated flag."""
    population = [_individual() for _ in range(3)]
    fitness = evaluate_population(population, _config())

    assert len(fitness) == 3
    for individual, score in zip(population, fitness, strict=True):
        assert individual.evaluated is True
        assert individual.fitness == score
        assert np.isfinite(individual.fitness)
        assert individual.fitness >= 0.0
        assert individual.start_position is not None
        assert individual.end_position is not None
        assert individual.total_distance >= 0.0


def test_evaluate_population_of_nobody() -> None:
    """An empty population should not try to compile a world."""
    assert evaluate_population([], _config()) == []


def test_directional_mode_places_a_target() -> None:
    """Directional fitness should record the target it aimed at."""
    individual = _individual()
    config = _config(
        use_directional_fitness=True,
        target_distance_min=5.0,
        target_distance_max=6.0,
    )

    evaluate_population([individual], config)

    assert individual.target_position is not None
    assert individual.start_position is not None
    gap = float(
        np.linalg.norm(
            individual.target_position[:2] - individual.start_position[:2],
        ),
    )
    assert 5.0 <= gap <= 6.0


def test_distance_mode_places_no_target() -> None:
    """Plain distance fitness should leave the target unset."""
    individual = _individual()
    evaluate_population([individual], _config(use_directional_fitness=False))

    assert individual.target_position is None
    assert individual.fitness == pytest.approx(individual.total_distance)


def test_evaluate_individual_wrapper() -> None:
    """The single-individual helper should agree with the state it wrote."""
    individual = _individual()
    result = evaluate_individual(individual, _config())

    assert result.fitness == individual.fitness
    assert result.total_distance == individual.total_distance
    assert np.array_equal(result.end_position, individual.end_position)
