"""Test: survivor selection policies."""

# Standard library
import random

# Third-party libraries
import numpy as np

# Local libraries
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.selection import (
    select_individuals,
    select_top_individuals,
)


def _population(fitnesses: list[float], generation: int = 0) -> list:
    """Build a population with the given fitness values."""
    return [
        SpatialIndividual(
            unique_id=i,
            generation=generation,
            fitness=fitness,
        )
        for i, fitness in enumerate(fitnesses)
    ]


def test_select_top_individuals_ranks_by_fitness() -> None:
    """Truncation should keep the fittest, best first."""
    population = _population([0.1, 0.9, 0.5, 0.7])
    selected = select_top_individuals(population, 2)

    assert [ind.fitness for ind in selected] == [0.9, 0.7]
    assert select_top_individuals(population, 0) == []


def test_fitness_based_holds_the_target_size() -> None:
    """Fitness selection should truncate to exactly the target."""
    survivors = select_individuals(
        _population([1.0, 2.0, 3.0, 4.0, 5.0]),
        3,
        method="fitness_based",
        current_generation=1,
    )

    assert len(survivors) == 3
    assert {ind.fitness for ind in survivors} == {3.0, 4.0, 5.0}


def test_age_based_keeps_the_youngest() -> None:
    """Age selection should prefer the most recently born."""
    population = [
        SpatialIndividual(unique_id=0, generation=0),
        SpatialIndividual(unique_id=1, generation=5),
        SpatialIndividual(unique_id=2, generation=3),
    ]
    survivors = select_individuals(
        population,
        2,
        method="age_based",
        current_generation=5,
    )

    assert [ind.unique_id for ind in survivors] == [1, 2]


def test_parents_die_retires_the_individuals_that_mated() -> None:
    """Mated parents should give way to their offspring."""
    parents = [
        SpatialIndividual(unique_id=i, generation=3, fitness=10.0 + i)
        for i in range(4)
    ]
    offspring = [
        SpatialIndividual(unique_id=10 + i, generation=4, fitness=0.1)
        for i in range(4)
    ]

    survivors = select_individuals(
        parents + offspring,
        4,
        method="parents_die",
        current_generation=3,
        paired_indices={0, 1},
    )

    ids = {ind.unique_id for ind in survivors}
    assert ids == {2, 3, 10, 11}
    # Despite being the fittest, the mated parents are gone.
    assert 0 not in ids
    assert 1 not in ids


def test_parents_die_keeps_best_parents_when_short() -> None:
    """When offspring are too few, the fittest mated parents are recalled."""
    parents = [
        SpatialIndividual(unique_id=i, generation=3, fitness=float(i))
        for i in range(4)
    ]
    survivors = select_individuals(
        parents,
        2,
        method="parents_die",
        current_generation=3,
        paired_indices={0, 1, 2, 3},
    )

    assert sorted(ind.unique_id for ind in survivors) == [2, 3]


def test_energy_based_kills_only_the_depleted() -> None:
    """Energy selection should ignore fitness and the target size."""
    population = _population([9.0, 9.0, 9.0])
    population[0].energy = -1.0
    population[1].energy = 0.0
    population[2].energy = 5.0

    survivors = select_individuals(
        population,
        1,
        method="energy_based",
        current_generation=1,
    )

    assert [ind.unique_id for ind in survivors] == [2]


def test_probabilistic_age_spares_the_newborn_and_kills_the_old() -> None:
    """Death probability should be zero at birth and certain at max age."""
    newborns = [SpatialIndividual(unique_id=i, generation=5) for i in range(20)]
    assert (
        len(
            select_individuals(
                newborns,
                5,
                method="probabilistic_age",
                current_generation=5,
                max_age=10,
            ),
        )
        == 20
    )

    elders = [SpatialIndividual(unique_id=i, generation=0) for i in range(20)]
    assert (
        select_individuals(
            elders,
            5,
            method="probabilistic_age",
            current_generation=10,
            max_age=10,
        )
        == []
    )


def test_probabilistic_age_does_not_enforce_population_size() -> None:
    """Survivors may outnumber the target size."""
    population = [
        SpatialIndividual(unique_id=i, generation=10) for i in range(30)
    ]
    survivors = select_individuals(
        population,
        2,
        method="probabilistic_age",
        current_generation=10,
        max_age=10,
    )

    assert len(survivors) == 30


def test_density_based_kills_more_in_crowds() -> None:
    """Crowded individuals should die more often than isolated ones."""
    random.seed(0)

    crowded = _population([1.0] * 12)
    crowded_positions = [np.array([5.0, 5.0, 0.1]) for _ in crowded]

    spread = _population([1.0] * 12)
    spread_positions = [
        np.array([float(i) * 8.0, float(i) * 8.0, 0.1])
        for i in range(len(spread))
    ]

    kwargs = {
        "method": "density_based",
        "current_generation": 1,
        "world_size": (100.0, 100.0),
        "density_locality_radius": 1.0,
        "density_critical_density": 1.0,
        "density_base_death_prob": 0.0,
        "density_max_death_prob": 1.0,
    }

    crowded_survivors = select_individuals(
        crowded,
        12,
        positions=crowded_positions,
        **kwargs,
    )
    spread_survivors = select_individuals(
        spread,
        12,
        positions=spread_positions,
        **kwargs,
    )

    assert len(crowded_survivors) < len(spread_survivors)


def test_density_based_falls_back_without_positions() -> None:
    """Missing positions should degrade to keeping everyone ranked."""
    population = _population([1.0, 2.0, 3.0])
    survivors = select_individuals(
        population,
        3,
        method="density_based",
        current_generation=1,
    )

    assert len(survivors) == 3


def test_zone_capacity_spreads_survivors_across_zones() -> None:
    """A zone must not exceed its share, even when it holds the fittest."""
    population = _population([9.0, 8.0, 7.0, 1.0])
    survivors = select_individuals(
        population,
        2,
        method="zone_capacity",
        current_generation=1,
        zone_indices=[0, 0, 0, 1],
        num_zones=2,
    )

    ids = {ind.unique_id for ind in survivors}
    assert len(survivors) == 2
    # The lone member of zone 1 survives despite the worst fitness.
    assert 3 in ids


def test_unknown_method_falls_back_to_fitness() -> None:
    """An unrecognised policy should not crash the run."""
    survivors = select_individuals(
        _population([1.0, 5.0, 3.0]),
        2,
        method="not_a_real_method",
        current_generation=1,
    )

    assert [ind.fitness for ind in survivors] == [5.0, 3.0]


def test_empty_population_selects_nothing() -> None:
    """Selection on an extinct population should return an empty list."""
    assert (
        select_individuals([], 5, method="fitness_based", current_generation=1)
        == []
    )
