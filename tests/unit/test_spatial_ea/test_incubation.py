"""Test: the incubation phase that precedes spatial evolution."""

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.genetics import create_initial_hyperneat_genome
from ariel.spatial_ea.incubation import (
    IncubationEvolution,
    seed_spatial_population_from_incubation,
)
from ariel.spatial_ea.individual import SpatialIndividual


def _config(**overrides: object) -> SpatialEAConfig:
    """Build a fast incubation config."""
    defaults = {
        "incubation_population_size": 4,
        "incubation_num_generations": 2,
        "incubation_elitism_count": 1,
        "simulation_time": 0.15,
        "print_generation_stats": False,
        "save_results": False,
    }
    return SpatialEAConfig(**{**defaults, **overrides})


def test_run_produces_an_evaluated_population() -> None:
    """Incubation should hand back a fully evaluated population."""
    incubator = IncubationEvolution(config=_config())
    population, next_id = incubator.run()

    assert len(population) == 4
    assert all(individual.evaluated for individual in population)
    assert all(individual.genotype["nodes"] for individual in population)
    assert next_id >= len(population)
    assert len(incubator.best_fitness_history) == 2


def test_unique_ids_never_repeat() -> None:
    """Every individual created during incubation gets its own id."""
    incubator = IncubationEvolution(config=_config(), next_unique_id=100)
    population, next_id = incubator.run()

    ids = [individual.unique_id for individual in population]
    assert len(set(ids)) == len(ids)
    assert min(ids) >= 100
    assert next_id > max(ids)


def test_elites_carry_over_unchanged() -> None:
    """The best genome should survive into the next generation intact."""
    config = _config(
        incubation_elitism_count=1,
        incubation_mutation_probability=1.0,
        incubation_mutation_strength=2.0,
    )
    incubator = IncubationEvolution(config=config)
    incubator.initialize_population()
    for i, individual in enumerate(incubator.population):
        individual.fitness = float(i)

    best_before = max(
        incubator.population,
        key=lambda individual: individual.fitness,
    )
    weights_before = [
        conn.weight for conn in best_before.genotype["connections"]
    ]

    incubator.create_next_generation()
    elite = incubator.population[0]

    assert [
        conn.weight for conn in elite.genotype["connections"]
    ] == weights_before
    assert elite.parent_ids == [best_before.unique_id]
    assert elite.evaluated is False


def test_tournament_selection_prefers_fitness() -> None:
    """With a tournament as large as the population, the best always wins."""
    incubator = IncubationEvolution(
        config=_config(incubation_tournament_size=4),
    )
    incubator.initialize_population()
    for i, individual in enumerate(incubator.population):
        individual.fitness = float(i)

    for _ in range(10):
        assert incubator.tournament_selection().fitness == 3.0


def test_cloning_when_crossover_does_not_fire() -> None:
    """A zero crossover rate should clone the first parent's topology."""
    config = _config(
        incubation_crossover_rate=0.0,
        incubation_mutation_probability=0.0,
        incubation_add_connection_rate=0.0,
        incubation_add_node_rate=0.0,
    )
    incubator = IncubationEvolution(config=config)
    parent1 = SpatialIndividual(
        unique_id=1,
        genotype=create_initial_hyperneat_genome(),
    )
    parent2 = SpatialIndividual(
        unique_id=2,
        genotype=create_initial_hyperneat_genome(),
    )

    offspring = incubator.create_offspring(parent1, parent2)

    assert [conn.weight for conn in offspring.genotype["connections"]] == [
        conn.weight for conn in parent1.genotype["connections"]
    ]
    assert offspring.parent_ids == [1, 2]


def test_seeding_takes_the_best_first() -> None:
    """The spatial population should be built from the fittest incubees."""
    incubated = [
        SpatialIndividual(
            unique_id=i,
            fitness=float(i),
            evaluated=True,
            genotype=create_initial_hyperneat_genome(),
        )
        for i in range(5)
    ]

    population, next_id = seed_spatial_population_from_incubation(
        incubated,
        target_population_size=3,
        generation=0,
        next_unique_id=50,
        initial_energy=42.0,
    )

    assert [individual.fitness for individual in population] == [4.0, 3.0, 2.0]
    assert [individual.unique_id for individual in population] == [50, 51, 52]
    assert next_id == 53
    assert all(individual.energy == 42.0 for individual in population)
    assert population[0].parent_ids == [4]


def test_seeding_clones_when_incubation_is_too_small() -> None:
    """A short incubation should be recycled rather than truncate the run."""
    incubated = [
        SpatialIndividual(
            unique_id=0,
            fitness=1.0,
            genotype=create_initial_hyperneat_genome(),
        ),
    ]

    population, _ = seed_spatial_population_from_incubation(
        incubated,
        target_population_size=4,
        generation=0,
        next_unique_id=0,
    )

    assert len(population) == 4
    assert len({individual.unique_id for individual in population}) == 4
    # Clones, not aliases.
    assert population[0].genotype is not population[1].genotype


def test_seeding_from_nothing() -> None:
    """An empty incubation should yield an empty population."""
    population, next_id = seed_spatial_population_from_incubation(
        [],
        target_population_size=5,
        generation=0,
        next_unique_id=7,
    )

    assert population == []
    assert next_id == 7
