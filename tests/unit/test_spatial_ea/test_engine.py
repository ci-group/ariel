"""Test: the spatial EA generation loop."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea import SpatialEA
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.individual import SpatialIndividual


def _config(**overrides: object) -> SpatialEAConfig:
    """Build a small, fast run configuration."""
    defaults = {
        "population_size": 4,
        "num_generations": 2,
        "simulation_time": 0.15,
        "world_size": (10.0, 10.0),
        "spawn_x_min": 1.0,
        "spawn_x_max": 9.0,
        "spawn_y_min": 1.0,
        "spawn_y_max": 9.0,
        "min_spawn_distance": 0.5,
        "print_generation_stats": False,
        "save_results": False,
        "save_plots": False,
    }
    return SpatialEAConfig(**{**defaults, **overrides})


def test_engine_starts_empty() -> None:
    """A fresh engine holds no population until it is initialised."""
    engine = SpatialEA(config=_config())

    assert engine.population == []
    assert engine.population_size == 0
    assert engine.get_best_individual() is None


def test_initialize_population_places_everyone() -> None:
    """Initialisation should give every individual a genome and a position."""
    engine = SpatialEA(config=_config(population_size=5))
    engine.initialize_population()

    assert engine.population_size == 5
    assert len({ind.unique_id for ind in engine.population}) == 5
    for individual in engine.population:
        assert individual.genotype["nodes"]
        assert individual.spawn_position is not None
        assert 0.0 <= individual.spawn_position[0] <= 10.0


def test_initialize_population_respects_min_spawn_distance() -> None:
    """Spawn points should honour the configured separation."""
    config = _config(population_size=4, min_spawn_distance=1.5)
    engine = SpatialEA(config=config)
    engine.initialize_population()

    positions = engine.positions
    for i, first in enumerate(positions):
        for second in positions[i + 1 :]:
            distance = float(np.linalg.norm(first[:2] - second[:2]))
            assert distance >= 1.5 - 1e-6


def test_run_returns_an_evaluated_best_individual() -> None:
    """A completed run should hand back an evaluated survivor."""
    engine = SpatialEA(config=_config())
    best = engine.run(generations=1)

    assert best is not None
    assert best.evaluated is True
    assert np.isfinite(best.fitness)
    assert best in engine.population


def test_run_records_one_row_per_generation() -> None:
    """The collector should mirror the generations that actually ran."""
    engine = SpatialEA(config=_config())
    engine.run(generations=3)

    collector = engine.data_collector
    assert collector.generations == [0, 1, 2]
    assert len(collector.fitness_best) == 3
    assert len(collector.population_size) == 3
    assert collector.stopped_early is False


def test_reproduction_produces_offspring_of_the_next_generation() -> None:
    """Pairing individuals placed together should yield mutated children."""
    config = _config(population_size=4, pairing_radius=100.0)
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    before = engine.population_size

    engine.create_next_generation(spawned)

    assert engine.population_size > before
    offspring = [
        ind
        for ind in engine.population
        if ind.generation == engine.generation + 1
    ]
    assert len(offspring) == 4
    assert all(ind.evaluated is False for ind in offspring)
    assert all(ind.parent_ids for ind in offspring)
    assert engine.paired_indices


def test_no_pairs_form_when_everyone_is_out_of_range() -> None:
    """A tiny pairing radius should leave the population childless."""
    config = _config(population_size=4, pairing_radius=1e-6)
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    assert engine.population_size == 4
    assert engine.data_collector.mating_pairs == [0]


def test_energy_depletes_each_generation() -> None:
    """Enabling energy should charge every individual per generation."""
    config = _config(
        population_size=4,
        pairing_radius=1e-6,
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=25.0,
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    assert all(ind.energy == 75.0 for ind in engine.population)


def test_energy_is_left_alone_when_disabled() -> None:
    """With energy off, nothing should be deducted."""
    config = _config(
        population_size=4,
        pairing_radius=1e-6,
        enable_energy=False,
        initial_energy=100.0,
        energy_depletion_rate=25.0,
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    assert all(ind.energy == 100.0 for ind in engine.population)


def test_mating_costs_energy() -> None:
    """Parents that reproduce pay the configured mating cost."""
    config = _config(
        population_size=2,
        pairing_radius=100.0,
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=10.0,
        mating_energy_effect="cost",
        mating_energy_amount=30.0,
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    parents = [ind for ind in engine.population if ind.generation == 0]
    # 100 - 10 depletion - 30 mating.
    assert all(ind.energy == 60.0 for ind in parents)


def test_mating_restores_energy() -> None:
    """The restore policy should reset parents to full energy."""
    config = _config(
        population_size=2,
        pairing_radius=100.0,
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=10.0,
        mating_energy_effect="restore",
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    parents = [ind for ind in engine.population if ind.generation == 0]
    assert all(ind.energy == 100.0 for ind in parents)


def test_run_stops_below_the_minimum_population() -> None:
    """Dropping under the floor should end the run with a recorded reason."""
    config = _config(population_size=4, min_population_limit=10)
    engine = SpatialEA(config=config)

    engine.run(generations=5)

    assert engine.data_collector.stopped_early is True
    assert "extinction" in engine.data_collector.stop_reason.lower()
    assert engine.data_collector.generations == [0]


def test_energy_starvation_drives_the_population_extinct() -> None:
    """Energy selection with no offspring should empty the world."""
    config = _config(
        population_size=4,
        pairing_radius=1e-6,
        selection_method="energy_based",
        enable_energy=True,
        initial_energy=100.0,
        energy_depletion_rate=60.0,
        min_population_limit=1,
        num_generations=6,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=6)

    assert engine.population == []
    assert engine.get_best_individual() is None
    assert engine.data_collector.stopped_early is True
    assert "extinction" in engine.data_collector.stop_reason.lower()
    # The extinct generation is recorded so it still appears in plots.
    assert engine.data_collector.population_size[-1] == 0


def test_run_stops_on_population_explosion() -> None:
    """Exceeding the maximum population should end the run."""
    config = _config(population_size=4, max_population_limit=3)
    engine = SpatialEA(config=config)
    engine.initialize_population()

    engine.run(generations=5)

    assert engine.data_collector.stopped_early is True
    assert "maximum limit" in engine.data_collector.stop_reason
    assert engine.data_collector.planned_generations == 5


def test_limits_are_ignored_when_disabled() -> None:
    """Turning limits off should let the run continue regardless."""
    config = _config(
        population_size=4,
        max_population_limit=1,
        stop_on_limits=False,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=2)

    assert engine.data_collector.stopped_early is False


def test_evaluation_happens_once_per_individual() -> None:
    """Already-evaluated individuals keep their inherited fitness."""
    engine = SpatialEA(config=_config(population_size=3))
    engine.initialize_population()
    engine.evaluate_population_fitness()

    scores = [ind.fitness for ind in engine.population]
    engine.evaluate_population_fitness()

    assert [ind.fitness for ind in engine.population] == scores


def test_zones_are_assigned_to_the_nearest_centre() -> None:
    """Each individual binds to the closest mating zone."""
    config = _config(population_size=2, num_mating_zones=2)
    engine = SpatialEA(config=config)
    engine.current_zone_centers = [(1.0, 1.0), (9.0, 9.0)]
    engine.population = [
        SpatialIndividual(
            unique_id=0,
            spawn_position=np.array([1.5, 1.5, 0.1]),
        ),
        SpatialIndividual(
            unique_id=1,
            spawn_position=np.array([8.5, 8.5, 0.1]),
        ),
    ]

    engine._assign_zones_to_population()

    assert engine.assigned_zones == {0: 0, 1: 1}
    assert engine.population[1].assigned_zone == 1


def test_event_driven_relocation_moves_only_the_named_zones() -> None:
    """Relocating one zone should leave the others where they are."""
    engine = SpatialEA(config=_config(num_mating_zones=3))
    engine.current_zone_centers = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]

    engine.relocate_mating_zones({1})

    assert engine.current_zone_centers[0] == (1.0, 1.0)
    assert engine.current_zone_centers[2] == (9.0, 9.0)
    assert len(engine.current_zone_centers) == 3


def test_analytical_movement_is_used_when_physics_is_off() -> None:
    """The cheap movement mode should still displace the population."""
    config = _config(
        population_size=3,
        use_physical_movement_phase=False,
        movement_bias="nearest_zone",
        movement_step_size=1.0,
        num_mating_zones=1,
        mating_zone_center=(5.0, 5.0),
        pairing_radius=1e-6,
    )
    engine = SpatialEA(config=config)
    engine.initialize_population()
    before = [pos.copy() for pos in engine.positions]

    spawned = engine.spawn_population()
    engine.evaluate_population_fitness()
    engine.create_next_generation(spawned)

    after = engine.positions[: len(before)]
    moved = [
        not np.allclose(first[:2], second[:2])
        for first, second in zip(before, after, strict=True)
    ]
    assert any(moved)


def test_save_results_writes_the_expected_files(tmp_path) -> None:
    """A saved run should leave statistics and controllers on disk."""
    config = _config(
        population_size=3,
        result_folder=tmp_path,
        save_results=True,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=1)

    names = sorted(path.name for path in tmp_path.iterdir())
    assert any(name.startswith("evolution_data_") for name in names)
    assert any(name.endswith(".csv") for name in names)
    assert any(name.startswith("final_controllers_") for name in names)
    assert any(name.startswith("best_controller_") for name in names)


@pytest.mark.parametrize(
    "method",
    [
        "fitness_based",
        "parents_die",
        "age_based",
        "probabilistic_age",
        "energy_based",
        "density_based",
        "zone_capacity",
    ],
)
def test_every_selection_method_completes_a_run(method: str) -> None:
    """No selection policy should crash the loop."""
    config = _config(
        population_size=4,
        pairing_radius=100.0,
        selection_method=method,
        max_population_limit=1000,
    )
    engine = SpatialEA(config=config)
    engine.run(generations=2)

    assert engine.data_collector.generations == [0, 1]
