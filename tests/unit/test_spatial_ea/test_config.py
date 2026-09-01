"""Test: spatial EA configuration loading."""

# Standard library
from pathlib import Path

# Third-party libraries
import pytest

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig

# The research prototype's own configuration, kept as the regression case for
# the legacy nested schema.
PROTOTYPE_CONFIG = (
    Path(__file__).resolve().parents[3] / "examples/spatial_ea/ea_config.yaml"
)


def _legacy() -> dict:
    """Build a representative nested configuration."""
    return {
        "incubation": {
            "enabled": True,
            "population_size": 25,
            "num_generations": 40,
            "mutation_rate": 0.7,
            "mutation_power": 0.3,
            "add_connection_rate": 0.06,
            "add_node_rate": 0.04,
            "crossover_rate": 0.8,
            "tournament_size": 4,
            "elitism_count": 3,
            "use_directional_fitness": True,
            "target_distance_min": 4.0,
            "target_distance_max": 9.0,
            "progress_weight": 0.6,
        },
        "population": {
            "size": 30,
            "num_generations": 12,
            "max_population_limit": 90,
            "min_population_limit": 2,
            "stop_on_limits": False,
        },
        "selection": {
            "pairing_radius": 10.0,
            "offspring_radius": 3.0,
            "target_population_size": 28,
            "selection_method": "density_based",
            "pairing_method": "mating_zone",
            "movement_bias": "assigned_zone",
            "max_age": 8,
            "enable_energy": True,
            "initial_energy": 120.0,
            "energy_depletion_rate": 15.0,
            "mating_energy_effect": "restore",
            "mating_energy_amount": 25.0,
            "num_mating_zones": 7,
            "mating_zone_center": [12.5, 12.5],
            "mating_zone_radius": 2.5,
            "min_zone_distance": 2.0,
            "zone_relocation_strategy": "event_driven",
            "zone_change_interval": 3,
            "locality_radius": 3.0,
            "critical_density": 5.0,
            "base_death_prob": 0.05,
            "max_density_death_prob": 0.4,
            "density_fitness_protection": 0.1,
        },
        "crossover": {"rate": 0.85},
        "mutation": {
            "rate": 0.75,
            "strength": 0.45,
            "add_connection_rate": 0.07,
            "add_node_rate": 0.02,
        },
        "simulation": {
            "time": 60.0,
            "control_clip_min": -1.5708,
            "control_clip_max": 1.5708,
            "use_periodic_boundaries": True,
        },
        "multi_robot": {
            "world_size": [25, 25, 0.2],
            "spawn_area": {
                "x_min": 0.1,
                "x_max": 24.9,
                "y_min": 0.1,
                "y_max": 24.9,
                "z": 0.05,
            },
            "min_spawn_distance": 1.0,
            "robot_size": 0.5,
        },
        "output": {
            "results_folder": "./__results__",
            "figures_folder": "./__figures__",
            "video_folder": "./__videos__",
        },
        "logging": {"print_generation_stats": False},
    }


def test_legacy_population_and_simulation_sections() -> None:
    """Population and simulation keys should map across."""
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.population_size == 30
    assert config.num_generations == 12
    assert config.target_population_size == 28
    assert config.max_population_limit == 90
    assert config.min_population_limit == 2
    assert config.stop_on_limits is False
    assert config.simulation_time == 60.0
    assert config.use_periodic_boundaries is True
    assert config.control_clip_min == pytest.approx(-1.5708)


def test_legacy_world_section() -> None:
    """The world size triple should split into planar size and height."""
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.world_size == (25.0, 25.0)
    assert config.world_z == 0.2
    assert config.spawn_x_max == 24.9
    assert config.spawn_z == 0.05
    assert config.robot_size == 0.5


def test_legacy_variation_section() -> None:
    """Crossover and mutation rates must reach the spatial phase.

    These sections were previously unmapped, leaving reproduction to run on
    hard-coded rates regardless of the configuration.
    """
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.crossover_rate == 0.85
    assert config.mutation_rate == 0.75
    assert config.mutation_strength == 0.45
    assert config.add_connection_rate == 0.07
    assert config.add_node_rate == 0.02


def test_legacy_energy_section() -> None:
    """The whole energy model should carry over, not just the initial value."""
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.enable_energy is True
    assert config.initial_energy == 120.0
    assert config.energy_depletion_rate == 15.0
    assert config.mating_energy_effect == "restore"
    assert config.mating_energy_amount == 25.0
    assert config.max_age == 8


def test_legacy_incubation_section() -> None:
    """Incubation, including its directional fitness settings, maps across."""
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.incubation_enabled is True
    assert config.incubation_population_size == 25
    assert config.incubation_num_generations == 40
    assert config.incubation_crossover_rate == 0.8
    assert config.incubation_tournament_size == 4
    assert config.incubation_elitism_count == 3
    assert config.use_directional_fitness is True
    assert config.target_distance_min == 4.0
    assert config.progress_weight == 0.6


def test_legacy_movement_bias_is_not_silently_disabled() -> None:
    """A requested bias must arrive with a usable step size.

    ``movement_step_size`` was previously left at its zero default, and the
    analytical movement helper returns early at zero, so loading any legacy
    file disabled movement bias no matter what the file asked for.
    """
    config = SpatialEAConfig.from_legacy_dict(_legacy())

    assert config.movement_bias == "assigned_zone"
    assert config.movement_step_size > 0.0


def test_legacy_movement_step_size_stays_zero_without_a_bias() -> None:
    """No bias means no analytical movement."""
    legacy = _legacy()
    legacy["selection"]["movement_bias"] = "none"

    config = SpatialEAConfig.from_legacy_dict(legacy)

    assert config.movement_bias == "none"
    assert config.movement_step_size == 0.0


def test_legacy_deprecated_dynamic_zones_flag() -> None:
    """The old boolean spelling should translate to a strategy name."""
    legacy = _legacy()
    del legacy["selection"]["zone_relocation_strategy"]

    legacy["selection"]["dynamic_mating_zones"] = True
    assert (
        SpatialEAConfig.from_legacy_dict(legacy).zone_relocation_strategy
        == "generation_interval"
    )

    legacy["selection"]["dynamic_mating_zones"] = False
    assert (
        SpatialEAConfig.from_legacy_dict(legacy).zone_relocation_strategy
        == "static"
    )


def test_legacy_defaults_centre_the_mating_zone() -> None:
    """Without an explicit centre, the zone sits in the middle of the world."""
    legacy = _legacy()
    del legacy["selection"]["mating_zone_center"]

    config = SpatialEAConfig.from_legacy_dict(legacy)

    assert config.mating_zone_center == (12.5, 12.5)


def test_from_yaml_detects_the_legacy_schema(tmp_path: Path) -> None:
    """Nested section keys should route to the legacy reader."""
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "population:\n  size: 11\nselection:\n  pairing_radius: 4.0\n",
        encoding="utf-8",
    )

    config = SpatialEAConfig.from_yaml(path)

    assert config.population_size == 11
    assert config.pairing_radius == 4.0


def test_from_yaml_reads_the_native_schema(tmp_path: Path) -> None:
    """A flat file maps straight onto the fields."""
    path = tmp_path / "native.yaml"
    path.write_text(
        "population_size: 13\nselection_method: age_based\n",
        encoding="utf-8",
    )

    config = SpatialEAConfig.from_yaml(path)

    assert config.population_size == 13
    assert config.selection_method == "age_based"


def test_from_yaml_rejects_a_non_mapping(tmp_path: Path) -> None:
    """A list at the document root is a configuration error."""
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        SpatialEAConfig.from_yaml(path)


def test_from_yaml_of_an_empty_file(tmp_path: Path) -> None:
    """An empty document should fall back to the defaults."""
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")

    assert SpatialEAConfig.from_yaml(path).population_size == 20


def test_effective_target_population_size() -> None:
    """Selection aims at the explicit target, or the population size."""
    assert (
        SpatialEAConfig(
            population_size=20,
        ).effective_target_population_size
        == 20
    )
    assert (
        SpatialEAConfig(
            population_size=20,
            target_population_size=35,
        ).effective_target_population_size
        == 35
    )


@pytest.mark.skipif(
    not PROTOTYPE_CONFIG.exists(),
    reason="research prototype configuration is not present",
)
def test_the_prototype_configuration_loads_intact() -> None:
    """The shipped prototype config should reach the engine unchanged."""
    config = SpatialEAConfig.from_yaml(PROTOTYPE_CONFIG)

    assert config.population_size == 30
    assert config.world_size == (25.0, 25.0)
    assert config.num_mating_zones == 17
    assert config.zone_relocation_strategy == "event_driven"
    assert config.pairing_method == "mating_zone"
    assert config.movement_bias == "assigned_zone"
    assert config.movement_step_size > 0.0
    assert config.selection_method == "fitness_based"
    assert config.use_periodic_boundaries is True
    assert config.use_directional_fitness is True
    assert config.enable_energy is True
    assert config.energy_depletion_rate == 10.0
    assert config.mating_energy_effect == "cost"
    assert config.mating_energy_amount == 35.0
    assert config.simulation_time == 60.0


def test_spawn_area_is_clamped_into_the_world() -> None:
    """Shrinking the world must not leave the spawn box outside it.

    The bounds and ``world_size`` are independent fields, so setting only the
    world used to spawn the whole population outside it and then clip it onto
    the world edge — silently.
    """
    config = SpatialEAConfig(world_size=(4.0, 4.0))

    assert config.spawn_x_max <= 4.0
    assert config.spawn_y_max <= 4.0
    assert config.spawn_x_min >= 0.0
    assert config.spawn_y_min >= 0.0
    # Still a usable box, not collapsed to a point.
    assert config.spawn_x_max > config.spawn_x_min


def test_explicit_spawn_area_inside_the_world_is_untouched() -> None:
    """A configuration that already fits is left exactly as written."""
    config = SpatialEAConfig(
        world_size=(25.0, 25.0),
        spawn_x_min=0.1,
        spawn_x_max=24.9,
        spawn_y_min=0.1,
        spawn_y_max=24.9,
    )

    assert config.spawn_x_min == 0.1
    assert config.spawn_x_max == 24.9
    assert config.spawn_y_max == 24.9


def test_spawn_clamp_survives_model_copy() -> None:
    """Overriding the world through model_copy re-runs the clamp."""
    config = SpatialEAConfig().model_copy(update={"world_size": (5.0, 5.0)})
    revalidated = SpatialEAConfig(**config.model_dump())

    assert revalidated.spawn_x_max <= 5.0


def test_recording_flags_map_from_the_legacy_video_section() -> None:
    """The prototype's ``video:`` section reaches the engine."""
    legacy = _legacy()
    legacy["video"] = {
        "record_generation_videos": True,
        "save_generation_snapshots": True,
    }

    config = SpatialEAConfig.from_legacy_dict(legacy)

    assert config.record_generation_videos is True
    assert config.save_generation_snapshots is True


def test_recording_defaults_are_off() -> None:
    """Recording is opt-in."""
    config = SpatialEAConfig()

    assert config.record_generation_videos is False
    assert config.save_generation_snapshots is False
