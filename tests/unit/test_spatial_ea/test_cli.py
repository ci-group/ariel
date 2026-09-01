"""Test: the spatial EA command-line interface."""

# Standard library
from pathlib import Path

# Local libraries
from ariel.spatial_ea.cli import build_parser, config_from_args
from ariel.spatial_ea.config import SpatialEAConfig


def _config(argv: list[str]) -> SpatialEAConfig:
    """Parse arguments and resolve them into a configuration."""
    return config_from_args(build_parser().parse_args(argv))


def test_no_arguments_gives_the_defaults() -> None:
    """An empty command line should leave every default in place."""
    config = _config([])
    defaults = SpatialEAConfig()

    assert config.population_size == defaults.population_size
    assert config.selection_method == defaults.selection_method
    assert config.world_size == defaults.world_size


def test_every_config_field_has_a_flag() -> None:
    """The parser is generated from the model, so it cannot drift."""
    flags = {
        option
        for action in build_parser()._actions
        for option in action.option_strings
    }

    for name in SpatialEAConfig.model_fields:
        assert f"--{name.replace('_', '-')}" in flags


def test_scalar_overrides_are_typed() -> None:
    """Numeric and path flags should parse into the right types."""
    config = _config([
        "--population-size",
        "17",
        "--simulation-time",
        "2.5",
        "--result-folder",
        "/tmp/spatial",
    ])

    assert config.population_size == 17
    assert isinstance(config.population_size, int)
    assert config.simulation_time == 2.5
    assert config.result_folder == Path("/tmp/spatial")


def test_boolean_flags_take_an_explicit_value() -> None:
    """Booleans are settable in both directions, not just on."""
    assert _config([
        "--use-periodic-boundaries",
        "true",
    ]).use_periodic_boundaries
    assert not _config(
        ["--use-periodic-boundaries", "false"],
    ).use_periodic_boundaries
    assert not _config(["--enable-energy", "no"]).enable_energy


def test_tuple_flags_take_two_values() -> None:
    """World size and zone centre are pairs."""
    config = _config([
        "--world-size",
        "25",
        "30",
        "--mating-zone-center",
        "12.5",
        "12.5",
    ])

    assert config.world_size == (25.0, 30.0)
    assert config.mating_zone_center == (12.5, 12.5)


def test_choice_flags_are_constrained() -> None:
    """Literal-typed fields expose their allowed values as choices."""
    parser = build_parser()
    selection = next(
        action
        for action in parser._actions
        if action.dest == "selection_method"
    )

    assert selection.choices is not None
    assert "density_based" in selection.choices
    assert "probabilistic_age" in selection.choices

    assert (
        _config(
            ["--selection-method", "density_based"],
        ).selection_method
        == "density_based"
    )


def test_flags_override_a_config_file(tmp_path: Path) -> None:
    """An explicit flag wins over the value in the file."""
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        "population_size: 40\nnum_generations: 9\n",
        encoding="utf-8",
    )

    config = _config([
        "--config",
        str(config_file),
        "--population-size",
        "5",
    ])

    assert config.population_size == 5
    assert config.num_generations == 9


def test_legacy_config_file_is_accepted(tmp_path: Path) -> None:
    """The prototype's nested schema loads through the same flag."""
    config_file = tmp_path / "legacy.yaml"
    config_file.write_text(
        "population:\n"
        "  size: 12\n"
        "selection:\n"
        "  selection_method: energy_based\n"
        "  movement_bias: assigned_zone\n",
        encoding="utf-8",
    )

    config = _config(["--config", str(config_file)])

    assert config.population_size == 12
    assert config.selection_method == "energy_based"
    assert config.movement_bias == "assigned_zone"
