"""Test: spatial EA bootstrap helpers and package surface."""

# Local libraries
from ariel import spatial_ea
from ariel.spatial_ea import build_default_bootstrap, spatial_ea_config


def test_build_default_bootstrap() -> None:
    """The bootstrap helper should compile a world with one robot in it."""
    bootstrap = build_default_bootstrap()

    assert bootstrap.num_joints > 0
    assert bootstrap.model.nu == bootstrap.num_joints
    assert bootstrap.world is not None
    assert bootstrap.robot is not None


def test_default_config_is_usable() -> None:
    """The module-level default config should describe a runnable setup."""
    assert spatial_ea_config.population_size > 0
    assert spatial_ea_config.num_generations > 0
    assert spatial_ea_config.world_size == (10.0, 10.0)
    assert spatial_ea_config.min_population_limit >= 0


def test_public_surface_is_importable() -> None:
    """Everything named in ``__all__`` should actually be exported."""
    for name in spatial_ea.__all__:
        assert hasattr(spatial_ea, name), name


def test_public_surface_has_no_duplicates() -> None:
    """A name listed twice hides a merge mistake.

    Ordering is left to ruff's RUF022, which uses a natural sort rather than
    Python's, so it is not asserted here.
    """
    assert len(spatial_ea.__all__) == len(set(spatial_ea.__all__))
