"""Test: obstacle generation and MjSpec attachment."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.simulation.environments import (
    Obstacle,
    ObstacleConfig,
    ObstacleType,
    SimpleFlatWorld,
    attach_obstacles,
    generate_obstacles,
)


def test_none_returns_empty_list() -> None:
    cfg = ObstacleConfig(obstacle_type=ObstacleType.NONE)
    assert generate_obstacles(cfg) == []


def test_custom_returns_user_list() -> None:
    user = [Obstacle("sphere", np.array([0, 0, 1.0]), np.array([0.1]))]
    cfg = ObstacleConfig(obstacle_type=ObstacleType.CUSTOM, custom_obstacles=user)
    out = generate_obstacles(cfg)
    assert len(out) == 1 and out[0].geom_type == "sphere"


def test_forest_seed_reproducibility() -> None:
    cfg = ObstacleConfig(
        obstacle_type=ObstacleType.FOREST, num_obstacles=15,
        arena_size=(2.0, 2.0, 1.5), seed=42,
    )
    a = generate_obstacles(cfg)
    b = generate_obstacles(cfg)
    assert len(a) == len(b)
    for oa, ob in zip(a, b):
        np.testing.assert_array_equal(oa.position, ob.position)
        np.testing.assert_array_equal(oa.size, ob.size)


def test_safe_zone_excludes_spawn() -> None:
    spawn = np.array([[0.0, 0.0, 0.0]])
    cfg = ObstacleConfig(
        obstacle_type=ObstacleType.FOREST, num_obstacles=50,
        arena_size=(2.0, 2.0, 1.5), safe_zone_radius=0.8,
        safe_zone_centers=spawn, seed=0,
    )
    obstacles = generate_obstacles(cfg)
    for o in obstacles:
        assert np.linalg.norm(o.position[:2] - spawn[0, :2]) >= 0.8


def test_min_spacing_enforced() -> None:
    cfg = ObstacleConfig(
        obstacle_type=ObstacleType.URBAN, num_obstacles=20,
        arena_size=(3.0, 3.0, 2.0), min_spacing=0.5, seed=1,
    )
    obstacles = generate_obstacles(cfg)
    positions = np.stack([o.position[:2] for o in obstacles])
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            assert np.linalg.norm(positions[i] - positions[j]) >= 0.5 - 1e-9


def test_indoor_emits_walls_and_ceiling() -> None:
    cfg = ObstacleConfig(
        obstacle_type=ObstacleType.INDOOR, num_obstacles=8,
        arena_size=(1.5, 1.5, 1.5), seed=0,
    )
    out = generate_obstacles(cfg)
    # First 5 are 4 walls + ceiling
    assert all(o.geom_type == "box" for o in out[:5])
    assert len(out) >= 5


def test_attach_obstacles_grows_ngeom() -> None:
    world = SimpleFlatWorld()
    base_model = world.spec.compile()
    base_ngeom = base_model.ngeom

    # Build new world for the actual attachment + compile (compile is one-shot)
    world = SimpleFlatWorld()
    cfg = ObstacleConfig(
        obstacle_type=ObstacleType.FOREST, num_obstacles=10,
        arena_size=(2.0, 2.0, 1.5), seed=0,
    )
    obstacles = generate_obstacles(cfg)
    n_attached = attach_obstacles(world.spec, obstacles)
    model = world.spec.compile()
    assert n_attached == len(obstacles)
    assert model.ngeom == base_ngeom + n_attached


def test_attach_obstacles_visual_only() -> None:
    world = SimpleFlatWorld()
    obstacles = [Obstacle("sphere", np.array([0.5, 0.5, 1.0]), np.array([0.1]))]
    attach_obstacles(world.spec, obstacles, collidable=False, name_prefix="marker")
    model = world.spec.compile()
    # Find the marker geom and verify contype/conaffinity are zero
    found = False
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name == "marker_geom_0":
            assert int(model.geom(i).contype) == 0
            assert int(model.geom(i).conaffinity) == 0
            found = True
            break
    assert found, "marker_geom_0 not present in compiled model"


def test_attach_obstacles_rejects_bad_geom_type() -> None:
    world = SimpleFlatWorld()
    bad = [Obstacle("torus", np.array([0, 0, 1.0]), np.array([0.1, 0.1]))]
    with pytest.raises(ValueError, match="Unsupported geom_type"):
        attach_obstacles(world.spec, bad)


def test_all_geom_types_compile() -> None:
    """Forest (cylinder) + urban (box) + random (mixed) all compile without error."""
    for kind in (ObstacleType.FOREST, ObstacleType.URBAN, ObstacleType.RANDOM,
                 ObstacleType.GATES, ObstacleType.INDOOR):
        world = SimpleFlatWorld()
        cfg = ObstacleConfig(
            obstacle_type=kind, num_obstacles=8,
            arena_size=(2.0, 2.0, 1.5), seed=0,
        )
        attach_obstacles(world.spec, generate_obstacles(cfg))
        world.spec.compile()
