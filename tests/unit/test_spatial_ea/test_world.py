"""Test: shared-world construction for the spatial EA."""

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.world import (
    build_single_robot_world,
    generate_spawn_positions,
    set_robot_yaw,
    spawn_population_in_world,
)

WORLD_SIZE = (10.0, 10.0, 0.1)


def test_generate_spawn_positions_respects_separation() -> None:
    """Sampled positions should stay apart and inside the requested box."""
    positions = generate_spawn_positions(
        population_size=8,
        spawn_x_range=(0.5, 9.5),
        spawn_y_range=(0.5, 9.5),
        spawn_z=0.1,
        min_spawn_distance=1.0,
    )

    assert len(positions) == 8
    for position in positions:
        assert 0.5 <= position[0] <= 9.5
        assert 0.5 <= position[1] <= 9.5
        assert position[2] == 0.1

    for i, first in enumerate(positions):
        for second in positions[i + 1 :]:
            distance = float(np.linalg.norm(first[:2] - second[:2]))
            assert distance >= 1.0 - 1e-9


def test_spawn_population_resolves_per_robot_handles() -> None:
    """Every robot should get its own core geom and free joint."""
    population = [SpatialIndividual(unique_id=i) for i in range(3)]
    positions = [
        np.array([2.0, 2.0, 0.1]),
        np.array([5.0, 5.0, 0.1]),
        np.array([8.0, 8.0, 0.1]),
    ]

    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)

    assert len(spawned.core_geom_ids) == 3
    assert len(set(spawned.core_geom_ids)) == 3
    assert all(geom_id >= 0 for geom_id in spawned.core_geom_ids)

    # Named as robot1_core, robot2_core, robot3_core.
    for i, geom_id in enumerate(spawned.core_geom_ids):
        name = mujoco.mj_id2name(
            spawned.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            geom_id,
        )
        assert name == f"robot{i + 1}_core"

    assert all(adr >= 0 for adr in spawned.free_joint_qpos_adr)
    assert len(set(spawned.free_joint_qpos_adr)) == 3
    assert spawned.num_joints > 0
    assert spawned.model.nu == spawned.num_joints * 3


def test_spawn_population_places_robots_where_asked() -> None:
    """Reported core positions should match the requested spawn points."""
    population = [SpatialIndividual(unique_id=i) for i in range(2)]
    positions = [np.array([2.0, 3.0, 0.1]), np.array([7.0, 8.0, 0.1])]

    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)
    actual = spawned.core_positions()

    for requested, reported in zip(positions, actual, strict=True):
        assert abs(reported[0] - requested[0]) < 0.5
        assert abs(reported[1] - requested[1]) < 0.5

    # The individuals learn where they were put.
    assert population[0].robot_index == 0
    assert population[1].robot_index == 1
    assert np.allclose(population[1].spawn_position, positions[1])


def test_free_joint_stride_matches_qpos_layout() -> None:
    """Free joints should be spaced by seven plus the joint count.

    The mating controller slices joint angles out of ``qpos`` using that
    stride, so the layout assumption is load-bearing.
    """
    population = [SpatialIndividual(unique_id=i) for i in range(3)]
    positions = [np.array([float(i) * 3.0 + 1.0, 1.0, 0.1]) for i in range(3)]

    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)
    stride = 7 + spawned.num_joints

    assert spawned.free_joint_qpos_adr == [0, stride, 2 * stride]


def test_set_robot_yaw_writes_a_unit_quaternion() -> None:
    """Setting a heading should leave a normalised quaternion behind."""
    _, model, _, qpos_adr = build_single_robot_world(WORLD_SIZE)
    data = mujoco.MjData(model)

    set_robot_yaw(data, qpos_adr, np.pi / 2.0)
    quaternion = data.qpos[qpos_adr + 3 : qpos_adr + 7]

    assert float(np.linalg.norm(quaternion)) == 1.0
    assert quaternion[0] == np.cos(np.pi / 4.0)
    assert quaternion[3] == np.sin(np.pi / 4.0)


def test_set_robot_yaw_ignores_missing_joints() -> None:
    """A negative address should be a no-op rather than an index error."""
    _, model, _, _ = build_single_robot_world(WORLD_SIZE)
    data = mujoco.MjData(model)
    before = data.qpos.copy()

    set_robot_yaw(data, -1, 1.0)

    assert np.array_equal(data.qpos, before)


def test_build_single_robot_world() -> None:
    """The isolated evaluation world should hold exactly one robot."""
    _, model, core_geom_id, qpos_adr = build_single_robot_world(WORLD_SIZE)

    assert core_geom_id >= 0
    assert qpos_adr == 0
    assert model.nu > 0
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, core_geom_id)
    assert name == "robot1_core"
