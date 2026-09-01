"""Test: the shared-world mating movement phase."""

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.genetics import create_initial_hyperneat_genome
from ariel.spatial_ea.individual import SpatialIndividual
from ariel.spatial_ea.movement import (
    MatingController,
    build_substrates,
    run_mating_movement_phase,
    sensor_inputs,
)
from ariel.spatial_ea.world import spawn_population_in_world

WORLD_SIZE = (10.0, 10.0, 0.1)
NUM_JOINTS = 8


def _population(count: int) -> list[SpatialIndividual]:
    """Build individuals with random genomes."""
    return [
        SpatialIndividual(
            unique_id=i,
            genotype=create_initial_hyperneat_genome(),
        )
        for i in range(count)
    ]


def _controller(
    population: list[SpatialIndividual],
    **overrides: object,
) -> MatingController:
    """Build a controller over a population with real substrates."""
    defaults = {
        "population": population,
        "substrates": build_substrates(population, NUM_JOINTS),
        "num_joints": NUM_JOINTS,
        "control_clip_min": -1.5708,
        "control_clip_max": 1.5708,
        "world_size": (10.0, 10.0),
    }
    return MatingController(**{**defaults, **overrides})


def test_build_substrates_one_per_individual() -> None:
    """Each individual gets its own decoded controller."""
    population = _population(3)
    substrates = build_substrates(population, NUM_JOINTS)

    assert len(substrates) == 3
    assert len({id(s) for s in substrates}) == 3
    assert substrates[0].weights_hidden_output.shape[1] == NUM_JOINTS


def test_sensor_input_layout() -> None:
    """The input vector is joints, four oscillators, heading, then bias."""
    population = _population(2)
    positions = [np.array([2.0, 2.0, 0.1]), np.array([6.0, 6.0, 0.1])]
    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)

    heading = np.array([0.6, 0.8])
    inputs = sensor_inputs(spawned.data, 1, spawned.num_joints, heading)

    assert len(inputs) == spawned.num_joints + 4 + 2 + 1
    assert inputs[-3] == pytest.approx(0.6)
    assert inputs[-2] == pytest.approx(0.8)
    assert inputs[-1] == 1.0


def test_zone_bias_requires_zones() -> None:
    """Asking for a zone bias without zones is a configuration error."""
    population = _population(2)

    with pytest.raises(ValueError, match="no mating zone"):
        _controller(population, movement_bias="nearest_zone")

    with pytest.raises(ValueError, match="Unknown movement_bias"):
        _controller(population, movement_bias="towards_the_sun")


def test_nearest_neighbour_heading_points_at_the_neighbour() -> None:
    """The heading should be a unit vector aimed at the closest robot."""
    population = _population(2)
    controller = _controller(population, movement_bias="nearest_neighbor")
    positions = [np.array([1.0, 1.0, 0.1]), np.array([4.0, 5.0, 0.1])]

    heading = controller.directional_inputs(0, positions)

    assert float(np.linalg.norm(heading)) == pytest.approx(1.0)
    assert heading[0] == pytest.approx(0.6)
    assert heading[1] == pytest.approx(0.8)


def test_nearest_neighbour_heading_wraps_around_the_world() -> None:
    """On a torus the shortest way to a neighbour may be through the edge."""
    population = _population(2)
    controller = _controller(
        population,
        movement_bias="nearest_neighbor",
        use_periodic_boundaries=True,
    )
    positions = [np.array([0.5, 5.0, 0.1]), np.array([9.5, 5.0, 0.1])]

    heading = controller.directional_inputs(0, positions)

    # Going left through the wrap, not right across the world.
    assert heading[0] == pytest.approx(-1.0)
    assert heading[1] == pytest.approx(0.0)


def test_assigned_zone_heading_uses_the_bound_zone() -> None:
    """Each robot aims at its own zone, not the nearest one."""
    population = _population(2)
    controller = _controller(
        population,
        movement_bias="assigned_zone",
        mating_zone_centers=[(1.0, 1.0), (9.0, 9.0)],
        assigned_zones={0: 1, 1: 0},
    )
    positions = [np.array([5.0, 5.0, 0.1]), np.array([5.0, 5.0, 0.1])]

    # Robot 0 is bound to the far zone at (9, 9), so it heads up-right.
    first = controller.directional_inputs(0, positions)
    second = controller.directional_inputs(1, positions)

    assert first[0] > 0
    assert first[1] > 0
    assert second[0] < 0
    assert second[1] < 0


def test_none_bias_gives_no_heading() -> None:
    """Without a bias the network gets zeros, forcing a pure gait."""
    population = _population(2)
    controller = _controller(population, movement_bias="none")
    positions = [np.array([1.0, 1.0, 0.1]), np.array([4.0, 5.0, 0.1])]

    assert np.array_equal(
        controller.directional_inputs(0, positions),
        np.zeros(2),
    )


def test_a_lone_robot_has_no_neighbour_to_aim_at() -> None:
    """Nearest-neighbour bias degrades gracefully for a population of one."""
    population = _population(1)
    controller = _controller(population, movement_bias="nearest_neighbor")

    heading = controller.directional_inputs(0, [np.array([5.0, 5.0, 0.1])])

    assert np.array_equal(heading, np.zeros(2))


def test_apply_writes_clipped_controls() -> None:
    """Every actuator gets a command inside the configured bounds."""
    population = _population(2)
    positions = [np.array([2.0, 2.0, 0.1]), np.array([6.0, 6.0, 0.1])]
    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)

    controller = _controller(
        population,
        substrates=build_substrates(population, spawned.num_joints),
        num_joints=spawned.num_joints,
        control_clip_min=-0.4,
        control_clip_max=0.4,
        movement_bias="nearest_neighbor",
    )
    controller.apply(spawned.model, spawned.data, spawned.core_positions())

    assert np.all(spawned.data.ctrl >= -0.4)
    assert np.all(spawned.data.ctrl <= 0.4)
    assert np.isfinite(spawned.data.ctrl).all()


def test_movement_phase_advances_time_and_records_trajectories() -> None:
    """The phase should step physics and sample one path per robot."""
    population = _population(2)
    positions = [np.array([2.0, 2.0, 0.1]), np.array([6.0, 6.0, 0.1])]
    spawned = spawn_population_in_world(population, positions, WORLD_SIZE)
    controller = _controller(
        population,
        substrates=build_substrates(population, spawned.num_joints),
        num_joints=spawned.num_joints,
        movement_bias="nearest_neighbor",
    )

    trajectories = run_mating_movement_phase(
        spawned,
        controller,
        duration=0.5,
        trajectory_samples=5,
    )

    assert spawned.data.time > 0.0
    assert len(trajectories) == 2
    assert all(len(path) >= 2 for path in trajectories)
    assert all(len(point) == 2 for path in trajectories for point in path)


def test_periodic_wrapping_returns_escapees_to_the_world() -> None:
    """A robot pushed past the edge should reappear on the other side.

    The research prototype wrote wrapped coordinates to ``geom_xpos``, which
    MuJoCo recomputes from ``qpos`` on the next step, so wrapping never took
    effect. Wrapping is applied to ``qpos`` here.
    """
    population = _population(1)
    spawned = spawn_population_in_world(
        population,
        [np.array([5.0, 5.0, 0.1])],
        WORLD_SIZE,
    )
    controller = _controller(
        population,
        substrates=build_substrates(population, spawned.num_joints),
        num_joints=spawned.num_joints,
        movement_bias="none",
        use_periodic_boundaries=True,
    )

    # Teleport the robot well outside the world.
    qpos_adr = spawned.free_joint_qpos_adr[0]
    spawned.data.qpos[qpos_adr] = 13.0
    spawned.data.qpos[qpos_adr + 1] = -2.0

    run_mating_movement_phase(
        spawned,
        controller,
        duration=0.05,
        use_periodic_boundaries=True,
        world_size=(10.0, 10.0),
    )

    assert 0.0 <= spawned.data.qpos[qpos_adr] <= 10.0
    assert 0.0 <= spawned.data.qpos[qpos_adr + 1] <= 10.0
