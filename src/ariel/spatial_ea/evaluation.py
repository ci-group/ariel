"""Isolated fitness evaluation for the spatial EA.

Fitness is measured one robot at a time in an empty world, so it reflects pure
locomotion ability and is decoupled from the social dynamics of the shared
world. A single world is compiled once and reused for the whole population,
because compiling dominates the cost of a short evaluation.

Notes
-----
    * Directional fitness places a random target and rewards travelling *toward*
      it, not merely travelling. ``ariel.simulation.tasks.targeted_locomotion``
      is deliberately not reused here: those functions are minimisation
      objectives, the opposite convention to the rest of this package.

"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.spatial_ea.hyperneat import (
    CPPN,
    Coordinate,
    SubstrateNetwork,
    create_substrate_for_gecko,
)
from ariel.spatial_ea.world import build_single_robot_world, set_robot_yaw

# Type Aliases
type SubstrateCoords = tuple[
    list[Coordinate],
    list[Coordinate] | None,
    list[Coordinate],
]

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.config import SpatialEAConfig
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
FREE_JOINT_DOF = 7
SUBSTRATE_WEIGHT_THRESHOLD = 0.2
MIN_DIRECTION_NORM = 0.01
MIN_DISTANCE_FOR_PROGRESS = 1e-6


@dataclass(slots=True)
class SpatialEvaluationResult:
    """Outcome of evaluating one individual.

    Parameters
    ----------
    start_position
        Core position at the start of the evaluation.
    end_position
        Core position at the end of the evaluation.
    total_distance
        Straight-line distance between start and end, in the plane.
    progress_toward_target
        Distance closed toward the target, negative when the robot moved away.
        Zero when directional fitness is disabled.
    fitness
        The score assigned to the individual.
    """

    start_position: FloatArray
    end_position: FloatArray
    total_distance: float
    progress_toward_target: float
    fitness: float


def directional_fitness(
    total_distance: float,
    progress_toward_target: float,
    progress_weight: float,
) -> float:
    """Score movement by how much of it went toward the target.

    Parameters
    ----------
    total_distance
        Straight-line distance travelled.
    progress_toward_target
        Distance closed toward the target.
    progress_weight
        Size of the directional bonus. At ``0.5`` a perfectly aimed robot
        scores fifty percent more than an equally fast aimless one.

    Returns
    -------
        The directional fitness, never negative.
    """
    if total_distance <= MIN_DISTANCE_FOR_PROGRESS:
        return 0.0

    direction_quality = progress_toward_target / total_distance
    return max(
        0.0,
        total_distance * (1.0 + progress_weight * direction_quality),
    )


def _sample_target(
    start_position: FloatArray,
    robot_yaw: float,
    config: SpatialEAConfig,
) -> FloatArray:
    """Place a target at a random bearing and distance from the robot.

    Parameters
    ----------
    start_position
        Where the robot starts.
    robot_yaw
        The robot's initial heading, so the bearing is relative to the robot
        rather than to the world.
    config
        Supplies ``target_distance_min`` and ``target_distance_max``.

    Returns
    -------
        The target position, at the robot's own height.
    """
    target_distance = np.random.uniform(
        config.target_distance_min,
        config.target_distance_max,
    )
    target_angle = robot_yaw + np.random.uniform(-np.pi, np.pi)

    offset = np.array([
        target_distance * np.cos(target_angle),
        target_distance * np.sin(target_angle),
        0.0,
    ])
    return np.asarray(start_position + offset, dtype=np.float64)


def evaluate_population(
    population: list[SpatialIndividual],
    config: SpatialEAConfig,
    *,
    duration: float | None = None,
) -> list[float]:
    """Evaluate individuals in isolation and record their fitness.

    Parameters
    ----------
    population
        Individuals to evaluate. Each has ``fitness``, ``evaluated``,
        ``start_position``, ``end_position``, ``total_distance``,
        ``progress_toward_target`` and, under directional fitness,
        ``target_position`` updated in place.
    config
        Run configuration.
    duration
        Length of each evaluation in simulated seconds, defaulting to
        ``config.simulation_time``.

    Returns
    -------
        The fitness of every individual, in population order.
    """
    if not population:
        return []

    world_size = (
        config.world_size[0],
        config.world_size[1],
        config.world_z,
    )
    _, model, core_geom_id, free_joint_qpos_adr = build_single_robot_world(
        world_size,
        spawn_position=(
            config.world_size[0] / 2.0,
            config.world_size[1] / 2.0,
            config.spawn_z,
        ),
    )
    num_joints = int(model.nu)

    input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
        num_joints=num_joints,
        use_hidden_layer=True,
        hidden_layer_size=num_joints,
    )

    sim_time = duration if duration is not None else config.simulation_time
    sim_steps = max(1, int(sim_time / model.opt.timestep))

    fitness_values: list[float] = []
    for individual in population:
        result = _evaluate_one(
            individual,
            model=model,
            core_geom_id=core_geom_id,
            free_joint_qpos_adr=free_joint_qpos_adr,
            num_joints=num_joints,
            substrate_coords=(input_coords, hidden_coords, output_coords),
            sim_steps=sim_steps,
            config=config,
        )
        fitness_values.append(result.fitness)

    return fitness_values


def _evaluate_one(
    individual: SpatialIndividual,
    *,
    model: mujoco.MjModel,
    core_geom_id: int,
    free_joint_qpos_adr: int,
    num_joints: int,
    substrate_coords: SubstrateCoords,
    sim_steps: int,
    config: SpatialEAConfig,
) -> SpatialEvaluationResult:
    """Run one individual through the isolated world.

    Parameters
    ----------
    individual
        The individual to evaluate, updated in place.
    model
        The shared compiled single-robot model.
    core_geom_id
        Geom id of the robot core.
    free_joint_qpos_adr
        ``qpos`` address of the robot's free joint.
    num_joints
        Number of actuated joints.
    substrate_coords
        Pre-computed ``(input, hidden, output)`` substrate coordinates.
    sim_steps
        Number of physics steps to run.
    config
        Run configuration.

    Returns
    -------
        The evaluation outcome.
    """
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    robot_yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    set_robot_yaw(data, free_joint_qpos_adr, robot_yaw)
    mujoco.mj_forward(model, data)

    start_position = data.geom_xpos[core_geom_id].copy()

    target_position: FloatArray | None = None
    if config.use_directional_fitness:
        target_position = _sample_target(start_position, robot_yaw, config)
        individual.target_position = target_position.copy()

    input_coords, hidden_coords, output_coords = substrate_coords
    substrate = SubstrateNetwork(
        input_coords=input_coords,
        hidden_coords=hidden_coords,
        output_coords=output_coords,
        cppn=CPPN(individual.genotype),
        weight_threshold=SUBSTRATE_WEIGHT_THRESHOLD,
    )

    for _ in range(sim_steps):
        joint_angles = data.qpos[
            FREE_JOINT_DOF : FREE_JOINT_DOF + num_joints
        ].copy()
        cpg_inputs = np.array([
            np.sin(data.time * 1.0),
            np.cos(data.time * 1.0),
            np.sin(data.time * 2.0),
            np.cos(data.time * 2.0),
        ])

        directional_inputs = np.zeros(2, dtype=float)
        if target_position is not None:
            target_vector = (
                target_position[:2] - data.geom_xpos[core_geom_id][:2]
            )
            distance = float(np.linalg.norm(target_vector))
            if distance > MIN_DIRECTION_NORM:
                directional_inputs = target_vector / distance

        sensor_inputs = np.concatenate([
            joint_angles,
            cpg_inputs,
            directional_inputs,
            np.array([1.0]),
        ])

        data.ctrl[:] = np.clip(
            substrate.activate(sensor_inputs),
            config.control_clip_min,
            config.control_clip_max,
        )
        mujoco.mj_step(model, data)

    end_position = data.geom_xpos[core_geom_id].copy()
    total_distance = float(
        np.linalg.norm(end_position[:2] - start_position[:2]),
    )

    progress = 0.0
    if target_position is not None:
        initial_gap = float(
            np.linalg.norm(target_position[:2] - start_position[:2]),
        )
        final_gap = float(
            np.linalg.norm(target_position[:2] - end_position[:2]),
        )
        progress = initial_gap - final_gap
        fitness = directional_fitness(
            total_distance,
            progress,
            config.progress_weight,
        )
    else:
        fitness = total_distance

    individual.start_position = start_position
    individual.end_position = end_position
    individual.total_distance = total_distance
    individual.progress_toward_target = progress
    individual.fitness = fitness
    individual.evaluated = True

    return SpatialEvaluationResult(
        start_position=start_position,
        end_position=end_position,
        total_distance=total_distance,
        progress_toward_target=progress,
        fitness=fitness,
    )


def evaluate_individual(
    individual: SpatialIndividual,
    config: SpatialEAConfig,
    *,
    duration: float | None = None,
) -> SpatialEvaluationResult:
    """Evaluate a single individual in its own world.

    Convenience wrapper around :func:`evaluate_population` for callers holding
    one individual; prefer the population form, which reuses one compiled
    world.

    Parameters
    ----------
    individual
        The individual to evaluate, updated in place.
    config
        Run configuration.
    duration
        Length of the evaluation in simulated seconds.

    Returns
    -------
        The evaluation outcome.
    """
    evaluate_population([individual], config, duration=duration)
    return SpatialEvaluationResult(
        start_position=(
            individual.start_position
            if individual.start_position is not None
            else np.zeros(3)
        ),
        end_position=(
            individual.end_position
            if individual.end_position is not None
            else np.zeros(3)
        ),
        total_distance=individual.total_distance,
        progress_toward_target=individual.progress_toward_target,
        fitness=individual.fitness,
    )
