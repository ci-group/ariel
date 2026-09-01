"""Shared-world construction for the spatial EA.

The spatial phase puts the whole population into a single MuJoCo world so that
robots can approach one another physically. This module builds that world and
resolves the per-robot handles the controller and the boundary wrapping need.

Notes
-----
    * ``BaseWorld.spawn`` names each attachment ``robot{n}_`` where ``n`` counts
      from one, so robot ``i`` owns the geom ``robot{i + 1}_core`` and the body
      ``robot{i + 1}_world`` that carries its free joint.
    * Free joints are added by ``spawn`` without a name, so they are located
      through their body rather than by name lookup.

"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel import log
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.tracker import Tracker

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import Dimension, FloatArray, Position
    from ariel.spatial_ea.individual import SpatialIndividual

# Global constants
FREE_JOINT_DOF = 7
CORE_GEOM_SUFFIX = "core"


def floor_size_covering(world_size: Dimension) -> Dimension:
    """Size a floor plane so it covers the whole logical world.

    ARIEL builds the floor as a plane centred on the origin, and ``floor_size``
    is its full extent — so ``(4, 4, z)`` renders ``[-2, +2]``. The spatial EA's
    logical world instead spans ``[0, W]``, which would leave robots visibly off
    the checkerboard. Doubling the extent covers ``[-W, +W]``, which contains
    the logical world.

    This is a rendering concern only: a MuJoCo plane is infinite for collision,
    so robots never fall off regardless.

    Parameters
    ----------
    world_size
        Logical world dimensions ``(width, height, thickness)``.

    Returns
    -------
        Floor dimensions that cover the logical world.
    """
    return (world_size[0] * 2.0, world_size[1] * 2.0, world_size[2])


@dataclass
class SpawnedPopulation:
    """Handles into a compiled world holding a whole population.

    Parameters
    ----------
    world
        The world specification the robots were attached to.
    model
        The compiled MuJoCo model.
    data
        Simulation state for ``model``.
    robots
        The robot bodies, in population order.
    core_geom_ids
        Geom id of each robot's core, in population order.
    free_joint_qpos_adr
        ``qpos`` address of each robot's free joint, in population order.
    num_joints
        Number of actuated joints per robot.
    tracker
        ARIEL tracker bound to every robot core, recording ``xpos`` history.
        Bound in spawn order, which is population order.
    """

    world: SimpleFlatWorld
    model: mujoco.MjModel
    data: mujoco.MjData
    robots: list[Any] = field(default_factory=list)
    core_geom_ids: list[int] = field(default_factory=list)
    free_joint_qpos_adr: list[int] = field(default_factory=list)
    num_joints: int = 0
    tracker: Tracker | None = None

    def record(self) -> None:
        """Append the current core positions to the tracker history."""
        if self.tracker is not None:
            self.tracker.update(self.data)

    def tracked_trajectories(self) -> list[list[FloatArray]]:
        """Return the recorded path of every robot.

        Returns
        -------
            One list of ``(x, y)`` samples per robot, in population order.
            Empty when nothing was recorded.
        """
        if self.tracker is None:
            return []

        history = self.tracker.history.get("xpos", {})
        return [
            [position[:2].copy() for position in history[idx]]
            for idx in sorted(history)
        ]

    def core_positions(self) -> list[FloatArray]:
        """Read the current world position of every robot core.

        Returns
        -------
            One ``(3,)`` array per robot, in population order.
        """
        return [
            self.data.geom_xpos[geom_id].copy()
            for geom_id in self.core_geom_ids
        ]


def generate_spawn_positions(
    population_size: int,
    spawn_x_range: tuple[float, float],
    spawn_y_range: tuple[float, float],
    spawn_z: float,
    min_spawn_distance: float,
    max_attempts: int = 1000,
) -> list[FloatArray]:
    """Sample non-overlapping spawn positions.

    Parameters
    ----------
    population_size
        Number of positions to generate.
    spawn_x_range
        Inclusive ``(min, max)`` bounds for the x coordinate.
    spawn_y_range
        Inclusive ``(min, max)`` bounds for the y coordinate.
    spawn_z
        Height shared by every spawn position.
    min_spawn_distance
        Minimum planar separation between any two positions.
    max_attempts
        Rejection-sampling attempts before falling back to a grid slot.

    Returns
    -------
        One ``(3,)`` position array per individual.
    """
    positions: list[FloatArray] = []

    for i in range(population_size):
        placed = False
        for _ in range(max_attempts):
            candidate = np.array([
                np.random.uniform(*spawn_x_range),
                np.random.uniform(*spawn_y_range),
                spawn_z,
            ])
            if all(
                float(np.linalg.norm(candidate[:2] - existing[:2]))
                >= min_spawn_distance
                for existing in positions
            ):
                positions.append(candidate)
                placed = True
                break

        if not placed:
            msg = (
                f"No non-overlapping spawn position found for robot {i}, "
                f"falling back to grid placement"
            )
            log.warning(msg)
            grid_size = int(np.ceil(np.sqrt(population_size)))
            positions.append(
                np.array([
                    (i % grid_size) * min_spawn_distance + spawn_x_range[0],
                    (i // grid_size) * min_spawn_distance + spawn_y_range[0],
                    spawn_z,
                ]),
            )

    return positions


def _resolve_robot_handles(
    model: mujoco.MjModel,
    population_size: int,
) -> tuple[list[int], list[int]]:
    """Resolve the core geom and free-joint address of every spawned robot.

    Parameters
    ----------
    model
        A compiled model containing ``population_size`` spawned robots.
    population_size
        Number of robots that were spawned.

    Returns
    -------
    core_geom_ids
        Geom id of each robot's core.
    free_joint_qpos_adr
        ``qpos`` address of each robot's free joint.

    Raises
    ------
    RuntimeError
        If a robot's core geom cannot be found in the compiled model.
    """
    core_geom_ids: list[int] = []
    free_joint_qpos_adr: list[int] = []

    for i in range(population_size):
        prefix = f"robot{i + 1}"

        geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{prefix}_core",
        )
        if geom_id < 0:
            msg = f"Could not resolve geom '{prefix}_core' in compiled model"
            raise RuntimeError(msg)
        core_geom_ids.append(geom_id)

        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"{prefix}_world",
        )
        qpos_adr = -1
        if body_id >= 0:
            for joint_id in range(model.njnt):
                is_free = model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
                if is_free and model.jnt_bodyid[joint_id] == body_id:
                    qpos_adr = int(model.jnt_qposadr[joint_id])
                    break
        free_joint_qpos_adr.append(qpos_adr)

    return core_geom_ids, free_joint_qpos_adr


def set_robot_yaw(
    data: mujoco.MjData,
    qpos_adr: int,
    yaw: float,
) -> None:
    """Set a robot's heading by writing its free-joint quaternion.

    Parameters
    ----------
    data
        Simulation state to modify.
    qpos_adr
        ``qpos`` address of the robot's free joint. Negative values are
        ignored, which covers robots without a free joint.
    yaw
        Heading in radians, measured about the world z axis.
    """
    if qpos_adr < 0:
        return

    data.qpos[qpos_adr + 3] = np.cos(yaw / 2.0)
    data.qpos[qpos_adr + 4] = 0.0
    data.qpos[qpos_adr + 5] = 0.0
    data.qpos[qpos_adr + 6] = np.sin(yaw / 2.0)


def spawn_population_in_world(
    population: list[SpatialIndividual],
    positions: list[FloatArray],
    world_size: Dimension,
    orientations: list[float] | None = None,
    *,
    correct_collision_with_floor: bool = False,
) -> SpawnedPopulation:
    """Spawn a whole population into one shared world.

    Parameters
    ----------
    population
        Individuals to spawn. Each has its ``spawn_position`` and
        ``robot_index`` updated in place.
    positions
        Spawn position per individual.
    world_size
        Logical world dimensions ``(x, y, z)``. The rendered floor is sized to
        cover it; see :func:`floor_size_covering`.
    orientations
        Initial yaw per individual in radians. Defaults to zero for all.
    correct_collision_with_floor
        Whether to let ARIEL lift each robot clear of the floor. This compiles
        a throwaway model per spawn, so it is off by default and the caller's
        spawn height is used as-is.

    Returns
    -------
        Handles into the compiled shared world.
    """
    world = SimpleFlatWorld(floor_size=floor_size_covering(world_size))
    robots: list[Any] = []

    for i, individual in enumerate(population):
        robot = gecko()
        robots.append(robot)
        position = positions[i]
        individual.spawn_position = np.asarray(position, dtype=float).copy()
        individual.robot_index = i
        world.spawn(
            robot.spec,
            position=(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            ),
            correct_collision_with_floor=correct_collision_with_floor,
        )

    model = world.spec.compile()
    data = mujoco.MjData(model)

    core_geom_ids, free_joint_qpos_adr = _resolve_robot_handles(
        model,
        len(population),
    )

    if orientations is not None:
        for qpos_adr, yaw in zip(
            free_joint_qpos_adr,
            orientations,
            strict=False,
        ):
            set_robot_yaw(data, qpos_adr, yaw)

    mujoco.mj_forward(model, data)

    num_joints = model.nu // len(population) if population else 0

    # ARIEL's tracker binds every geom whose name contains "core", in spec
    # order, which is the order the robots were spawned in.
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind=CORE_GEOM_SUFFIX,
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)

    return SpawnedPopulation(
        world=world,
        model=model,
        data=data,
        robots=robots,
        core_geom_ids=core_geom_ids,
        free_joint_qpos_adr=free_joint_qpos_adr,
        num_joints=num_joints,
        tracker=tracker,
    )


def build_single_robot_world(
    world_size: Dimension,
    spawn_position: Position = (0.0, 0.0, 0.1),
) -> tuple[SimpleFlatWorld, mujoco.MjModel, int, int]:
    """Build a one-robot world for isolated fitness evaluation.

    Compiling a world is expensive, so evaluation reuses a single compiled
    model across the whole population rather than rebuilding it per individual.

    Parameters
    ----------
    world_size
        Logical world dimensions ``(x, y, z)``.
    spawn_position
        Where the robot is placed.

    Returns
    -------
    world
        The world specification.
    model
        The compiled model.
    core_geom_id
        Geom id of the robot core.
    free_joint_qpos_adr
        ``qpos`` address of the robot's free joint, or ``-1`` if absent.
    """
    world = SimpleFlatWorld(floor_size=floor_size_covering(world_size))
    robot = gecko()
    world.spawn(
        robot.spec,
        position=spawn_position,
        correct_collision_with_floor=False,
    )
    model = world.spec.compile()
    core_geom_ids, free_joint_qpos_adr = _resolve_robot_handles(model, 1)
    return world, model, core_geom_ids[0], free_joint_qpos_adr[0]
