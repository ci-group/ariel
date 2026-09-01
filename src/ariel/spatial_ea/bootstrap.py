"""Bootstrap helpers for the spatial EA.

Compiling a throwaway world is the cheapest way to learn how many actuated
joints a robot has, which the substrate layout needs before any individual
exists.
"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

# Third-party libraries
# Local libraries
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld

if TYPE_CHECKING:
    import mujoco


@dataclass(slots=True)
class SpatialBootstrap:
    """A compiled world holding one robot.

    Parameters
    ----------
    world
        The world specification.
    robot
        The spawned robot body.
    model
        The compiled MuJoCo model.
    num_joints
        Number of actuated joints on the robot.
    """

    world: SimpleFlatWorld
    robot: Any
    model: mujoco.MjModel
    num_joints: int


def build_default_bootstrap(
    world_size: tuple[float, float, float] = (10.0, 10.0, 0.1),
    spawn_position: tuple[float, float, float] = (5.0, 5.0, 0.1),
) -> SpatialBootstrap:
    """Compile the default one-robot world.

    Parameters
    ----------
    world_size
        Floor size of the world, ``(x, y, z)``.
    spawn_position
        Where the robot is placed.

    Returns
    -------
        The compiled world and its joint count.
    """
    world = SimpleFlatWorld(floor_size=world_size)
    robot = gecko()
    world.spawn(
        robot.spec,
        position=spawn_position,
        correct_collision_with_floor=True,
    )
    model = world.spec.compile()
    return SpatialBootstrap(
        world=world,
        robot=robot,
        model=model,
        num_joints=int(model.nu),
    )
