"""Robogen robot using MjSpec.

Author:     jmdm
Date:       2025-07-06
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅

References
----------
    [1] https://github.com/google-deepmind/dm_control/blob/main/dm_control/mjcf/README.md
    [2] https://mujoco.readthedocs.io/en/stable/python.html#attachment
    [3] https://quaternion.readthedocs.io/en/latest/quaternion/

"""

# Standard library
import contextlib
from pathlib import Path

# Third-party libraries
import mujoco
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsTheta,
)
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)

# Global functions
console = Console(width=180)


class DummyRobotTestAttach:
    """Robot model using MuJoCo MjSpec with attachment points."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        core = CoreModule(index := 0)

        for side_i in ModuleFaces:
            if side_i in core.sites:
                module_1 = HingeModule(index := index + 1)
                core.sites[side_i].attach_body(
                    body=module_1.body,
                    prefix=f"{module_1.index}-{side_i.value}-",
                )

                module_2 = BrickModule(index := index + 1)
                module_1.sites[ModuleFaces.FRONT].attach_body(
                    body=module_2.body,
                    prefix=f"{module_2.index}-{side_i.value}-",
                )
                for side_j in ModuleFaces:
                    if side_j in module_2.sites:
                        module_3 = HingeModule(index := index + 1)
                        module_2.sites[side_j].attach_body(
                            body=module_3.body,
                            prefix=f"{module_3.index}-{side_j.value}-",
                        )

        # Save model specification
        self.spec: mujoco.MjSpec = core.spec


class DummyRobotTestRotate:
    """Robot model using MuJoCo MjSpec with rotation."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        bricks = []
        hinges = []

        index = 0
        for rot_i in ModuleRotationsTheta:
            console.log(index, rot_i.value)
            bricks.append(BrickModule(index := index + 1))
            new_hinge = HingeModule(index := index + 1)
            new_hinge.rotate(rot_i.value)
            hinges.append(new_hinge)

        console.log(len(bricks), len(hinges))

        bricks[0].sites[ModuleFaces.RIGHT].attach_body(
            body=bricks[1].body,
            prefix=f"{bricks[1].index}-front-",
        )
        bricks[0].sites[ModuleFaces.FRONT].attach_body(
            body=hinges[0].body,
            prefix=f"{hinges[0].index}-front-",
        )
        for i in range(1, len(bricks)):
            console.log(i)
            bricks[i].sites[ModuleFaces.LEFT].attach_body(
                body=hinges[i].body,
                prefix=f"{hinges[i].index}-front-",
            )
            with contextlib.suppress(IndexError):
                bricks[i].sites[ModuleFaces.FRONT].attach_body(
                    body=bricks[i + 1].body,
                    prefix=f"{bricks[i + 1].index}-front-",
                )

        # Save model specification
        self.spec: mujoco.MjSpec = bricks[0].spec


class DummyRobotTestCtrl:
    """Robot model using MuJoCo MjSpec with rotation."""

    def __init__(self) -> None:
        """Initialize the robot model."""
        core = CoreModule(index := 0)

        core.sites[ModuleFaces.BOTTOM].attach_body(
            body=BrickModule(index := index + 1).body,
            prefix="brick-bottom-",
        )

        for side_i in (ModuleFaces.FRONT, ModuleFaces.BACK):
            if side_i in core.sites:
                module_1 = HingeModule(index := index + 1)
                core.sites[side_i].attach_body(
                    body=module_1.body,
                    prefix=f"{module_1.index}-{side_i.value}-",
                )

                module_2 = BrickModule(index := index + 1)
                module_1.sites[ModuleFaces.FRONT].attach_body(
                    body=module_2.body,
                    prefix=f"{module_2.index}-{side_i.value}-",
                )
                for side_j in ModuleFaces:
                    if side_j in module_2.sites:
                        module_3 = HingeModule(index := index + 1)
                        module_2.sites[side_j].attach_body(
                            body=module_3.body,
                            prefix=f"{module_3.index}-{side_j.value}-",
                        )
        # Save model specification
        self.spec: mujoco.MjSpec = core.spec


def run(
    robot: DummyRobotTestAttach | DummyRobotTestCtrl | DummyRobotTestRotate,
    *,
    with_viewer: bool = False,
) -> None:
    """Entry point."""
    # MuJoCo configuration
    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    world = SimpleFlatWorld()

    # Set random colors for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Save the model to XML
    xml = world.spec.to_xml()
    with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
        f.write(xml)

    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # View
    if with_viewer:
        viewer.launch(model=model, data=data)
    else:
        # Render
        single_frame_renderer(model, data, steps=10)


def main() -> None:
    """Entry point."""
    robot_1 = DummyRobotTestCtrl()
    run(robot_1, with_viewer=True)

    robot_2 = DummyRobotTestRotate()
    run(robot_2, with_viewer=True)

    robot_3 = DummyRobotTestAttach()
    run(robot_3, with_viewer=True)


if __name__ == "__main__":
    main()
