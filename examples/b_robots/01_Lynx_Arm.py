import pathlib
import sys
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np

sys.path.append(pathlib.Path.cwd())

import pathlib

from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.scenes.table import TableWorld
from ariel.body_phenotypes.Lynx_Arm.lynx.robots.lynx_manipulator.mj_constructor import (
    construct_lynx,
)


# Currently Completed
def random_controller(model, data) -> None:
    # Generate random actions
    # Action range: (-pi/2, pi/2)

    data.ctrl[:] = np.random.uniform((-np.pi) / 2,

                        (np.pi) / 2,
                        size=model.nu,
                            )

    # data.ctrl[:] =  np.ones(shape=(model.nu,))


def run(robot, with_viewer: bool = True) -> None:
    # MuJoCo configuration
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    # world = SimpleFlatWorld()
    world = TableWorld()  # Use the table world instead of the simple flat world

    # Set transparency of all geoms:
    # for i in range(len(robot.spec.geoms)):
    #     robot.spec.geoms[i].rgba[-1] = 1.0

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    # Save the model to XML
    # xml = world.spec.to_xml()
    # with open("scripts/visualize_mj_lynx.xml", "w", encoding="utf-8") as f:
    #     f.write(xml)

    # print(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    mujoco.set_mjcb_control(None)
    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # View
    mujoco.set_mjcb_control(random_controller)
    if with_viewer:
        with mujoco.viewer.launch(model, data) as viewer:
            # Set camera
            viewer.cam.azimuth = -180
            viewer.cam.elevation = -10
            viewer.cam.distance = 2.0
            viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.8])

            # ======= auto screenshot =======

            renderer.update_scene(data, camera=viewer.cam)
            img = renderer.render()

            pathlib.Path(".results/screenshots").mkdir(exist_ok=True, parents=True)
            imageio.imwrite(f".results/screenshots/lynx_view_{int(time.time())}.png", img)

            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)

                viewer.sync()
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


# @hydra.main(config_path="../configs", config_name="sim", version_base="1.3")


def main() -> None:
    robot = construct_lynx(
        # robot_description_dict=cfg.MorphConfig.robot_description_dict,
        )
    run(robot, with_viewer=True)


if __name__ == "__main__":
    main()
