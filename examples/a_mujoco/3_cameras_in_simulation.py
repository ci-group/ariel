"""Example usage for cameras inside mujoco simualtions."""

# Standard library
from pathlib import Path

# Third-party libraries
import cv2
import time
import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld as World
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.utils.tracker import Tracker


# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- TERMINAL OUTPUT SETUP --- #
install(show_locals=False)
console = Console()


SPEED_MULTIPLIER = 5.0
RENDER_SKIP = 50

def main() -> None:
    """Entry point."""
    # Base world
    world = World()

    gecko_core = gecko()
    world.spawn(gecko_core.spec, position=[0, 0, 0])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict)

    ctrl = Controller(controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
                      tracker=tracker)
    ctrl.tracker.setup(world.spec, data)
    
    mujoco.set_mjcb_control(ctrl.set_control)
    mujoco.mj_resetData(model, data)

    # Render a single frame
    found_camera_name = None
    all_cameras = [model.camera(i).name for i in range(model.ncam)]
    
    for cam_name in all_cameras:
        if "mycamera" in cam_name:
            found_camera_name = cam_name
            break
    
    renderer = mujoco.Renderer(model, height=240, width=240)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            mujoco.mj_step(model, data)
            viewer.sync()

            renderer.update_scene(data, camera=found_camera_name)
            pixels = renderer.render()
            
            bgr_pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
            cv2.imshow("Inset Camera View", bgr_pixels)
            cv2.waitKey(1)

            # Sync time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
