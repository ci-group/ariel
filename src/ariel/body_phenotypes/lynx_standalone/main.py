import os
import time
import yaml
import mujoco
import mujoco.viewer

# Local imports
from ariel.body_phenotypes.lynx_standalone.environment.table import TableWorld
from ariel.body_phenotypes.lynx_standalone.robot.constructor import construct_lynx

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIR, "sim.yaml")

def load_config(config_path=CONFIG_PATH):
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    print("Loading configuration...")
    config = load_config()
    robot_desc = config.get("MorphConfig", {}).get("robot_description_dict", None)

    print("Constructing Lynx Arm...")
    robot = construct_lynx(robot_description_dict=robot_desc)

    print("Setting up Table World...")
    world = TableWorld()

    print("Spawning robot onto the table...")
    world.spawn(robot.spec)

    print("Compiling MuJoCo model...")
    model = world.spec.compile()
    data = mujoco.MjData(model)

    print("Launching MuJoCo viewer (Press ESC to exit)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60.0)

if __name__ == "__main__":
    main()