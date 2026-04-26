"""Launching the MuJoCo viewer."""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel.simulation.environments import OlympicArena as World

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


def main() -> None:
    """Entry point."""
    # Base world
    world = World()

    # Spawn a test object to validate the environment
    xml = r"""
    <mujoco>
    <worldbody>
        <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
        <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </worldbody>
    </mujoco>
    """
    test_object = mujoco.MjSpec.from_string(xml)
    world.spawn(test_object, correct_collision_with_floor=True)

    # Compile the model and create data
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Render a single frame
    viewer.launch(model, data)


if __name__ == "__main__":
    main()
