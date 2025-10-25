"""TODO(jmdm): description of script.

Author:     jmdm
Date:       2025-06-25
Py Ver:     3.12
OS:         macOS  Sequoia 15.3.1
Hardware:   M4 Pro
Status:     Completed ✅
"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.l_system_genotype import (
    load_graph_from_json,
)


from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

if TYPE_CHECKING:
    from networkx import DiGraph


# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 40

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def main() -> None:
    """Entry point."""

    graph_parent1 = load_graph_from_json("parent1.json")
    graph_parent2 = load_graph_from_json("parent2.json")
    graph_parent1_mutated = load_graph_from_json("parent1-mutated.json")
    graph_parent2_mutated = load_graph_from_json("parent2-mutated.json")
    graph_offspring1 = load_graph_from_json("offspring1.json")
    graph_offspring2 = load_graph_from_json("offspring2.json")
    graph_offspring3 = load_graph_from_json("offspring3.json")
    graph_offspring4 = load_graph_from_json("offspring4.json")

    # Construct the robot from the graph
    core_parent1 = construct_mjspec_from_graph(graph_parent1)
    core_parent2 = construct_mjspec_from_graph(graph_parent2)
    core_parent1_mutated = construct_mjspec_from_graph(graph_parent1_mutated)
    core_parent2_mutated = construct_mjspec_from_graph(graph_parent2_mutated)
    core_offspring1 = construct_mjspec_from_graph(graph_offspring1)
    core_offspring2 = construct_mjspec_from_graph(graph_offspring2)
    core_offspring3 = construct_mjspec_from_graph(graph_offspring3)
    core_offspring4 = construct_mjspec_from_graph(graph_offspring4)
    # Simulate the robot
    run(core_parent1, with_viewer=True)
    run(core_parent2, with_viewer=True)
    run(core_parent1_mutated, with_viewer=True)
    run(core_parent2_mutated, with_viewer=True)
    run(core_offspring1, with_viewer=True)
    run(core_offspring2, with_viewer=True)
    run(core_offspring3, with_viewer=True)
    run(core_offspring4, with_viewer=True)

def run(
    robot: CoreModule,
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

    # Render
    single_frame_renderer(model, data, steps=10)

    # View
    if with_viewer:
        viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()
