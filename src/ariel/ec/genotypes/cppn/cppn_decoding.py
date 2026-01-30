from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)
SEED = 42
console = Console()
RNG = np.random.default_rng(SEED)
np.set_printoptions(precision=3, suppress=True)


def run(robot: CoreModule, *, with_viewer: bool = False) -> None:
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # xml = world.spec.to_xml()
    # with (DATA / f"{SCRIPT_NAME}.xml").open("w", encoding="utf-8") as f:
    #     f.write(xml)
    # console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    mujoco.mj_resetData(model, data)
    single_frame_renderer(model, data, steps=10)

    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict)

    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time)
    )
    ctrl.tracker.setup(world.spec, data)

    mujoco.set_mjcb_control(ctrl.set_control)
    mujoco.mj_resetData(model, data)

    if with_viewer:
        viewer.launch(model=model, data=data)


if __name__ == "__main__":
    DECODER_TYPE = "best_first"
    MAX_MODULES = 5
    num_initial_mutations = 5

    T, R = NUM_OF_TYPES_OF_MODULES, NUM_OF_ROTATIONS
    NUM_CPPN_INPUTS, NUM_CPPN_OUTPUTS = 6, 1 + T + R

    # 1. Define the starting innovation ID for the first genome as 0.
    initial_innov_id = 0

    # 2. Calculate how many connections will be created.
    num_initial_conns = NUM_CPPN_INPUTS * NUM_CPPN_OUTPUTS

    # 3. Initialize the IdManager. Its last-used ID is (num_conns - 1).
    id_manager = IdManager(
        node_start=NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS - 1,
        innov_start=num_initial_conns
        - 1,  # we have 78 conns (6 inputs * 13 outputs) --> last ID is 77
    )

    # 4. Create the initial random genome.
    my_cppn_genome = Genome.random(
        num_inputs=NUM_CPPN_INPUTS,
        num_outputs=NUM_CPPN_OUTPUTS,
        next_node_id=(NUM_CPPN_INPUTS + NUM_CPPN_OUTPUTS),
        next_innov_id=initial_innov_id,  # This is now 0
    )

    # 5. Apply initial mutations to the genome.
    # (this was done in old revolve2, and it makes sense to keep it but it's optional)
    for i in range(num_initial_mutations):
        # print(f"Applying mutation {i+1}/{num_initial_mutations}...")
        my_cppn_genome.mutate(
            1.0,  # Use floats for rates
            1.0,  # Use floats for rates
            id_manager.get_next_innov_id,
            id_manager.get_next_node_id,
        )
        # print("Number of Nodes: ", len(my_cppn_genome.nodes))
        # print("Number of Connections: ", len(my_cppn_genome.connections))

    # 6. Decode the genome into a robot morphology.
    # console.log(f"Decoding with [bold cyan]{DECODER_TYPE}[/bold cyan]...")

    decoder = MorphologyDecoderBestFirst(
        cppn_genome=my_cppn_genome, max_modules=MAX_MODULES
    )
    decoded_robot_graph = decoder.decode()

    # console.log(f"[bold green]Success! Found a morphology with {decoded_robot_graph.number_of_nodes()} modules.[/bold green]")
    final_robot_graph = decoded_robot_graph

    # 7. Construct and run the final robot.
    core = construct_mjspec_from_graph(final_robot_graph)
    # console.log("[bold green]Robot constructed! Starting simulation...[/bold green]")
    run(core, with_viewer=True)
