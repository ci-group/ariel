"""TODO(jmdm): description of script.

Notes
-----
    * Do we consider survivors to be of the new generation?
"""

# Standard library
import random
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

# Third-party libraries
import mujoco as mj
import nevergrad as ng
import numpy as np
from mujoco import viewer
from rich.console import Console
from rich.traceback import install

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.ec.a000 import IntegerMutator
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep
from ariel.ec.a005 import Crossover
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)


# Global constants
SEED = 42
DB_HANDLING_MODES = Literal["delete", "halt"]
SPAWN_POS = [-0.8, 0, 0]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# Global functions
install()
console = Console(width=120)
RNG = np.random.default_rng(SEED)
config = EASettings()


# Same folder imports
from plot_function import show_xpos_history


# ------------------------------------------------------------------------ #
# POPULATION OPS
# ------------------------------------------------------------------------ #
def parent_selection(population: Population) -> Population:
    random.shuffle(population)
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Compare fitness values
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_i.tags = {"ps": True}
            ind_j.tags = {"ps": False}
        else:
            ind_i.tags = {"ps": False}
            ind_j.tags = {"ps": True}
    return population


def crossover(population: Population) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents), 2):
        parent_i = parents[idx]
        parent_j = parents[idx]
        genotype_i, genotype_j = Crossover.one_point(
            cast("list[int]", parent_i.genotype),
            cast("list[int]", parent_j.genotype),
        )

        # First child
        child_i = Individual()
        child_i.genotype = genotype_i
        child_i.tags = {"mut": True}
        child_i.requires_eval = True

        # Second child
        child_j = Individual()
        child_j.genotype = genotype_j
        child_j.tags = {"mut": True}
        child_j.requires_eval = True

        population.extend([child_i, child_j])
    return population


def mutation(population: Population) -> Population:
    for ind in population:
        if ind.tags.get("mut", False):
            genes = cast("list[int]", ind.genotype)
            mutated = IntegerMutator.integer_creep(
                individual=genes,
                span=1,
                mutation_probability=0.5,
            )
            ind.genotype = mutated
            ind.requires_eval = True
    return population


def survivor_selection(population: Population) -> Population:
    random.shuffle(population)
    current_pop_size = len(population)
    for idx in range(0, len(population) - 1, 2):
        ind_i = population[idx]
        ind_j = population[idx + 1]

        # Kill worse individual
        if ind_i.fitness > ind_j.fitness and config.is_maximisation:
            ind_j.alive = False
        else:
            ind_i.alive = False

        # Termination condition
        current_pop_size -= 1
        if current_pop_size <= config.target_population_size:
            break
    return population


def learning(population: Population) -> Population:
    for ind in population:
        if ind.requires_eval:
            robot = genotype_to_phenotype(ind)
            brain = learning_robot(robot)
            ind.tags["brain"] = brain
            # exit()
    return population


def evaluate(population: Population) -> Population:
    for ind in population:
        if ind.requires_eval:
            robot = genotype_to_phenotype(ind)
            ind.fitness = evaluate_robot(robot)
    return population


# ------------------------------------------------------------------------ #
# INDIVIDUAL OPS
# ------------------------------------------------------------------------ #
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def learning_robot(
    robot: CoreModule,
) -> dict[str, np.ndarray]:
    """Entry function to run the simulation with random movements."""
    # Config
    num_of_workers = 50
    budget = 500

    # Check inputs
    mj.set_mjcb_control(None)
    model = robot.spec.compile()
    data = mj.MjData(model)
    num_of_inputs = len(data.ctrl)
    del model, data

    # Setup Nevergrad optimizer
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        w=ng.p.Array(shape=(num_of_inputs,)).set_bounds(-2 * np.pi, 2 * np.pi),
        amplitudes=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            -2 * np.pi,
            2 * np.pi,
        ),
        ha=ng.p.Array(shape=(num_of_inputs,)).set_bounds(-10, 10),
        b=ng.p.Array(shape=(num_of_inputs,)).set_bounds(-100, 100),
    )

    optim = ng.optimizers.PSO
    optimizer = optim(
        parametrization=params,
        budget=budget,
        num_workers=num_of_workers,
    )

    # Run optimization loop
    best_fitness = float("inf")
    best_params = None
    for idx in range(optimizer.budget):
        x = optimizer.ask()
        brain = x.kwargs
        loss = evaluate_robot(robot, brain)
        optimizer.tell(x, loss)
        if loss < best_fitness:
            best_fitness = loss
            best_params = x.kwargs
            console.log(
                f"({idx}) Current loss: {loss}, Best loss: {best_fitness}",
            )
    return best_params


def experiment(
    robot: CoreModule,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "simple",
) -> None:
    """Run the simulation with random movements."""
    # ------------------------------------------------------------------ #
    # WORLD OBJECTS
    # ------------------------------------------------------------------ #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld(
        load_precompiled=False,
    )

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # ------------------------------------------------------------------ #
    # CONTROLLER
    # ------------------------------------------------------------------ #
    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    # RENDERING
    # ------------------------------------------------------------------ #
    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Modes
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            cam_quat = np.zeros(4)
            mj.mju_euler2Quat(cam_quat, np.deg2rad([30, 0, 0]), "XYZ")
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
                cam_fovy=7,
                cam_pos=[2, -1, 2],
                cam_quat=cam_quat,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )


def evaluate_robot(
    robot: CoreModule,
    brain: dict[str, np.ndarray] | None = None,
    mode: ViewerTypes = "simple",
) -> float:
    # -------------------------------------------------------------- #
    # TRACKER
    # -------------------------------------------------------------- #
    # Define a tracker to track the x-position of the robot
    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # -------------------------------------------------------------- #
    # CONTROLLER
    # -------------------------------------------------------------- #
    # Setup the NaCPG controller
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    model = robot.spec.compile()
    data = mj.MjData(model)
    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    del model, data
    na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)
    if brain is not None:
        na_cpg_mat.set_param_with_dict(brain)

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
        tracker=tracker,
    )

    # -------------------------------------------------------------- #
    # EXPERIMENT
    # -------------------------------------------------------------- #
    experiment(robot=robot, controller=ctrl, mode=mode)

    # Calculate and print the fitness of your robot
    return fitness_function(tracker.history["xpos"][0])


def genotype_to_phenotype(individual: Individual) -> CoreModule:
    # Genotype
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(individual.genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Construct the robot from the graph
    return construct_mjspec_from_graph(robot_graph)


def create_individual() -> Individual:
    ind = Individual()
    scale = 8192
    genotype_size = 64
    type_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
    conn_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
    rot_p_genes = RNG.uniform(-scale, scale, genotype_size).astype(
        np.float32,
    )
    ind.genotype = [
        type_p_genes.tolist(),
        conn_p_genes.tolist(),
        rot_p_genes.tolist(),
    ]
    return ind


def main() -> None:
    """Entry point."""
    # Create initial population
    pop_size = 10
    n_gens = 100
    population_list = [create_individual() for _ in range(pop_size)]
    population_list = evaluate(population_list)

    # Create EA steps
    ops = [
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutation", mutation),
        EAStep("learning", learning),
        EAStep("evaluation", evaluate),
        EAStep("survivor_selection", survivor_selection),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=n_gens,
    )

    ea.run()

    best = ea.get_solution(only_alive=False)
    console.log(best)

    median = ea.get_solution("median", only_alive=False)
    console.log(median)

    worst = ea.get_solution("worst", only_alive=False)
    console.log(worst)


if __name__ == "__main__":
    main()
