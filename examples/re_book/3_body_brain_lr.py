"""TODO(jmdm): description of script.

Notes
-----
    * Do we consider survivors to be of the new generation?
"""

# Standard library
import datetime
import random
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

# Third-party libraries
import mujoco as mj
import nevergrad as ng
import numpy as np
import torch
from mujoco import viewer
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.traceback import install

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.cppn_neat.genome import Genome
from ariel.body_phenotypes.robogen_lite.cppn_neat.id_manager import IdManager
from ariel.body_phenotypes.robogen_lite.decoders.cppn_best_first import (
    MorphologyDecoderBestFirst,
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
from ariel.simulation.controllers.simple_cpg import SimpleCPG
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
DATA = CWD / "__data__" / SCRIPT_NAME
shutil.rmtree(DATA)
DATA.mkdir(exist_ok=True)
VIDEOS = DATA / "videos"
VIDEOS.mkdir(exist_ok=True)

# Global constants
SEED = 42
DB_HANDLING_MODES = Literal["delete", "halt"]

# Global structures
install()
console = Console(width=120)
RNG = np.random.default_rng(SEED)
config = EASettings()
ID_MANAGER = IdManager(node_start=6 + 13 - 1, innov_start=(6 * 13) - 1)

# --- Overal Params ---
# Robot
NUM_OF_MODULES = 10
SPAWN_POS = (-0.8, 0.0, 0.1)
TARGET_POSITION = np.array([2.0, 0.0, 0.5])

# EA
POPULATION_SIZE = 80
GENERATIONS = 80
SIMULATION_DURATION = 30
LEARNING_BUDGET = 80
LEARNING_ALGORITHM = ng.optimizers.PSO


# ------------------------------------------------------------------------ #
# POPULATION OPS
# ------------------------------------------------------------------------ #
def parent_selection(population: list[Individual]) -> list[Individual]:
    """
    Perform truncation parent selection by tagging the top 50% of individuals.

    Parameters
    ----------
    population : list[Individual]
        The current population to be tagged.

    Returns
    -------
    list[Individual]
        The population with updated 'ps' tags.
    """
    # Sort descending: higher fitness (closer to 0) is better
    population.sort(key=lambda x: x.fitness, reverse=True)

    cutoff = len(population) // 2
    for i, ind in enumerate(population):
        ind.tags["ps"] = i < cutoff

    return population


def crossover(population: Population) -> Population:
    """
    Perform joint crossover on morphology (CPPN) and control (vector).

    Parameters
    ----------
    population : Population
        The current population containing selected parents.

    Returns
    -------
    Population
        The population expanded with new offspring.
    """
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    if len(parents) < 2:
        return population

    for idx in range(0, len(parents) - 1, 2):
        p1, p2 = parents[idx], parents[idx + 1]

        # 1. Morphology Crossover (CPPN)
        m1 = Genome.from_dict(p1.genotype["morph"])
        m2 = Genome.from_dict(p2.genotype["morph"])
        child_morph = m1.copy()
        # Randomly replace some connections with parent 2's connections
        try:
            for conn in child_morph.connections:
                if RNG.random() < 0.5 and len(m2.connections) > 0:
                    p2_conn = random.choice(m2.connections)
                    conn.weight = p2_conn.weight
        except:
            # If crossover fails, just use child as-is
            pass

        # 2. Control Crossover (Uniform)
        c1, c2 = np.array(p1.genotype["ctrl"]), np.array(p2.genotype["ctrl"])
        mask = RNG.random(len(c1)) < 0.5
        child_ctrl = np.where(mask, c1, c2).tolist()

        # Instantiate offspring
        child = Individual()
        child.genotype = {"morph": child_morph.to_dict(), "ctrl": child_ctrl}
        child.tags = {"mut": True, "requires_lr": True}
        child.requires_eval = True
        population.append(child)

    return population


def mutation(population: Population) -> Population:
    """
    Apply morphology mutations and Gaussian creep to control vectors.

    Parameters
    ----------
    population : Population
        The current population containing potential mutants.

    Returns
    -------
    Population
        The population with mutated individuals.
    """
    for ind in population:
        if not ind.tags.get("mut", False):
            continue

        # 1. Morphology Mutation (Structural)
        morph = Genome.from_dict(ind.genotype["morph"])
        morph.mutate(
            0.8, 0.5, ID_MANAGER.get_next_innov_id, ID_MANAGER.get_next_node_id
        )
        ind.genotype["morph"] = morph.to_dict()

        # 2. Control Mutation (Gaussian Creep)
        ctrl = np.array(ind.genotype["ctrl"])
        mask = RNG.random(ctrl.shape) < 0.4
        noise = RNG.normal(0, 0.6, ctrl.shape)
        ctrl[mask] += noise[mask]
        ind.genotype["ctrl"] = np.clip(ctrl, -1.0, 1.0).tolist()

        ind.requires_eval = True

    return population


def survivor_selection(population: list[Individual]) -> list[Individual]:
    """
    Perform truncation survivor selection to maintain a fixed population size.

    Parameters
    ----------
    population : list[Individual]
        The current population including parents and offspring.

    Returns
    -------
    list[Individual]
        The population with 'alive' status updated.
    """
    # target_population_size from main loop
    target_size = 10

    population.sort(key=lambda x: x.fitness, reverse=True)

    for i, ind in enumerate(population):
        ind.alive = i < target_size

    return population


def learning(population: Population) -> Population:
    for ind in population:
        if ind.tags["requires_lr"]:
            robot = genotype_to_phenotype(ind)
            brain = learning_robot(robot)
            ind.tags["brain"] = brain
            ind.tags["requires_lr"] = False

    # Move videos folder to give it a timestamp
    timestamp = int(datetime.datetime.now(datetime.UTC).timestamp())
    shutil.move(VIDEOS, DATA / f"{timestamp}")

    # Ensure the empty folder exists
    VIDEOS.mkdir()
    return population


def evaluate(
    population: Population, mode: ViewerTypes = "simple"
) -> Population:
    for ind in population:
        if ind.requires_eval:
            robot = genotype_to_phenotype(ind)
            ind.fitness = evaluate_robot(robot, mode=mode)
    return population


# ------------------------------------------------------------------------ #
# INDIVIDUAL OPS
# ------------------------------------------------------------------------ #
def fitness_function(
    history: list[tuple[float, float, float]],
    duration: float,
) -> float:
    """
    Calculate fitness based on displacement after a 1-second delay.

    Parameters
    ----------
    history : list[tuple[float, float, float]]
        The trajectory of the robot's core.
    duration : float
        Total simulation time in seconds.

    Returns
    -------
    float
        The negative 2D distance to target (for maximization).
    """
    if not history:
        return -999.0

    # 1. Calculate delay indices (1 second offset)
    delay_time = min(1.0, duration)
    delay_fraction = delay_time / duration if duration > 0 else 0
    start_idx = max(0, int(len(history) * delay_fraction))

    # 2. Calculate valid movement vector (Final Pos - Pos after 1s)
    pos_after_delay = np.array(history[start_idx])
    pos_final = np.array(history[-1])
    valid_movement_vector = pos_final - pos_after_delay

    # 3. Project movement from the starting spawn point
    effective_pos = np.array(SPAWN_POS) + valid_movement_vector

    # 4. Calculate 2D Euclidean distance to target
    dist = np.sqrt(
        np.sum((effective_pos[:2] - np.array(TARGET_POSITION[:2])) ** 2)
    )

    # Return negative for maximization compatibility in file 3
    return -float(dist)


def learning_robot(
    robot: CoreModule,
) -> dict[str, np.ndarray] | None:
    """Entry function to run the simulation with random movements."""
    # Config
    budget = LEARNING_BUDGET
    num_of_workers = 1

    # Check inputs
    mj.set_mjcb_control(None)
    model = robot.spec.compile()
    data = mj.MjData(model)
    num_of_inputs = len(data.ctrl)
    del model, data

    # No hinges == no learning
    if num_of_inputs == 0:
        return None

    # Setup Nevergrad optimizer
    params = ng.p.Instrumentation(
        phase=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            (-2 * np.pi) - 1,
            (2 * np.pi) + 1,
        ),
        w=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            (-2 * np.pi) - 1,
            (2 * np.pi) + 1,
        ),
        amplitudes=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            (-2 * np.pi) - 1,
            (2 * np.pi) + 1,
        ),
        ha=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            -3,
            3,
        ),
        b=ng.p.Array(shape=(num_of_inputs,)).set_bounds(
            -2,
            2,
        ),
    )
    optimizer = LEARNING_ALGORITHM(
        parametrization=params,
        budget=budget,
        num_workers=num_of_workers,
    )

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[loss_info]}"),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Learning:", total=budget, loss_info="")

        best_fitness = float("inf")
        best_params = None

        for _ in range(budget):
            x = optimizer.ask()
            brain = x.kwargs
            loss = evaluate_robot(robot, brain)
            optimizer.tell(x, loss)

            if loss < best_fitness:
                best_fitness = loss
                best_params = x.kwargs

            # Update counter and custom loss text
            progress.update(
                task_id,
                advance=1,
                loss_info=f"| Current: {loss:.4f} | Best: {best_fitness:.4f}",
            )

    # View best
    x = optimizer.provide_recommendation()
    brain = x.kwargs
    loss = evaluate_robot(robot, brain, mode="video")
    return serialise_brain_to_json(best_params)


def experiment(
    robot: CoreModule,
    controller: Controller,
    duration: float,
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
            path_to_video_folder = str(VIDEOS)
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
    duration: float = SIMULATION_DURATION,
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
    # Load robot model
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    model = robot.spec.compile()
    data = mj.MjData(model)
    num_of_joints = model.nu
    del model, data

    # SimpleCPG setup
    adj_dict = create_fully_connected_adjacency(num_of_joints)
    cpg = SimpleCPG(adj_dict)

    if brain is not None:
        # JSON object to numpy arrays
        brain_np = deserialise_brain_to_numpy(brain)

        # Map flat parameters to SimpleCPG tensors
        with torch.no_grad():
            for key, val in brain_np.items():
                getattr(cpg, key).data.copy_(torch.from_numpy(val).float())

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: cpg.forward(d.time),
        tracker=tracker,
    )

    # -------------------------------------------------------------- #
    # EXPERIMENT
    # -------------------------------------------------------------- #
    experiment(
        robot=robot,
        controller=ctrl,
        mode=mode,
        duration=duration,
    )

    # Calculate and print the fitness of your robot
    return fitness_function(
        tracker.history["xpos"][0],
        duration=duration,
    )


def genotype_to_phenotype(individual: Individual) -> CoreModule:
    """Decode CPPN genome into a MuJoCo robot spec."""
    genome = Genome.from_dict(individual.genotype["morph"])
    decoder = MorphologyDecoderBestFirst(
        cppn_genome=genome,
        max_modules=NUM_OF_MODULES,
    )
    robot_graph = decoder.decode()
    return construct_mjspec_from_graph(robot_graph)


def create_individual() -> Individual:
    """Initialize individual with CPPN morphology and flat controller vector."""
    genome = Genome.random(
        num_inputs=6,
        num_outputs=13,
        next_node_id=ID_MANAGER.get_next_node_id(),
        next_innov_id=ID_MANAGER.get_next_innov_id(),
    )
    ind = Individual()
    ind.genotype = {
        "morph": genome.to_dict(),
        "ctrl": RNG.uniform(-1.0, 1.0, size=150).tolist(),  # Default size
    }
    ind.tags["requires_lr"] = True
    return ind


# ------------------------------------------------------------------------ #
# FUNCTIONS
# ------------------------------------------------------------------------ #
def serialise_brain_to_json(brain: dict[str, np.ndarray]) -> dict[str, list]:
    brain_json = {}
    for key, value in brain.items():
        brain_json[key] = value.tolist()
    return brain_json


def deserialise_brain_to_numpy(brain: dict[str, list]) -> dict[str, np.ndarray]:
    brain_np = {}
    for key, value in brain.items():
        brain_np[key] = np.array(value)
    return brain_np


# ------------------------------------------------------------------------ #
# OVERARCHING LOOP
# ------------------------------------------------------------------------ #
def main() -> None:
    """Entry point."""
    # Create initial population
    population_list = [create_individual() for _ in range(POPULATION_SIZE)]
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
        num_of_generations=GENERATIONS,
    )

    ea.run()

    best = ea.get_solution(only_alive=False)
    console.log(f"{best.fitness=}")

    median = ea.get_solution("median", only_alive=False)
    console.log(f"{median.fitness=}")

    worst = ea.get_solution("worst", only_alive=False)
    console.log(f"{worst.fitness=}")


if __name__ == "__main__":
    main()
