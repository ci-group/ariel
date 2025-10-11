"""Assignment 3 template code."""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Tuple
import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
import os
from deap import base, tools
import random
from functools import partial
from time import time
import pickle
from datetime import datetime
from dataclasses import field, dataclass, replace
from typing import List, Callable
from scoop import futures

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner, continue_simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from scripts_A3.creator_toolbox_prep import ensure_deap_types, register_factories
from scripts_A3.eval_tools import compute_brain_genome_size, nn_controller, decode_brain_genotype, decode_body_genotype, find_in_out_size
from scripts_A3.statistics_output import print_statistics
# from scripts_A3.file_management import save_generation, load_population_from_generation, load_generation_data, get_next_run_id

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph
# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

#######################################################################################################

### === Dataclass definitions for configuration files ===
@dataclass
class EAConfig:
    """
    General configuration dataclass, containing all global variables needed
    """
    # Seed
    rng_seed: int = 42
    # Simulation durations for checkpoints
    duration_flat: int = 15
    duration_rugged: int = 30
    duration_elevated: int = 55
    # Checkpoints
    checkpoint_rugged: list[float] = field(default_factory=lambda: [0.6, 0, 0.1])
    checkpoint_elevated: list[float] = field(default_factory=lambda: [2.4, 0, 0.1])
    # Possible starting positions
    start_normal: list[float] = field(default_factory=lambda: [-0.8, 0, 0.1])
    start_rugged: list[float] = field(default_factory=lambda: [1.6, 0, 0.1])
    # Simulation positions
    spawn_position: list[float] = field(init=False)
    target_position: list[float] = field(default_factory=lambda: [5, 0, 0.5])
    # Robot parameters
    num_of_modules: int = 30
    data: Path = field(init=False)

    
    def __post_init__(self):
        self.spawn_position = self.start_normal.copy()
        # Data setup
        script_name = __file__.split("/")[-1][:-3]
        cwd = Path.cwd()
        self.data = cwd / "__data__" / script_name
        self.data.mkdir(parents = True, exist_ok=True)
        
    def create_reproducible_rng(self, context_seed: int = 0) -> np.random.Generator:
        """Create reproducible RNG with context-specific seed"""
        # Combine base seed with context for reproducible but varied sequences
        combined_seed = (self.rng_seed + context_seed) % (2**32)
        return np.random.default_rng(combined_seed)

@dataclass
class EABrainConfig:
    """Configuration for Brain Evolutionary Algorithm"""
    # KEEP RUN TO 1!!
    runs_brain:int=                     1
    # General EA parameters
    ngen_brain:int=                     50
    pop_size_brain:int=                 100
    cxpb_brain:float=                   0.5
    mutpb_brain:float=                  0.5
    elites_brain:int=                   1
    # Network structure
    hidden_size:int=                    128
    no_hidden_layers:int=               3
    # Initialization function for brain genotype
    init_func: Callable[[], float]=     partial(np.random.uniform, -1, 1)
    # Mutation parameters
    gauss_mut_mu:float=                 0.0
    gauss_mut_sigma:float=              0.1
    gauss_mut_indpb:float=              0.3
    # Crossover parameters
    wa_alpha:float=                     0.4
    # Selection parameters
    tourn_size:int=                     3
    
@dataclass
class EABodyConfig:
    """Configuration for Body Evolutionary Algorithm"""
    # General EA parameters
    runs_body: int = 1
    ngen_body: int = 2
    pop_size_body: int = 10
    cxpb_body: float = 0.5
    mutpb_body: float = 0.5
    elites_body: int = 1
    # Mutation parameters 
    gauss_mut_mu:float=                 0.0
    gauss_mut_sigma:float=              0.1
    gauss_mut_indpb:float=              0.3
    # Crossover parameters
    wa_alpha:float=                     0.4
    # Selection parameters
    tourn_size:int=                     3
    

### === Saving and loading generations, and related functions ===
def save_generation(
    generation: int, 
    pop_body_genotype, 
    best_body, 
    best_brain, 
    run_id = 0, 
    sim_config = None):
    """
    Save generation data including population and best performers, to the specified directory.
    The data is saved as follows:
    - A population as a list of individuals called body_population.pkl
    - A dictionary object recording important information on that generation 
      called best_performers.pkl, including the following entries:
        - "generation" (int): The generation saved
        - "timestamp" (str): The date and time when this was saved
        - "best_body_genotype" (DEAP list): The body genotype of the best individual
        - "best_brain_genotype" (DEAP list): The brain genotype of the best individual
        - "body_fitness" (float): The fitness of the best individual 
        - "nde" (instance): The nde used to decode the body genotype of the best individual
    
    Args:
        generation (int):           The generation number you want it saved under. 
                                    F.e. if generation=4, and run_id=0, then the data 
                                    will be saved under __data__/A3_modified/run_0/generation_004/

        run_id (int):               The run ID you want it saved under

        pop_body_genotype (list):   List of body genotypes in the population to save
        
        best_body:                  The best body genotype of the generation

        best_brain:                 The best brain genotype of the generation

        sim_config:                 The simulation configuration object, use the one provided in the higher level
                                    this function is located in.
        
    Returns:
        None
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)
    
    with open(gen_dir / "body_population.pkl", "wb") as f:
        pickle.dump(pop_body_genotype, f)
    
    best_data = {
        "generation": generation,
        "timestamp": timestamp,
        "best_body_genotype": list(best_body),  # Convert to list for serialization
        "best_brain_genotype": list(best_brain) if best_brain is not None else None,
        "body_fitness": best_body.fitness.values[0] if best_body.fitness.valid else None,
    }
    
    with open(gen_dir / "best_performers.pkl", "wb") as f:
        pickle.dump(best_data, f)
    
    print(f"Saved generation {generation} data to {gen_dir}")

def load_population_from_generation(sim_config, generation, run_id = 0):
    """
    Load saved generation data, by specifying which run and generation to load.
    
    Args:
        generation (int):           The generation number to load
        run_id (int):               The run number to load
        sim_config (instance):      The simulation configuration object
        
    Returns:
        population (list[DEAP lists]): The population loaded from the generation
        best_data (dict):              The relevant data for the best formining individual
                                       of that generation.
    """
    gen_dir = sim_config.data / f"run_{run_id}" / f"generation_{generation:03d}"
    if not gen_dir.exists():
        raise FileNotFoundError(f"Generation {generation} data not found at {gen_dir}")
    # Load population
    with open(gen_dir / "body_population.pkl", "rb") as f:
        population = pickle.load(f)
    # Load best data
    with open(gen_dir / "best_performers.pkl", "rb") as f:
        best_data = pickle.load(f)
    # Print statement
    print(f"Resumed from generation {generation} with {len(population)} individuals")
    print(f"Best fitness from that generation: {best_data.get('body_fitness', 'Unknown')}")
    return population, best_data

def get_next_run_id(sim_config: EAConfig) -> int:
    """
    Find the next available run ID to avoid overwriting.
    F.e. if '__data__/A3_modified_run_0' already exists, 
    it will try run_1, and so on, until a valid number is
    found.
    
    Args:
        sim_config (instance): The simulation configuration file
        
    Returns: 
        run_id (int): The next available run ID
    """
    run_id = 0
    while (sim_config.data / f"run_{run_id}").exists():
        run_id += 1
    return run_id

def find_latest_generation(sim_config, run_id):
    """
    For a given run, find the latest generation that exists,
    and return the number of that generation.
    
    Args:
        sim_config (instance):  The simulation configuration file
        run_id (int):           The run where to search the latest 
                                generation
    
    Returns:
        latest_gen (int):       The number of the latest generation found.
                                If run directory doesn't extist, returns (-1).
    """
    run_dir = sim_config.data / f"run_{run_id}"
    if not run_dir.exists():
        return -1
    latest_gen = -1
    # Look for generation directories
    for gen_dir in run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            try:
                gen_num = int(gen_dir.name.split("_")[1])
                if (gen_dir / "body_population.pkl").exists() and (gen_dir / "best_performers.pkl").exists():
                    latest_gen = max(latest_gen, gen_num)
            except (ValueError, IndexError):
                continue
    return latest_gen


### === Plotting ===
def plot_run_statistics(sim_config, run_id):
    run_dir = sim_config.data / f"run_{run_id}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory {run_dir} does not exist.")
    
    run_means = []
    run_stds = []
    run_bests = []
    for gen_dir in run_dir.iterdir():
        if gen_dir.is_dir() and gen_dir.name.startswith("generation_"):
            gen_num = int(gen_dir.name.split("_")[1])
            pop, best_data = load_population_from_generation(sim_config, gen_num, run_id)
            mean = np.mean([ind.fitness.values[0] for ind in pop if ind.fitness.valid])
            std = np.std([ind.fitness.values[0] for ind in pop if ind.fitness.valid])
            best = best_data.get("body_fitness", None)
            run_means.append(mean)
            run_stds.append(std)
            run_bests.append(best)
    
    run_means = np.array(run_means)
    run_stds = np.array(run_stds)
    run_bests = np.array(run_bests)
    x = np.arange(len(run_means))
    fig, ax = plt.subplots()
    ax.plot(x, run_means, linestyle = "--", linewidth = 1.5, color = "blue", label = "Mean Fitness")
    ax.fill_between(x, run_means - run_stds, run_means + run_stds, color="blue", alpha=0.2, label="Std. Dev.")
    ax.plot(x, run_bests, linestyle = "-", linewidth = 2, color = "red", label = "Best Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    ax.legend()
    plt.title(f"Run {run_id} - Fitness over Generations")
    save_unique_png(fig, path=f"__data__/A3_modified/run_{run_id}_", ext = f".png")

def save_unique_png(fig, path = "__data__/", ext = ".png"):
    """Function to save plt figures with unique filenames. To prevent
       overwriting existing plots.

    Args:
        fig (matplotlib.figure): Figure to save
        path (str, optional): The path of the directory where the figure 
                              is to be saved. Defaults to "__data__/".

    Returns:
        _type_: The path of the saved file
    """
    i = 0
    filename = f"{path}image{ext}"
    while os.path.exists(filename):
        i += 1
        filename = f"{path}image_{i}{ext}"
    fig.savefig(filename)
    return filename

def show_xpos_history(history: list[float], sim_config) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90
    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(sim_config.data / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )
    # Setup background image
    img = plt.imread(save_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, sim_config.spawn_position[0]
    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)
    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")
    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    # Title
    plt.title("Robot Path in XY Plane")
    # Show results
    save_unique_png(fig)

### === Distance measures and fitnesses
def fitness_function(history: list[float], sim_config) -> float:
    """
    A fitness function that maximizes the inverse of distance to target
    
    Args:
        history (list[float]):  The history of positions
        sim_config (instance):  The simulation configuration

    Returns:
        fitness (float):        The fitness value, to be maximized

    """
    xt, yt, zt = sim_config.target_position
    xc, yc, zc = history[-1]
    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    fitness = -cartesian_distance
    return fitness

def fitness_function_without_landing(history: list[float], sim_config) -> float:
    """
    Same as the above fitness function, but ignores the distance traveled 
    (4 seconds)
    """
    xt, yt, zt = sim_config.target_position
    # Determine positions after "landing" and at simulation end
    landing_frame = 4
    xi, yi, zi = history[landing_frame]
    xf, yf, zf = history[-1]
    initial_distance_to_target = np.sqrt((xt - xi) ** 2 + (yt - yi) ** 2 + (zt - zi) ** 2)
    final_distance_to_target = np.sqrt((xt - xf) ** 2 + (yt - yf) ** 2 + (zt - zf) ** 2)
    # Calculate distance traveled towards target, which is to be maximized
    distance_traveled_towards_target = initial_distance_to_target - final_distance_to_target
    return distance_traveled_towards_target

def diff_distance(history):
    """
    Calculate how much the robot moved between two frames of the simulation: 
    The 2nd second(as in time) until the end.

    Args: 
        history (list[float]):      The history of positions
    
    Returns:
        cartesian_distance (float): The distance moved from 2nd second until 
                                    the end of the simulation.
    """
    xc, yc = history[2][:2]
    xt, yt = history[-1][:2]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2)
    return cartesian_distance

def passed_checkpoint(checkpoint, history):
    """
    Check if the current robot position has passed a certain checkpoint in the x-direction
    (Note: x is y on the map)
    
    Args: 
        checkpoint (list[float]):   The checkpoint with (x, y, z) coordinates
        history (list[float]):      The simulation history of a robot
    
    Returns:
        passed_checkpoint (bool):   Whether the robot has passed the point in
                                    the simulation.
    """
    xc = history[-1][0]
    xt = checkpoint[0]
    passed_checkpoint = (xc >= xt)
    return passed_checkpoint
    

### === Controllers === 
def rw_controller(
    model: mj.MjModel,
    data: mj.MjData,
    matrices = None, 
    sim_config = None
) -> npt.NDArray[np.float64]:
    """
    Random walk controller
    """
    # Ensure seed is safe for parallelization
    if sim_config is None:
        sim_config = EAConfig()
    process_id = os.getpid()
    context_seed = int(time()) % 1000 + process_id
    rng = sim_config.create_reproducible_rng(context_seed)
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu
    # Initialize the networks weights randomly
    w1 = rng.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = rng.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))
    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos
    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))
    # Scale the outputs
    return outputs * np.pi


### === Experiment and Brain Evaluation ===  
def experiment(
    robot: Any,
    controller: Controller,
    matrices,
    sim_config,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """
    Run the simulation for a single robot body with a controller. The 
    duration of the simulation is dependent on whether the robot has
    passed specified checkpoints in sim_config
    
    Args:
        robot (DiGraph):            The robot graph object
        controller (Controller):    The controller to use for the simulation
                                    (NOT the controller function!)
        matrices:                   The weight matrices of the brain
        sim_config:                 The simulation configuration object
        duration (int):             The duration of the simulation
        mode (ViewerTypes):         The mode to use for the simulation
    """
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE
    # Initialise world
    world = OlympicArena()
    # Spawn robot in the world, check docstring for spawn conditions
    world.spawn(robot.spec, spawn_position=sim_config.spawn_position.copy())
    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)
    # Reset state and time of simulation
    mj.mj_resetData(model, data)
    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)
    # Set the control callback function
    args: list[Any] = [matrices, sim_config]  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
            # Continue simulation for robots starting from normal spawn, if they passed checkpoints
            if sim_config.spawn_position == sim_config.start_normal and passed_checkpoint(sim_config.checkpoint_rugged, controller.tracker.history["xpos"][0]):
                console.log("Passed checkpoint, continue simulation")
                continue_simple_runner(
                    model,
                    data,
                    duration = sim_config.duration_rugged
                )
            if sim_config.spawn_position == sim_config.start_normal and passed_checkpoint(sim_config.checkpoint_elevated, controller.tracker.history["xpos"][0]):
                console.log("Passed checkpoint, continue simulation")
                continue_simple_runner(
                    model,
                    data,
                    duration = sim_config.duration_elevated
                )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(sim_config.data / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(sim_config.data / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
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
    brain_genotype,
    robot_graph,
    controller_func,
    network_specs,
    sim_config,
    experiment_mode = "simple",
    initial_duration = 15
    ) -> Tuple[float, ] :
    """
    Evaluate a single robot fitness based on its performance in the experiment.

    Args:
        brain_genotype (list):              Individual brain genotype
        robot_graph (Any):                  The robot specs (body phenotype)
        controller_func (Callable):         The controller function to use
        network_specs (dict[str, int]):     A dictionary including the following keys 
                                            - 'input_size' 'output_size' 'hidden_size' 'no_hidden_layers'
        experiment_mode (str):              Rendering/simulation mode options. Defaults to "simple".

    Returns:
        fitness (float, ): DEAP style fitness score
    """
    # Construct robot specs, tracker, and controller
    robot_spec = construct_mjspec_from_graph(robot_graph)
    # Decode genotype to weight matrices
    w_matrices = decode_brain_genotype(brain_genotype=brain_genotype, network_specs=network_specs)
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    # Run experiment
    ctrl = Controller(controller_callback_function=controller_func, tracker=tracker)
    experiment(
        robot =         robot_spec,
        controller =    ctrl,
        matrices =      w_matrices,
        sim_config =    sim_config,
        duration =      initial_duration,
        mode =          experiment_mode)
    # Return fitness
    fitness = (fitness_function(tracker.history["xpos"][0], sim_config = sim_config), )
    return fitness
    
    
### === Handle body NDE and robot graphs === 
def create_robot_graph(
    body_genotype,
    sim_config,
    nde=None
    ):
    """
    Create a robot graph from a body genotype using NDE and HPD.
    Provide an nde to generate an exact robot_graph, otherwise
    the robot graph will be stochastically generated (by a generic NDE).
    
    Args:   
        body_genotype (DEAP list):          The body genotype
        sim_config (instance):              The simulation configuration file
        nde (NeuralDevelopmentalEncoding):  An optional NDE instance
    
    Returns:
        robot_graph (DiGraph):              The robot specifications created
                                            from the genotype and nde.
    """
    if nde is None:
        body_genotype = decode_body_genotype(genotype = body_genotype, genotype_size=64)
        # Input body genotype
        nde = NeuralDevelopmentalEncoding(number_of_modules=sim_config.num_of_modules)
        p_matrices = nde.forward(body_genotype)
        # Decode the high-probability graph
        hpd = HighProbabilityDecoder(sim_config.num_of_modules)
        robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )
        # Save the graph to a file
        save_graph_as_json(
            robot_graph,
            sim_config.data / "robot_graph.json",
        )
    else:
        body_genotype = decode_body_genotype(genotype = body_genotype, genotype_size=64)
        # Use given nde
        p_matrices = nde.forward(body_genotype)
        # Decode the high-probability graph
        hpd = HighProbabilityDecoder(sim_config.num_of_modules)
        robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
            p_matrices[0],
            p_matrices[1],
            p_matrices[2],
        )
        # Save the graph to a file
        save_graph_as_json(
            robot_graph,
            sim_config.data / "robot_graph.json",
        )
    return robot_graph

def delete_after_variation(ind)-> None: 
    """
    Delete fitness, best_brain, nde, and robot_graph attributes, f.e. after mutation changes the body
    genotype.

    Args:
        ind (DEAP list): The body genotype
    """
    del ind.fitness.values
    if hasattr(ind, 'best_brain'):
        del ind.best_brain
    if hasattr(ind, 'nde'):
        del ind.nde
    if hasattr(ind, 'robot_graph'):
        del ind.robot_graph
            
def attach_nde_graph(ind, sim_config):
    """Attach a new NDE and robot graph to a body genotype as attributes. The robot graph is
       constructed using the newly created nde!

    Args:
        ind (DEAP list):            The body genotype
        sim_config (dataclass):     The simulation configuration file
    """
    ind.nde = NeuralDevelopmentalEncoding(number_of_modules=sim_config.num_of_modules)
    ind.robot_graph = create_robot_graph(ind, sim_config, ind.nde)
    

### === Functions to ensure viable robot bodies ===
def is_viable_body(
    body_genotype, 
    sim_config, 
    gate_time = 6.0, 
    delta_min = 0.05, 
    mode = "simple", 
    max_fps = None
    ):
    """
    Check whether a body_genotype is able to generate viable bodies for movement.

    Args:
        body_genotype (DEAP list):  The body genotype
        sim_config (instance):      The simulation configuration file
        gate_time (float):          The time duration for the check. Defaults to 6.0.
        delta_min (float):          The minimum displacement for viability. Defaults to 0.05.
        mode (str):                 The mode of experiment. Defaults to "simple".
        max_fps (int):              The maximum frames per second. Defaults to None.

    Returns:
        viable (bool): Whether the body is viable for movement.
    """
    # Create the robot graph and specs
    robot_graph =   create_robot_graph(body_genotype=body_genotype, sim_config = sim_config)
    robot_spec =    construct_mjspec_from_graph(robot_graph)
    # Conduct experiment
    tracker =       Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    ctrl =          Controller(controller_callback_function=rw_controller, tracker=tracker)
    experiment(robot_spec, controller=ctrl, matrices=None, sim_config=sim_config, duration=gate_time, mode=mode)
    # Check whether robot moves sufficiently to be viable
    xpos =          tracker.history["xpos"][0]
    disp =          diff_distance(xpos)
    vel_est =       disp/max(gate_time, 1e-6)
    viable =        (disp >= delta_min) or (vel_est >= delta_min / gate_time)
    return viable

def make_viable_body(
    base_body_generator, 
    sim_config, 
    delta = 0.1, 
    mode = "simple"
    ):
    """
    Generate a viable individual body genotype.

    Args:
        base_body_generator (function):     A function that generates a body genotype. The generated 
                                            genotype should be a DEAP list.
        sim_config (instance):              The simulation configuration object
        delta (float, optional):            The minimum displacement for viability. Defaults to 0.1.
        mode (str, optional):               The mode of experiment. Defaults to "simple".
    Returns:
        g (DEAP list): A viable individual body genotype
    """
    g = base_body_generator()
    while not is_viable_body(g, sim_config = sim_config, gate_time= 5.0, delta_min=delta, mode=mode):
        g = base_body_generator()
    return g


### === Mutation/Crossover Operators ===
def whole_arithmetic_recomb(ind1, ind2, alpha):
    """
    Whole arithmetic crossover operator for EA

    Args:
        ind1 (DEAP list):   Parent number 1
        ind2 (DEAP list):   Parent number 2
        alpha (float):      Blending factor

    Returns:
        Tuple[DEAP list, DEAP list]: Crossover offspring
    """
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        cross_value_1 = alpha * x2 + (1 - alpha) * x1
        cross_value_2 = (1-alpha) * x2 + alpha * x1
        ind1[i] = cross_value_1
        ind2[i] = cross_value_2
    return ind1, ind2    


### === Debug ===
def debug_population_diversity(population):
    """
    Debug function to check for truly identical individuals. Prints and outputs
    the number of identical pairs found.
    
    Args:
        population (list[DEAP list]):   The population of individuals to check.

    Returns:
        n_identical (int):              The number of identical pairs in the 
                                        population.
    """
    identical_pairs = []
    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            ind1, ind2 = population[i], population[j]
            # Check if truly identical
            if len(ind1) == len(ind2):
                # Element-wise comparison with very small tolerance
                differences = [abs(x1 - x2) for x1, x2 in zip(ind1, ind2)]
                max_diff = max(differences) if differences else 0
                if max_diff < 1e-10:  # Truly identical
                    identical_pairs.append((i, j, max_diff))
                elif max_diff < 1e-6:  # Very similar
                    print(f"Very similar individuals {i} and {j}: max_diff = {max_diff}")
    if identical_pairs:
        print(f"Found {len(identical_pairs)} truly identical pairs: {identical_pairs}")
    else:
        print("No truly identical individuals found in initial population")
    n_identical = len(identical_pairs)
    return n_identical


### === EAs === 
def EA_brain(robot_graph, ea_brain_config, sim_config, ind_type, mode):
    
    """
    The main EA algorithm for the brain. Finds the best brain possible for a given
    body, the given parameters, and the EA-constraints set in the configuration
    files, and returns the fitness produced, the brain genotype, and the nde used to produce 
    the body
    Args:
        robot
    """
    # Create brain toolbox
    toolbox_brain = base.Toolbox()
    # Define the network specifications
    input_size, output_size = find_in_out_size(robot_graph, sim_config.spawn_position.copy())
    network_specs = {
        "input_size" :          input_size,
        "output_size" :         output_size,
        "hidden_size" :         ea_brain_config.hidden_size,
        "no_hidden_layers" :    ea_brain_config.no_hidden_layers
    }    
    # Calculate the size of a brain genotype, based on network specs
    ind_size = compute_brain_genome_size(network_specs)
    # Register in toolbox 
    register_factories(
        t=              toolbox_brain,
        ind_type=       ind_type,
        init_func=      ea_brain_config.init_func,
        t_attr_name=    "attr_float",
        t_ind_name=     "create_brain_genome",
        t_pop_name=     "create_brain_genome_pop",
        no_alleles=     ind_size,
    )
    toolbox_brain.register("map", map)
    toolbox_brain.register(
        "EvaluateRobot",
        evaluate_robot,
        robot_graph =       robot_graph, # This is the phenotyp expression of the body genotype.
        controller_func =   nn_controller,
        experiment_mode =   mode,
        sim_config =        sim_config,
        network_specs =     network_specs,
        initial_duration =   15
    )
    toolbox_brain.register(
        "ParentSelectBrain",
        tools.selTournament,
        tournsize = ea_brain_config.tourn_size
    )
    toolbox_brain.register(
        "SurvivalSelectBrain",
        tools.selBest,
        k = ea_brain_config.pop_size_brain
    )
    toolbox_brain.register(
        "MateBrain",
        whole_arithmetic_recomb,
        alpha = ea_brain_config.wa_alpha
    )
    toolbox_brain.register(
        "MutateBrain",
        tools.mutGaussian,
        mu = ea_brain_config.gauss_mut_mu,
        sigma = ea_brain_config.gauss_mut_sigma,
        indpb = ea_brain_config.gauss_mut_indpb
    )
    ### === Evolutionary Algorithm for Brain ===
    champions = []
    for r in range(ea_brain_config.runs_brain):
        # Create population
        pop_brain_genotype = toolbox_brain.create_brain_genome_pop(n = ea_brain_config.pop_size_brain)
        # debug_population_diversity(pop_brain_genotype)
        # Assign each individual a fitness value
        f_brain_genotype = list(toolbox_brain.map(toolbox_brain.EvaluateRobot, pop_brain_genotype))
        for ind, f in zip(pop_brain_genotype, f_brain_genotype):
            ind.fitness.values = f
        print("First gen stats:")
        print_statistics(pop_brain_genotype)
        # Go through generations
        for g in range(ea_brain_config.ngen_brain):
            offspring = toolbox_brain.ParentSelectBrain(pop_brain_genotype, k = ea_brain_config.pop_size_brain)
            offspring = list(toolbox_brain.map(toolbox_brain.clone, offspring))
            random.shuffle(offspring)
            # Apply variation operators
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < ea_brain_config.cxpb_brain:
                    toolbox_brain.MateBrain(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            for mutant in offspring:
                if random.random() < ea_brain_config.mutpb_brain:
                    toolbox_brain.MutateBrain(mutant)
                    del mutant.fitness.values
            # Evaluate offspring fitnesses of individuals which had genotypes changed by mating and mutating
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_fitnesses = list(toolbox_brain.map(toolbox_brain.EvaluateRobot, invalid_ind))
            for ind, fit in zip(invalid_ind, invalid_fitnesses):
                ind.fitness.values = fit
            # Survival selection + Elitism
            pop_brain_genotype[:] = toolbox_brain.SurvivalSelectBrain(offspring + tools.selBest(pop_brain_genotype, k = ea_brain_config.elites_brain))
            print("Brain EA")
            print_statistics(pop_brain_genotype)
        champions.append(tools.selBest(pop_brain_genotype, k = 1)[0])
    best_brain = tools.selBest(champions, k = 1)[0]
    return best_brain.fitness.values, best_brain
    

def EA_body(
    ea_brain_config,
    ea_body_config,
    sim_config,
    resume_from_generation = -1,
    resume_run_id = 0
    ):
    """
    The EA algorithm for the body. Running the algorithm will save population data to __data__. 
    
    Args:
        ea_brain_config (dataclass):    The Brain EA configuration file
        ea_body_config (dataclass):     The Body EA configuration file
        sim_config (dataclass):         The Simulation configuration file
        resume_from_generation (int):   Which saved generation to resume from. Defaults to -1
        resume_run_id (int):            Which saved run to resume from. Defaults to 0

    Returns:
        best_of_all(DEAP list):         The best individual across all generations of this EA simulation.
    """
    # Define genotype size for body
    body_genotype_size = 3*64
    # Ensure the deap types are in creator
    _, ind_type = ensure_deap_types()
    # Create body toolbox
    toolbox_body = base.Toolbox()
    toolbox_body.register("map", futures.map) 
    toolbox_body.register("attr_float", random.random)
    toolbox_body.register("individual", tools.initRepeat, ind_type, toolbox_body.attr_float, n=body_genotype_size)
    toolbox_body.register("make_viable_body", make_viable_body, sim_config=sim_config, base_body_generator=toolbox_body.individual, delta=0.2)
    toolbox_body.register("population", tools.initRepeat, list, toolbox_body.make_viable_body)
    toolbox_body.register("EvaluateRobotBody", EA_brain, ea_brain_config=ea_brain_config, sim_config=sim_config, ind_type=ind_type, mode="simple")
    toolbox_body.register("ParentSelectBody", tools.selTournament, tournsize=ea_body_config.tourn_size)
    toolbox_body.register("SurvivalSelectBody", tools.selBest, k=ea_body_config.pop_size_body)
    toolbox_body.register("MateBody", whole_arithmetic_recomb, alpha=ea_body_config.wa_alpha)
    toolbox_body.register("MutateBody", tools.mutGaussian, mu=ea_body_config.gauss_mut_mu, sigma=ea_body_config.gauss_mut_sigma, indpb=ea_body_config.gauss_mut_indpb)
    # --- EA ---
    champions = []
    for _ in range(ea_body_config.runs_body):
        # If resuming from a generation, load that population
        if resume_from_generation >= 0:
            try:
                pop_body_genotype, _ = load_population_from_generation(
                    sim_config = sim_config,
                    generation = resume_from_generation,
                    run_id = resume_run_id)
                start_generation = resume_from_generation + 1
                print(f"Resuming run {resume_run_id} from generation {start_generation}")
            except FileNotFoundError as e:
                print(f"Resume failed: {e}")
                print("Starting new run instead")
                pop_body_genotype = toolbox_body.population(n = ea_body_config.pop_size_body)
                start_generation = 0
                # Evaluate initial population
                for ind in pop_body_genotype:
                    attach_nde_graph(ind, sim_config)
                f_body_genotype = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [ind.robot_graph for ind in pop_body_genotype]))
                for ind, (f, best_brain) in zip(pop_body_genotype, f_body_genotype):
                    ind.fitness.values = f
                    ind.best_brain = best_brain
        else:
            # New run
            pop_body_genotype = toolbox_body.population(n = ea_body_config.pop_size_body)
            start_generation = 0
            # Attach an nde and the robot graph created from that nde to each body genome 
            for ind in pop_body_genotype:
                attach_nde_graph(ind, sim_config)
            # Initial population evaluation
            f_body_genotype = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [ind.robot_graph for ind in pop_body_genotype]))
            for ind, (f, best_brain) in zip(pop_body_genotype, f_body_genotype):
                ind.fitness.values = f
                ind.best_brain = best_brain
                
        # Go through generations
        end_generation = start_generation + ea_body_config.ngen_body
        for g in range(start_generation, end_generation):
            offspring = toolbox_body.ParentSelectBody(pop_body_genotype, k = ea_body_config.pop_size_body)
            offspring = list(toolbox_body.map(toolbox_body.clone, offspring))
            random.shuffle(offspring)
            # Apply variation operators
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < ea_body_config.cxpb_body:
                    toolbox_body.MateBody(child1, child2)
                    delete_after_variation(child1)
                    delete_after_variation(child2)
            for mutant in offspring:
                if random.random() < ea_body_config.mutpb_body:
                    toolbox_body.MutateBody(mutant)
                    delete_after_variation(mutant)
            # Select individuals that were modified (previous fitness and graph are invalid now)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Make sure alleles remain within bounds, then assign new nde and robot graphs
            for ind in invalid_ind:
                for i in range(len(ind)):
                    if ind[i] < 0.0:
                        ind[i] = 0.0
                    elif ind[i] > 1.0:
                        ind[i] = 1.0
                attach_nde_graph(ind, sim_config)
            invalid_fitnesses = list(toolbox_body.map(toolbox_body.EvaluateRobotBody, [ind.robot_graph for ind in invalid_ind]))
            for ind, (f, best_brain) in zip(invalid_ind, invalid_fitnesses):
                ind.fitness.values = f
                ind.best_brain = best_brain
            # Survival selection + Elitism
            pop_body_genotype[:] = toolbox_body.SurvivalSelectBody(offspring + tools.selBest(pop_body_genotype, k = ea_body_config.elites_body))
            # Save generation data (Note that genotypes are not saved as DEAP lists, only normal lists)
            best_body = tools.selBest(pop_body_genotype, k = 1)[0]
            best_brain = best_body.best_brain
            save_generation(g, pop_body_genotype, best_body, best_brain, run_id = str(resume_run_id), sim_config= sim_config)
            # Print generation statistics
            print("Body EA")
            print_statistics(pop_body_genotype)
        # Only resume from saved generation for the first run
        resume_from_generation = -1
        champions.append(tools.selBest(pop_body_genotype, k = 1)[0])
    best_of_all = tools.selBest(champions, k = 1)[0]
    return best_of_all

### === Main ===
def main(
    sim_config,
    run_id = 0,
    auto_resume = True,
    force_new_run = False):
    """
    This is the main simulation file. All the parameters of the EA can be changed here.
    To do so, take a look at the configuration files that are instantiated below. 

    Args:
        sim_config (dataclass):         The main configuration, specified outside main 
        run_id (int, optional):         The run ID number for the simulation. Defaults to 0.
        auto_resume (bool, optional):   Whether to automatically resume from the latest generation. Defaults to True.
        force_new_run (bool, optional): Whether to force a new run, ignoring any existing data. 
                                        If true, it ignores the run_id. Defaults to False.
    """
    
    # Create configuration files
    sim_config = replace(sim_config) # Instantiated outside main
    
    ### --- [INTERFACE] Choose your EA parameters here! ---  
    
    # Brain EA
    ea_brain_config = EABrainConfig(
        # General EA parameters
        runs_brain =            1,
        ngen_brain =            50,
        pop_size_brain =        50,
        cxpb_brain =            0.7,
        mutpb_brain =           0.1,
        elites_brain =          1,
        # Network structure
        hidden_size=            128,
        no_hidden_layers=       3,
        # Initialization function for brain genotype
        init_func=              partial(np.random.uniform, -1, 1),
        # Mutation parameters
        gauss_mut_mu=           0.0,
        gauss_mut_sigma=        0.15,
        gauss_mut_indpb=        0.15,
        # Crossover parameters
        wa_alpha=               0.5,
        # Selection parameters
        tourn_size=             3
    )
    ea_body_config = EABodyConfig(
        # General EA parameters
        runs_body=              1,
        ngen_body=              1,
        pop_size_body=          10,
        cxpb_body=              0.7,
        mutpb_body=             0.1,
        elites_body=            1, # PLEASE note: If no elites, the final generation 
                                   #              of a run might not contain the best
                                   #              individual over the whole run!
        # Mutation parameters 
        gauss_mut_mu=           0.0,
        gauss_mut_sigma=        0.15,
        gauss_mut_indpb=        0.15,
        # Crossover parameters
        wa_alpha=               0.5,
        # Selection parameters
        tourn_size=             3,
        )
    
    ### ---
    
    # Set seeds for reproducibility
    random.seed(sim_config.rng_seed)
    np.random.seed(sim_config.rng_seed)
    

    resume_gen = -1
    resume_run = 0
    
    if force_new_run:
        # Force new run - get next available run ID
        new_run_id = get_next_run_id(sim_config)
        print(f"Forcing new run with ID: {new_run_id}")
        resume_gen = -1
        resume_run = new_run_id
        
    elif auto_resume:
        # Try to resume from existing run
        latest_gen = find_latest_generation(sim_config = sim_config, run_id = run_id)
        print(f"Latest generation found: {latest_gen}")
        
        if latest_gen >= 0:
            print(f"Found previous run with data up to generation {latest_gen}")
            response = input(f"Resume from generation {latest_gen + 1}? (y/n): ")
            if response == "y":
                resume_gen = latest_gen
                resume_run = run_id
            else:
                print("Abort Simulation")
                return None
        else:
            print(f"No generations found in run {resume_run}")
            print("Abort Simulation")
            return None
    else:
        # No auto resume, no force new - use provided run_id
        print(f"Starting fresh run with ID: {run_id}")
        resume_run = run_id
        
    start = time()
    best_robot = EA_body(
        ea_brain_config=ea_brain_config,
        ea_body_config=ea_body_config,
        sim_config=sim_config,
        resume_from_generation=resume_gen,
        resume_run_id=resume_run
    )
    end = time()
    print(f"Elapsed time: {end - start:.2f} seconds")
    print(f"Best fitness found in this simulation: {best_robot.fitness.values[0]:.6f}")
         
    # ? ------------------------------------------------------------------ #
    
if __name__ == "__main__":
    # Setup main configuration file
    sim_config = EAConfig(rng_seed=42)
    
    ##################
    # Main Interface #
    ##################    
    """
    The interface currently has three options: Running an EA algorithm, rendering a particular
    individual produced by the EA algorithm, or inspect the best fitness of a generation.
    To choose which to do, set the below variables accordingly (you can have all three active).
    
    Note: If you want to have the simulation, you need to run the script with
        python -m scoop examples/A3_modified.py  
    otherwise, there is no parallelization, slowing down the EA substantially. 
    
    But if you are not running the simulation, I would recommend just running it normally, f.e. 
    with uv run, because you do not get these annoying syntax warnings that scoop gives. 
    """
    SIMULATE = True 
    RENDER = False      
    INSPECT = False

    """
    SIMULATION
    ----------
    Here you can choose to either run a new cycle, or to restart a previous run. 
    For resuming a run, set RESUME to True, and specify the run to continue. 
    For starting a new run, set RESUME to False, and the RESUME_RUN parameter 
    is ignored (you can just leave it).
    """
    RESUME = False
    RESUME_RUN = 0
    
    """
    RENDERING
    ---------
    If you want to render the best individual of a given generation and run, specify them.
    Note that the renderer assumes the default network structure. If you change the network in main(),
    you then have to make the same changes to the interface code at the bottom of the script.
    """
    RENDER_GEN = 5
    RENDER_RUN = 0
    
    """
    INSPECT
    -------
    To evaluate the best found fitness for a run and generation, set which run and generation you
    want to inspect:
    """
    INSPECT_GEN = 1
    INSPECT_RUN = 0
    
    ### --- Interface code ---
    if SIMULATE:
        
        if RESUME:
            main(
                sim_config =    sim_config, 
                auto_resume=    True, 
                run_id =        RESUME_RUN # specify which run to continue
                )
        else:
            main(
                sim_config =    sim_config, 
                force_new_run = True
                )

    if RENDER:
        pop, best_data = load_population_from_generation(
            sim_config = sim_config,
            generation = RENDER_GEN,
            run_id = RENDER_RUN
        )
        best_robot = best_data["best_body_genotype"]
        best_brain = best_data["best_brain_genotype"]
        best_robot_graph = tools.selBest(pop, k=1)[0].robot_graph
        input_size, output_size = find_in_out_size(best_robot_graph, sim_config.spawn_position.copy())
        print_statistics(pop)
        evaluate_robot(
            brain_genotype = best_brain,
            robot_graph = best_robot_graph,
            controller_func = nn_controller,
            network_specs = {
                "input_size" :          input_size,
                "output_size" :         output_size,
                "hidden_size" :         128,
                "no_hidden_layers" :    3
            },
            sim_config = sim_config,
            experiment_mode = "launcher",
            initial_duration = 120
        )
    if INSPECT:
        # Loading itself should output the best fitness of that gen.
        # pop, best_data = load_population_from_generation(
        #     sim_config = sim_config,
        #     generation = INSPECT_GEN,
        #     run_id = INSPECT_RUN
        # )
        plot_run_statistics(sim_config, INSPECT_RUN)

    
    
    
    
    
        
        


    
    
