# Imports
import argparse
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco as mj
import mujoco.viewer

# Learning EA
import nevergrad as ng

# from mujoco import viewer
import numpy as np

# Ray for parallelisation
import torch

# Type Checking
# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install
from sqlmodel import Session, col, create_engine, select

# Import torch for brain controller
from torch import nn

# Image processing
from PIL import Image

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)

# Ariel Imports
# New EA engine (ea.py)
from ariel.ec import (
    EA,
    Crossover,
    EAOperation,
    EASettings,
    Individual,
    Population,
    config,
)

# Body imports
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.utils.data_get import get_state_from_data

# Import World
from ariel.simulation.environments import (
    SimpleFlatWorld,
)
from ariel.utils.runners import simple_runner

if TYPE_CHECKING:
    from networkx import DiGraph

# Initialize rich console and traceback handler
install()
console = Console()

"""
ARIEL Robot Evolution Example

This example demonstrates evolving robot morphologies and control policies together
using the new EA engine. It combines:
- Neural Developmental Encoding (NDE) for morphology generation
- High-probability decoding for valid robot graphs
- Nevergrad CMA-ES for controller learning
- ARIEL's EC framework for evolutionary optimization
- MuJoCo physics simulation and visualization

At the end, the best evolved robot is visualized and saved as a PNG image.

WARNING: This is computationally expensive due to the learning phase.
Each individual must run a full learning cycle (CMA-ES optimization).
For quick testing, adjust LEARNING_BUDGET and SIMULATION_DURATION.
For production runs, increase them for better results.
"""


# Will probably have to fix the paths at some point
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / "Robot_Evolution")
DATA.mkdir(exist_ok=True)

# Show cwd
# print(f"Current working directory: {CWD}")
# print(f"Saving data to {DATA}")

# A seed is optional, but it helps with reproducibility
SEED = None  # e.g., 42

NUM_MODULES = 20
GENE_SIZE = 64  # default is 64, change it in the decoder/NDE
GENE_RANGE = 64
POP_SIZE = 10
NUM_GENERATIONS = 5

# Learning hyperparameters (balance speed and quality)
LEARNING_BUDGET = 10  # Number of CMA-ES generations (default: 50)
LEARNING_POP_SIZE = 20  # Population size for CMA-ES (default: 30) - increased for better convergence
SIMULATION_DURATION = 2  # Simulation time in seconds (default: 15)

# Initialize RNG
RNG = np.random.default_rng(SEED)

# Set config
config = EASettings(
    is_maximisation=False,
    db_handling="delete",
    target_population_size=10,
    output_folder=DATA,
    db_file_name="database.db",
)

NDE = NeuralDevelopmentalEncoding(
    number_of_modules=NUM_MODULES,  # Seems to be a good value
    genotype_size=64,
)
NDE_PATH = DATA / "NDE.pth"
DB_PATH = DATA / "database.db"


def load_or_create_nde_weights() -> None:
    """Load saved NDE weights if available; otherwise create and save them."""
    if NDE_PATH.exists():
        state_dict = torch.load(NDE_PATH, map_location="cpu")
        NDE.load_state_dict(state_dict)
        console.log(f"Loaded NDE weights from: {NDE_PATH}")
        return

    torch.save(NDE.state_dict(), NDE_PATH)
    console.log(f"Saved new NDE weights to: {NDE_PATH}")


load_or_create_nde_weights()

class Network(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

        self.hidden_activation = nn.ELU()
        self.output_activation = nn.Tanh()

        self.input = input_size

        # Disable gradients for all parameters
        for param in self.parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def forward(self, model, data):
        x = torch.Tensor(get_state_from_data(data))

        x = self.hidden_activation(self.fc1(x))
        x = self.hidden_activation(self.fc2(x))
        x = self.output_activation(self.fc4(x)) * (torch.pi / 2)

        return x.detach().numpy()


@torch.no_grad()
def fill_parameters(net: nn.Module, vector: torch.Tensor) -> None:
    """Fill the parameters of a torch module (net) from a 1-D vector.

    No gradient information is kept.

    The vector's length must be exactly the same with the number
    of parameters of the PyTorch module.

    Parameters
    ----------
        net: nn.Module
            The torch module whose parameter values will be filled.
        vector: torch.Tensor
            A 1-D torch tensor which stores the parameter values.

    """
    address = 0
    for p in net.parameters():
        d = p.data.view(-1)
        n = len(d)
        d[:] = torch.as_tensor(vector[address : address + n], device=d.device)
        address += n


# Currently Completed
def create_random_individual() -> Individual:
    """Create and initialise a random BODY individual.

    Returns
    -------
    ind: Individual
        A newly created individual with a randomized genotype.
    """
    ind = Individual()
    ind.genotype = RNG.normal(
        loc=0,
        scale=64,
        size=(3, GENE_SIZE),
    ).tolist()

    return ind


# Currently Completed
def gene_to_robot(genotype, num_modules: int = 20) -> mj.MjSpec:
    """Generate a robot specification from a gene.

    Parameters
    ----------
    genotype: list or array-like
        The genotype of the individual to decode.
    num_modules: int, default=20
        The number of modules to use in the HighProbabilityDecoder.

    Returns
    -------
    mujoco.MjSpec
        The generated MuJoCo specification for the robot.
    """
    p_matrices = NDE.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    robot_spec = construct_mjspec_from_graph(robot_graph)
    return robot_spec.spec


# Currently Completed
def parent_selection(population: Population) -> Population:
    """Tournament Selection.

    Selects parents for the next generation using a tournament selection 
    mechanism. Tags the winners with 'ps' (parent selection).

    Parameters
    ----------
    population : Population
        The current population of individuals.

    Returns
    -------
    Population
        The updated population with selected parents tagged.
    """
    tournament_size: int = 3

    # Ensure all individuals have a tags dict and reset parent-selection tag
    for ind in population:
        ind.tags["ps"] = False

    # Decide how many parents we want (even number)
    num_parents = (len(population) // 2) * 2
    if num_parents == 0 and len(population) >= 2:
        num_parents = 2

    for _ in range(num_parents):
        # sample competitors with replacement
        competitors = [
            random.choice(population) for _ in range(tournament_size)
        ]

        # pick best competitor depending on maximisation/minimisation
        if config.is_maximisation:
            winner = max(competitors, key=lambda ind: ind.fitness)
        else:
            winner = min(competitors, key=lambda ind: ind.fitness)

        winner.tags["ps"] = True

    return population


# Currently Completed
def crossover(population: Population) -> Population:
    """One point crossover.

    Performs one-point crossover on individuals tagged for parent 
    selection ('ps'). Generates children and appends them to the 
    population with a 'mut' tag.

    Parameters
    ----------
    population : Population
        The current population containing selected parents.

    Returns
    -------
    Population
        The population extended with the newly created children.
    """
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents), 2):
        parent_i = parents[idx]
        parent_j = parents[idx]
        genotype_i, genotype_j = Crossover.one_point(
            parent_i.genotype,
            parent_j.genotype,
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

        # Add children to population
        population.extend([child_i, child_j])

    return population


# Currently Completed
def mutation(population: Population) -> Population:
    """"Gaussian mutation.

    Applies Gaussian mutation to individuals tagged for mutation ('mut').

    Parameters
    ----------
    population : Population
        The current population containing children to be mutated.

    Returns
    -------
    Population
        The population with mutated individuals.
    """
    for ind in population:
        if ind.tags.get("mut", False):
            gene_shape = np.array(ind.genotype).shape
            genes = np.array(ind.genotype).flatten().copy()

            if random.random() < 0.7:
                mutated = genes + RNG.normal(
                    loc=0,
                    scale=4,
                    size=len(genes),
                )
            else:
                mutated = genes.copy()

            ind.genotype = mutated.reshape(gene_shape).astype(float).tolist()
    return population


# Currently Completed
def survivor_selection(population: Population) -> Population:
    """Tournament Survivor Selection.

    Kills off individuals based on a tournament selection until the 
    population size is reduced back to the target size.

    Parameters
    ----------
    population : Population
        The current population including parents and children.

    Returns
    -------
    Population
        The surviving population.
    """
    tournament_size: int = 5

    pop_len = len(population)

    for _ in range(POP_SIZE):
        # Sample competitors with replacement
        pop_alive = [ind for ind in population if ind.alive is True]
        death_candidates = [
            # RNG.choice(pop_alive) for _ in range(tournament_size)
            random.choice(pop_alive)
            for _ in range(tournament_size)
        ]

        # Pick best competitor depending on maximisation/minimisation
        if config.is_maximisation:
            about_to_be_killed_lol = min(
                death_candidates,
                key=lambda ind: ind.fitness,
            )
        else:
            about_to_be_killed_lol = max(
                death_candidates,
                key=lambda ind: ind.fitness,
            )

        about_to_be_killed_lol.alive = False

        pop_len -= 1
        if pop_len <= POP_SIZE:
            break

    return population


def individual_learn(individual: Individual) -> float:
    """Perform learning for one individual.

    Evaluates an individual by decoding its genotype into a MuJoCo robot
    specification, setting up the simulation, and learning optimal controller
    weights using CMA-ES via Nevergrad.

    Parameters
    ----------
    individual: Individual
        The individual whose fitness and controller are to be evaluated.

    Returns
    -------
    float
        The minimum fitness (distance) achieved during the learning budget.
    """
    p_matrices = NDE.forward(np.array(individual.genotype))

    # Hardcoded num_modules=20 based on your global var, or pass it in if dynamic
    hpd = HighProbabilityDecoder(num_modules=20)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    robot_spec = construct_mjspec_from_graph(robot_graph).spec

    # 2. Simulation Setup (Logic from evaluate_single)
    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(
        robot_spec,
        position=(0.0, 0.0, 0.1),
        correct_collision_with_floor=True,
    )

    model = world.spec.compile()
    data = mj.MjData(model)

    net = Network(
        input_size=len(get_state_from_data(data)),
        output_size=model.nu,
        hidden_size=32,
    )

    # Generate weights for vec
    num_vars: int = sum(p.numel() for p in net.parameters())

    min_fit = np.inf

    param = ng.p.Array(shape=(num_vars,))
    temp_vec_learner = ng.optimizers.registry["CMA"](
        parametrization=param,
        budget=(LEARNING_POP_SIZE * LEARNING_BUDGET),
    )

    tracker = Tracker(
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )

    tracker.setup(world.spec, data)

    controller = Controller(
        controller_callback_function=net.forward, tracker=tracker
    )
    for _ in range(LEARNING_BUDGET):
        vecs = [temp_vec_learner.ask() for _ in range(LEARNING_POP_SIZE)]

        for vec_candidate in vecs:
            vec = vec_candidate.value
            # 3. Network Construction
            fill_parameters(net, vec)

            mj.mj_resetData(model, data)

            mj.set_mjcb_control(controller.set_control)
            # 4. Run Simulation
            simple_runner(model, data, duration=SIMULATION_DURATION)

            # 5. Calculate Fitness
            xc, yc, zc = data.qpos[0:3].copy()
            xt, yt, zt = (2.0, 0.0, 0.1)
            fitness = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

            temp_vec_learner.tell(vec_candidate, fitness)

            min_fit = min(min_fit, fitness)

    return min_fit


def pop_learn(population: Population) -> Population:
    """Do learning for the entire population.

    Iterates over the population and evaluates the fitness of each individual
    by performing a learning cycle.

    Parameters
    ----------
    population : Population
        The current population to be evaluated.

    Returns
    -------
    Population
        The evaluated population with updated fitness values.
    """
    for idx, ind in enumerate(population):
        console.log(f"Learning individual {idx + 1}/{len(population)}...")
        ind.fitness = individual_learn(ind)
        console.log(f"  → Fitness: {ind.fitness:.3f}")

    return population


def get_best_individual_from_db(
    db_path: Path,
    *,
    is_maximisation: bool = False,
) -> Individual:
    """Load best evaluated individual from a saved EA SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")
    with Session(engine) as session:
        statement = select(Individual).where(Individual.requires_eval == False)  # noqa: E712
        if is_maximisation:
            statement = statement.order_by(col(Individual.fitness_).desc())
        else:
            statement = statement.order_by(col(Individual.fitness_).asc())

        best_individual = session.exec(statement).first()

    if best_individual is None:
        raise RuntimeError(
            "No evaluated individuals found in database. "
            "Run evolution first to populate fitness values."
        )

    return best_individual

def evolve() -> EA:
    """Entry point for the evolutionary algorithm.

    Initializes the population, evaluates it, defines the evolutionary 
    steps (parent selection, crossover, mutation, learning, survivor selection),
    and runs the EA.

    Returns
    -------
    EA
        The completed EA object containing the run history and solutions.
    """
    console.log("Initializing population...")

    # Initialise Body & Hivemind Population
    population_list = [create_random_individual() for _ in range(POP_SIZE)]
    console.log(f"Created {len(population_list)} random individuals")

    # Initial Eval
    console.log(f"Starting initial population learning (this may take a while)...")
    population_list = pop_learn(Population(population_list)).to_list()
    console.log("Initial population learning complete")

    # Define Evolution Loop
    # Operators work for both NDEs and Network Weight Vectors
    ops = [
        # Default EA operators
        EAOperation(parent_selection),  # Select parents for bodies
        EAOperation(crossover),  # Crossover Body
        EAOperation(mutation),  # Mutation Body
        # Learning acts as a fitness function
        EAOperation(pop_learn),  # Do learning for all the bodies
        EAOperation(survivor_selection),
    ]

    # Initialise EA object
    console.log("Starting evolutionary algorithm...")
    ea = EA(
        Population(population_list),
        operations=ops,
        num_steps=NUM_GENERATIONS,
        db_file_path=config.db_file_path,
        db_handling="delete",
    )
    ea.run()

    return ea

def visualize_individual(
    individual: Individual,
    title: str = "best",
    *,
    use_viewer: bool = False,
    viewer_duration: float | None = None,
) -> None:
    """Render a specific individual in simulation without re-learning."""
    console.log("\n[bold cyan]Visualizing best robot morphology...[/bold cyan]")

    console.log(f"{title.capitalize()} robot fitness: {individual.fitness:.3f}")
    
    # Decode morphology
    p_matrices = NDE.forward(np.array(individual.genotype))
    hpd = HighProbabilityDecoder(num_modules=20)
    robot_graph = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    robot_spec = construct_mjspec_from_graph(robot_graph).spec
    
    # Setup simulation
    mj.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(
        robot_spec,
        position=(0.0, 0.0, 0.1),
        correct_collision_with_floor=True,
    )
    
    model = world.spec.compile()
    data = mj.MjData(model)
    
    # Create network with randomized weights (no learning)
    net = Network(
        input_size=len(get_state_from_data(data)),
        output_size=model.nu,
        hidden_size=32,
    )
    
    tracker = Tracker(
        name_to_bind="core",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)
    
    controller = Controller(
        controller_callback_function=net.forward, tracker=tracker
    )
    
    # Run simulation with basic controller (no re-learning)
    mj.mj_resetData(model, data)
    mj.set_mjcb_control(controller.set_control)

    duration = SIMULATION_DURATION if viewer_duration is None else viewer_duration

    if use_viewer:
        console.log(
            f"Running best robot in MuJoCo viewer for {duration:.1f}s..."
        )
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sim_start = time.time()
            while viewer.is_running() and (time.time() - sim_start) < duration:
                step_start = time.time()
                mj.mj_step(model, data)
                viewer.sync()

                # Keep wall-clock and simulation time roughly aligned.
                remaining = model.opt.timestep - (time.time() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    else:
        console.log(f"Running best robot for {duration:.1f}s...")
        simple_runner(model, data, duration=duration)
    
    # Render and save final frame
    renderer = mj.Renderer(model)
    mj.mj_forward(model, data)
    pixels = renderer.render()
    
    # Save visualization
    img = Image.fromarray(pixels)
    output_path = DATA / "best_robot_visualization.png"
    img.save(output_path)
    console.log(f"✅ Visualization saved to: {output_path}")
    
    # Print final position
    xc, yc, zc = data.qpos[0:3].copy()
    console.log(f"Final position: ({xc:.3f}, {yc:.3f}, {zc:.3f})")
    console.log(f"Target position: (2.000, 0.000, 0.100)")


def simulate_best_from_saved_artifacts(
    *,
    use_viewer: bool = False,
    viewer_duration: float | None = None,
) -> None:
    """Load best individual from saved DB + NDE weights and simulate it."""
    if not NDE_PATH.exists():
        raise FileNotFoundError(f"NDE weights file not found: {NDE_PATH}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    # Ensure NDE in memory matches saved file used during evolution.
    state_dict = torch.load(NDE_PATH, map_location="cpu")
    NDE.load_state_dict(state_dict)
    console.log(f"Loaded NDE weights from: {NDE_PATH}")

    best_individual = get_best_individual_from_db(
        DB_PATH,
        is_maximisation=config.is_maximisation,
    )
    console.log(f"Loaded best individual from DB: {DB_PATH}")

    visualize_individual(
        best_individual,
        title="best (from saved db)",
        use_viewer=use_viewer,
        viewer_duration=viewer_duration,
    )


def main() -> EA:
    """Is the main entry loop to the code."""
    return evolve()


start = time.time()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--evolve",
    action="store_true",
    help="Run full evolution instead of loading best individual from saved artifacts.",
)
parser.add_argument(
    "--viewer",
    action="store_true",
    help="Open MuJoCo viewer and simulate in real time.",
)
parser.add_argument(
    "--viewer-duration",
    type=float,
    default=None,
    help="Simulation duration in seconds for viewer/headless playback.",
)
args = parser.parse_args()

# Default behavior now reuses saved artifacts when available.
if args.evolve:
    ea = main()

    best_fitness = ea.get_solution("best", only_alive=False).fitness

    console.log(f"Best fitness found: {best_fitness:.3f}")
    console.log("Best fitness possible: 0")

    # Visualize the best robot from the fresh EA run
    visualize_individual(
        ea.get_solution("best", only_alive=False),
        title="best",
        use_viewer=args.viewer,
        viewer_duration=args.viewer_duration,
    )
else:
    simulate_best_from_saved_artifacts(
        use_viewer=args.viewer,
        viewer_duration=args.viewer_duration,
    )

end = time.time()

time_taken = end - start

# Literally just to see the results better while testing
if time_taken < 60:
    console.log(f"Code took {time_taken:.3f} seconds to run")
elif time_taken < 60 * 60:
    console.log(f"Code took {time_taken / 60:.3f} minutes to run")
else:
    console.log(f"Code took {time_taken / (60 * 60):.3f} hours to run")
