# Imports
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import random
import time
from pathlib import Path

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

# Import torch for brain controller
from torch import nn

# Ariel Imports
# New EA engine (ea.py)
from ariel.ec import (
    EA,
    EAOperation,
    EASettings,
    Individual,
    Population,
)

# Body imports
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

import gymnasium as gym
import envs

from evogym_helper import EvoGymHelper, EvoGymWrapper

# Initialize rich console and traceback handler
install()
console = Console()

# Will probably have to fix the paths at some point
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / "Robot_Evolution")
DATA.mkdir(exist_ok=True)

# A seed is optional, but it helps with reproducibility
SEED = None  # e.g., 42

NUM_MODULES = 20
GENE_SIZE = 64  # default is 64, change it in the decoder/NDE
GENE_RANGE = 64
POP_SIZE = 5
NUM_GENERATIONS = 2

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
torch.save(NDE.state_dict(), DATA / "NDE.pth")


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
    genotype = np.full((5, 5), 0.0)

    genotype[RNG.integers(0, 5)][RNG.integers(0, 5)] = float(RNG.integers(3, 5))
    for i in range(14):
        success = False
        while not success:
            new_grid = np.copy(genotype)
            x = RNG.integers(0, 5)
            y = RNG.integers(0, 5)
            if new_grid[x][y] != 0.0:
                continue

            new_grid[x][y] = float(RNG.integers(1, 5))
            if not EvoGymHelper.grid_is_ok(new_grid, 5):
                continue

            genotype = new_grid
            success = True

    ind = Individual()
    ind.genotype = genotype.tolist()
    return ind


# Currently Completed
def gene_to_robot(individual: Individual):
    return np.array(individual.genotype)


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
        ind.tags["ps"] = 0

    # Decide how many parents we want (even number)
    num_parents = len(population)

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

        winner.tags["ps"] += 1

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
    for ind in population:
        if ind.tags["ps"] == 0:
            continue
        for i in range(ind.tags["ps"]):
            child = Individual()
            child.genotype = ind.genotype
            child.tags = {"mut": True}
            child.requires_eval = True

            population.append(child)
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
            while True:
                change_indices = random.sample(range(25), 1)

                new_grid = []
                i = 0
                for horizontal in ind.genotype:
                    new_horizontal = []
                    for vertical in horizontal:
                        new_value = vertical

                        if i in change_indices:
                            while new_value == vertical:
                                new_value = random.sample(range(5), 1)[0]

                        i += 1
                        new_horizontal.append(new_value)
                    new_grid.append(new_horizontal)

                if EvoGymHelper.grid_is_ok(np.array(new_grid), 5):
                    break
            ind.genotype = new_grid
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
    robot_spec = gene_to_robot(individual)

    # 2. Simulation Setup (Logic from evaluate_single)
    evogym_env = gym.make('SimpleWalkingEnv-v0', body=robot_spec, disable_env_checker=True)
    evogym_env = EvoGymWrapper(evogym_env)
    evogym_env.reset()
    num_vars: int = 2 # A lot more than this

    pop_size = 30
    budget = 50
    param = ng.p.Array(shape=(num_vars,))
    temp_vec_learner = ng.optimizers.registry["CMA"](
        parametrization=param, budget=(pop_size * budget),
    )
    min_fit = np.inf
    for _ in range(budget):
        vecs = [temp_vec_learner.ask() for _ in range(pop_size)]

        for vec_candidate in vecs:
            vec = vec_candidate.value

            # TODO: Change, create controller

            # 4. Run Simulation
            sum_reward = 0
            for i in range(500):
                # TODO: Change, run controller
                action = evogym_env.get_action(None)
                _, reward, _, _, _ = evogym_env.step(action)
                sum_reward += reward

            fitness = sum_reward
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
    for ind in population:
        ind.fitness = individual_learn(ind)

    return population


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

    # Initial Eval
    population_list = pop_learn(Population(population_list)).to_list()

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
    ea = EA(
        Population(population_list),
        operations=ops,
        num_steps=NUM_GENERATIONS,
        db_file_path=config.db_file_path,
        db_handling="delete",
    )
    ea.run()

    return ea

def main() -> EA:
    """Is the main entry loop to the code."""
    return evolve()


start = time.time()

ea = main()

best_fitness = ea.get_solution("best", only_alive=False).fitness

console.log(f"Best fitness found: {best_fitness:.3f}")
console.log("Best fitness possible: 0")


end = time.time()

time_taken = end - start

# Literally just to see the results better while testing
if time_taken < 60:
    console.log(f"Code took {time_taken:.3f} seconds to run")
elif time_taken < 60 * 60:
    console.log(f"Code took {time_taken / 60:.3f} minutes to run")
else:
    console.log(f"Code took {time_taken / (60 * 60):.3f} hours to run")