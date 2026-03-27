# Standard library
import os
import random
import sys
import time
from typing import Literal, cast

# Third-party libraries
import numpy as np
import ray
from omegaconf import OmegaConf
from ompl import util as ou

# Pretty little errors and progress bars
from rich.console import Console
from rich.traceback import install

from ariel.body_phenotypes.Lynx_Arm.lynx.utils.env_sim_ompl import LynxPlanner

# Local libraries
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EA, EASettings, EAStep, Population
from ariel.ec.a005 import Crossover

sys.path.append(os.path.abspath('.'))

# Method 1: Set Log Level to suppress INFO and DEBUG
ou.noOutputHandler()

CFG = OmegaConf.load("src/ariel/body_phenotypes/Lynx_Arm/lynx/configs/sim.yaml")

# A seed is optional, but it helps with reproducibility
SEED = None  # e.g., 42

# The database has a few handling modes
#     "delete" will delete the existing database
#     "halt" will stop the execution if a database already exists
DB_HANDLING_MODES = Literal["delete", "halt"]

# Initialize RNG
RNG = np.random.default_rng(SEED)

# Initialize rich console and traceback handler
install()
console = Console()

# Set config
config = EASettings()
config.is_maximisation = False
config.db_handling = "delete"
config.target_population_size = 100


def create_individual(num_dims) -> Individual:
    ind = Individual()
    gene = np.array(np.random.normal(loc=0.5,
                            scale=0.3,
                            size=num_dims))
    ind.genotype = gene.clip(min=0, max=2).tolist()
    return ind


def parent_selection(population: Population) -> Population:
    """Tournament Selection."""
    tournament_size: int = 5

    # Ensure all individuals have a tags dict and reset parent-selection tag
    for ind in population:
        if ind.tags is None:
            ind.tags = {}
        ind.tags["ps"] = False

    # Decide how many parents we want (even number)
    num_parents = (len(population) // 2) * 2
    if num_parents == 0 and len(population) >= 2:
        num_parents = 2

    winners = []
    for _ in range(num_parents):
        # sample competitors with replacement
        competitors = [random.choice(population) for _ in range(tournament_size)]

        # pick best competitor depending on maximisation/minimisation
        if config.is_maximisation:
            winner = max(competitors, key=lambda ind: ind.fitness)
        else:
            winner = min(competitors, key=lambda ind: ind.fitness)

        winners.append(winner)

    # mark winners as parents
    for w in winners:
        w.tags["ps"] = True

    return population


def crossover(population: Population) -> Population:
    """One point crossover."""
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents), 2):
        parent_i = parents[idx]
        parent_j = parents[idx]
        genotype_i, genotype_j = Crossover.one_point(
            cast("list[float]", parent_i.genotype),
            cast("list[float]", parent_j.genotype),
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
            genes = list(ind.genotype)
            if random.random() < 0.5:
                mutated = [i + random.uniform(-0.1, 0.1) for i in genes]
            else:
                mutated = genes.copy()

            ind.genotype = mutated
    return population


def survivor_selection(population: Population) -> Population:
    tournament_size: int = 5

    # Decide how many parents we want (even number)
    pop_len = len(population)
    # if num_survivors == 0 and len(population) >= 2:
    #     num_survivors = 2

    for _ in range(config.target_population_size):
        # Sample competitors with replacement
        pop_alive = [ind for ind in population if ind.alive is True]
        death_candidates = [random.choice(pop_alive) for _ in range(tournament_size)]

        # Pick best competitor depending on maximisation/minimisation
        if config.is_maximisation:
            about_to_be_killed_lol = min(death_candidates, key=lambda ind: ind.fitness)
        else:
            about_to_be_killed_lol = max(death_candidates, key=lambda ind: ind.fitness)

        about_to_be_killed_lol.alive = False

        pop_len -= 1
        if pop_len <= config.target_population_size:
            break

    return population


@ray.remote
def evaluate_ind(ind):
    CFG.MorphConfig.robot_description_dict.l1_end_point_pos = [0.0, 0.0, ind[0]]
    CFG.MorphConfig.robot_description_dict.l2_end_point_pos = [0.0, 0.0, ind[1]]
    CFG.MorphConfig.robot_description_dict.l3_end_point_pos = [0.0, 0.0, ind[2]]
    CFG.MorphConfig.robot_description_dict.l4_end_point_pos = [0.0, 0.0, ind[3]]
    CFG.MorphConfig.robot_description_dict.l5_end_point_pos = [0.0, 0.0, ind[4]]
    CFG.MorphConfig.robot_description_dict.l6_end_point_pos = [0.0, 0.0, ind[5]]
    omega_conf = OmegaConf.create(CFG)
    # Initialize the OMPL planner
    planner = LynxPlanner(omega_conf)

    goal = np.random.uniform(0, 1, 3)
    _, error = planner.run(goal=goal, visualize=False)

    return error


def evaluate_pop(population: Population) -> Population:
    to_eval = [ind for ind in population if ind.requires_eval]
    if not to_eval:
        return population

    futures = [evaluate_ind.remote(ind.genotype) for ind in to_eval]

    results = ray.get(futures)
    for ind, fitness in zip(to_eval, results, strict=True):
        ind.fitness = fitness

    return population

def main(pop_size: int, num_gens: int) -> EA:
    """Entry point."""
    # Create initial population
    population_list = [create_individual(num_dims=6) for _ in range(pop_size)]
    population_list = evaluate_pop(population_list)

    # Create EA steps
    ops = [
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutation", mutation),
        EAStep("evaluation", evaluate_pop),
        EAStep("survivor_selection", survivor_selection),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=num_gens,
    )

    ea.run()

    best = ea.get_solution("best", only_alive=False)
    console.log(best)

    median = ea.get_solution("median", only_alive=False)
    console.log(median)

    worst = ea.get_solution("worst", only_alive=False)
    console.log(worst)

    console.log(ea.target_population_size)

    return ea


if __name__ == "__main__":
    ray.init()
    main(pop_size=5,
         num_gens=5,
         )
    # Tidy up Ray workers when you are done (optional)
    ray.shutdown()
