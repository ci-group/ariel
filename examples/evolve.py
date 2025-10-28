# Standard library
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Literal, cast
from argparse import ArgumentParser

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, col, select

# Local libraries
from ariel.ec.a000 import IntegerMutator, IntegersGenerator, Mutation, TreeMutator
from ariel.ec.a001 import Individual
from ariel.ec.a004 import EAStep, EA
from ariel.ec.a005 import Crossover, IntegerCrossover, TreeCrossover
from ariel.ec.genotypes.genotype import GenotypeEnum

# Global constants
SEED = 42
DB_HANDLING_MODES = Literal["delete", "halt"]

# Global functions
install()
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases
type Population = list[Individual]
type PopulationFunc = Callable[[Population], Population]

class EASettings(BaseSettings):
    quiet: bool = False

    # EC mechanisms
    is_maximisation: bool = True
    first_generation_id: int = 0
    num_of_generations: int = 100
    target_population_size: int = 100
    genotype: GenotypeEnum
    mutation: Mutation
    crossover: Crossover

    # Data config
    output_folder: Path = Path.cwd() / "__data__"
    db_file_name: str = "database.db"
    db_file_path: Path = output_folder / db_file_name
    db_handling: DB_HANDLING_MODES = "delete" 

# ------------------------ EA STEPS ------------------------ #
def parent_selection(population: Population, config: EASettings) -> Population:
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


def crossover(population: Population, config: EASettings) -> Population:
    parents = [ind for ind in population if ind.tags.get("ps", False)]
    for idx in range(0, len(parents), 2):
        parent_i = parents[idx]
        parent_j = parents[idx]
        genotype_i, genotype_j = config.crossover(
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

        population.extend([child_i, child_j])
    return population


def mutation(population: Population, config: EASettings) -> Population:
    for ind in population:
        if ind.tags.get("mut", False):
            genes = ind.genotype
            mutated = config.mutation(
                individual=genes,
                span=1,
                mutation_probability=0.5,
            )
            ind.genotype = mutated
            ind.requires_eval = True
    return population


def evaluate(population: Population) -> Population:
    pass


def survivor_selection(population: Population, config: EASettings) -> Population:
    random.shuffle(population)
    current_pop_size = len(population)
    for idx in range(len(population)):
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


def create_individual(config: EASettings) -> Individual:
    ind = Individual()
    ind.genotype = config.genotype.value.create_individual()
    return ind

def read_config_file() -> EASettings:
    cfg = tomllib.loads(Path("config.toml").read_text())

    # Resolve the active operators from the chosen genotype profile
    gname = cfg["run"]["genotype"]
    gblock = cfg["genotypes"][gname]
    mutation_name = cfg["run"].get("mutation", gblock["defaults"]["mutation"])
    crossover_name = cfg["run"].get("crossover", gblock["defaults"]["crossover"])

    if gname == 'tree':
        genotype = GenotypeEnum.TREE
        mutation = TreeMutator()
        mutation.which_mutation = mutation_name
        crossover = TreeCrossover()
        crossover.which_crossover = crossover_name
    elif gname == 'integers':
        pass

    settings = EASettings(
        quiet=cfg["ec"]["quiet"],
        is_maximisation=cfg["ec"]["is_maximisation"],
        first_generation_id=cfg["ec"]["first_generation_id"],
        num_of_generations=cfg["ec"]["num_of_generations"],
        target_population_size=cfg["ec"]["target_population_size"],
        genotype=genotype,
        mutation=mutation,
        crossover=crossover,
        output_folder=Path(cfg["data"]["output_folder"]),
        db_file_name=cfg["data"]["db_file_name"],
        db_handling=cfg["data"]["db_handling"],
        db_file_path=Path(cfg["data"]["output_folder"]) / cfg["data"]["db_file_name"],
    )
    return settings
    

def main() -> None:
    """Entry point."""
    config = read_config_file()
    # Create initial population
    population_list = [create_individual(config) for _ in range(10)]
    population_list = evaluate(population_list)

    # Create EA steps
    ops = [
        EAStep("parent_selection", parent_selection),
        EAStep("crossover", crossover),
        EAStep("mutation", mutation),
        EAStep("evaluation", evaluate),
        EAStep("survivor_selection", survivor_selection),
    ]

    # Initialize EA
    ea = EA(
        population_list,
        operations=ops,
        num_of_generations=100,
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