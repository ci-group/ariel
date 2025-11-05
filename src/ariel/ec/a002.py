"""TODO(jmdm): description of script."""

# Standard library
import random
from collections.abc import Sequence
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install
from sqlalchemy import Engine
from sqlmodel import Session, select

# Local libraries
from ariel.src.ariel.ec.mutations import IntegersGenerator
from ariel.ec.a001 import Individual, init_database

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)
DB_NAME = "database.db"
DB_PATH: Path = DATA / DB_NAME
SEED = 42

# Global functions
install(width=180)
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases
type Population = Sequence[Individual]


def fetch_population(
    engine: Engine,
    *,
    logic: tuple[bool, ...] | None = None,
    constrain_alive: bool = True,
) -> Population:
    statement = select(Individual)
    if constrain_alive:
        statement = statement.where(
            (Individual.alive),
        )
    if logic:
        statement = statement.where(
            *logic,
        )

    # Execute query
    with Session(engine) as session:
        return session.exec(statement).all()


def commit_population(population: Population, engine: Engine) -> None:
    with Session(engine) as session:
        session.add_all(population)
        session.commit()


def kill_members(population: Population) -> Population:
    for ind in population:
        if random.random() < 0.9:
            ind.alive = False
    return population


def evaluate(population: Population) -> Population:
    for ind in population:
        ind.fitness = random.random()
    return population


def create_individual() -> Individual:
    ind = Individual()
    ind.genotype = IntegersGenerator.integers(low=0, high=10, size=5)
    return ind


def test_fetch_logic() -> None:
    console.rule("[bold red]Fetch Logic")

    # Initialize the database
    engine = init_database()

    # Create initial population
    population_list = [create_individual() for _ in range(10_000)]

    # Evaluate and save
    start = console.get_time()
    evaluate(population_list[: len(population_list) // 2])
    kill_members(population_list)
    commit_population(population_list, engine)
    end = console.get_time()
    console.log(
        f"Initial commit: {end - start:.5f} sec ({len(population_list)})",
    )

    # Fetch population
    start = console.get_time()
    population = fetch_population(engine)
    end = console.get_time()
    console.log(f"Fetch: {end - start:.5f} sec ({len(population)})")

    # Fetch population (requires_eval and alive)
    start = console.get_time()
    population = fetch_population(
        engine,
        logic=((Individual.requires_eval),),
    )
    end = console.get_time()
    console.log(
        f"Fetch by Requires Eval: {end - start:.5f} sec ({len(population)})",
    )

    # Fetch population (fitness)
    start = console.get_time()
    threshold = 0.5
    population = fetch_population(
        engine,
        logic=(
            (Individual.fitness_ >= threshold),  # type: ignore[operator]
        ),
    )
    end = console.get_time()
    console.log(
        f"Fetch by Fitness: {end - start:.5f} sec ({len(population)})",
    )

    # Fetch population (fitness, unconstrained)
    start = console.get_time()
    threshold = 0.5
    population = fetch_population(
        engine,
        logic=(
            (Individual.fitness_ >= threshold),  # type: ignore[operator]
        ),
        constrain_alive=False,
    )
    end = console.get_time()
    console.log(
        f"Fetch by Fitness (unconstrained): {end - start:.5f}sec \
            ({len(population)})",
    )


def test_evaluate_mechanics() -> None:
    console.rule("[bold red]Evaluate Mechanics")

    # Initialize the database
    engine = init_database()

    # Create initial population
    population_list = [create_individual() for _ in range(10_000)]

    # Save data
    start = console.get_time()
    commit_population(population_list, engine)
    end = console.get_time()
    console.log(
        f"Initial commit: {end - start:.5f} sec ({len(population_list)})",
    )

    # Fetch population
    start = console.get_time()
    population = fetch_population(engine)
    end = console.get_time()
    console.log(f"Fetch: {end - start:.5f} sec ({len(population)})")

    # Evaluate population
    start = console.get_time()
    evaluate(population)
    end = console.get_time()
    console.log(
        f"Evaluating members: {end - start:.5f} sec ({len(population)})",
    )

    # Commit population
    start = console.get_time()
    commit_population(population, engine)
    end = console.get_time()
    console.log(
        f"Committing changes: {end - start:.5f} sec ({len(population)})",
    )


def test_kill_mechanics() -> None:
    console.rule("[bold red]Kill Mechanics")

    # Initialize the database
    engine = init_database()

    # Create initial population
    population_list = [create_individual() for _ in range(10_000)]

    # Save data
    start = console.get_time()
    commit_population(population_list, engine)
    end = console.get_time()
    console.log(
        f"Initial commit: {end - start:.5f} sec ({len(population_list)})",
    )

    # Fetch, kill, and commit population
    start = console.get_time()
    population = fetch_population(engine)
    kill_members(population)
    end = console.get_time()
    console.log(f"Killing members: {end - start:.5f} sec ({len(population)})")

    start = console.get_time()
    commit_population(population, engine)
    end = console.get_time()
    console.log(
        f"Committing changes: {end - start:.5f} sec ({len(population)})",
    )

    # Test how much faster fetching only alive individuals is
    start = console.get_time()
    population = fetch_population(engine)
    end = console.get_time()
    console.log(f"Fetch only alive: {end - start:.5f} sec ({len(population)})")


def main() -> None:
    """Entry point."""
    # --------------------- Test Kill Mechanics ---------------------
    test_kill_mechanics()

    # --------------------- Test Evaluate Mechanics ---------------------
    test_evaluate_mechanics()

    # --------------------- Test Fetch Logic ---------------------
    test_fetch_logic()


if __name__ == "__main__":
    main()
