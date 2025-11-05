"""TODO(jmdm): description of script.

Notes
-----
    * Do we consider survivors to be of the new generation?
"""

# Standard library
import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.traceback import install
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, col, select

# Local libraries
from ariel.ec.a001 import Individual

# Third-party libraries
# Global constants
SEED = 42
DB_HANDLING_MODES = Literal["delete", "halt"]

# Global functions
install(width=180)
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases
type Population = Sequence[Individual]
type PopulationFunc = Callable[[Population], Population]


class EASettings(BaseSettings):
    quiet: bool = False

    # EC mechanisms
    survivors_are_new_generation: bool = True
    is_maximisation: bool = True
    first_generation_id: int = 0
    num_of_generations: int = 100

    # Data config
    output_folder: Path = Path.cwd() / "__data__"
    db_file_name: str = "database.db"
    db_file_path: Path = output_folder / db_file_name
    db_handling: DB_HANDLING_MODES = "delete"


config = EASettings()


class AbstractEA:
    def init_database(
        self,
        db_file_path: str | Path | None = None,
        db_handling: DB_HANDLING_MODES | None = None,
    ) -> None:
        db_file_path = db_file_path or config.db_file_path
        db_handling = db_handling or config.db_handling

        # If file path is a string convert to Pathlib
        if isinstance(db_file_path, str):
            db_file_path = Path(db_file_path)

        # How to handle an existing database file
        db_exists = db_file_path.exists()
        if db_exists:
            msg = f"Database file exists at {db_file_path}!\n"
            msg += f"Behaviour is set to: '{db_handling}' --> "
            match db_handling:
                case "delete":
                    msg += "⚠️  Deleting file!"
                    console.log(msg, style="yellow")
                    db_file_path.unlink()
                case "halt":
                    msg += "⚠️  Halting execution!"
                    raise FileExistsError(msg)

        # Create the database engine
        self.engine = create_engine(
            f"sqlite:///{db_file_path}",
        )
        SQLModel.metadata.create_all(self.engine)


class BasicEA(AbstractEA):
    def __init__(
        self,
        population: Population,
        evaluation_function: Callable[[Population], None],
        survivor_selection: Callable[[Population], None],
        num_of_generations: int | None = None,
        *,
        survivors_are_new_generation: bool | None = None,
        is_maximisation: bool | None = None,
        first_generation_id: int | None = None,
        quiet: bool | None = None,
    ) -> None:
        # Local parameters
        self.evaluation_func = evaluation_function
        self.survivor_func = survivor_selection

        # Global parameters
        self.new_generation_are_survivors = (
            survivors_are_new_generation or config.survivors_are_new_generation
        )
        self.is_maximisation = is_maximisation or config.is_maximisation
        self.console = Console(quiet=quiet or config.quiet)
        self.current_generation = (
            first_generation_id or config.first_generation_id
        )
        self.num_of_generations = (
            num_of_generations or config.num_of_generations
        )

        # Initialisation
        self.init_database()
        self.population = population
        self.commit_population()
        self.console.rule("EA Initialised")

    @property
    def population_size(self) -> int:
        self.fetch_population()
        return len(self.population)

    def commit_population(
        self,
    ) -> None:
        with Session(self.engine) as session:
            for ind in self.population:
                # Update 'time_of_birth'
                if ind.time_of_birth == -1:
                    ind.time_of_birth = self.current_generation
                # Update 'time_of_death'
                ind.time_of_death = self.current_generation
                session.add(ind)
            session.commit()

    def fetch_population(
        self,
        best_comes: Literal["first", "last"] | None = "first",
        *,
        only_alive: bool = True,
        already_evaluated: bool | None = None,
        custom_logic: tuple[bool, ...] | None = None,
    ) -> None:
        # Build query
        statement = select(Individual)

        # Filters
        if only_alive is not True:
            statement = statement.where(
                (Individual.alive),
            )
        if already_evaluated is not None:
            statement = statement.where(
                (Individual.requires_eval != already_evaluated),
            )
        if custom_logic is not None:
            statement = statement.where(
                *custom_logic,
            )

        # Sorting
        if best_comes is not None:
            # Descending
            max_and_first = self.is_maximisation & (best_comes == "first")
            min_and_last = (not self.is_maximisation) & (best_comes == "last")

            # Ascending
            max_and_last = self.is_maximisation & (best_comes == "last")
            min_and_first = (not self.is_maximisation) & (best_comes == "first")

            # Produce sorting statement
            if max_and_first | min_and_last:
                statement = statement.order_by(col(Individual.fitness_).desc())
            elif max_and_last | min_and_first:
                statement = statement.order_by(col(Individual.fitness_).asc())
            else:
                msg = "Correct sorting statement not found!\n"
                msg += f"Got: {best_comes=} and {self.is_maximisation=}\n"
                msg += f"Is {best_comes=} a valid sorting option? "
                raise ValueError(msg)

        # Execute query
        with Session(self.engine) as session:
            self.population = session.exec(statement).all()

    def evaluate(
        self,
    ) -> None:
        # Only fetch un-evaluated individuals
        self.fetch_population(already_evaluated=False)

        # If fetch returns an empty population, something went wrong
        if len(self.population) == 0:
            msg = "No individuals found eligible for evaluation!"
            msg += " Did you forget to create individuals?"
            console.log(msg, style="red")

        # Apply evaluation function
        self.evaluation_func(self.population)
        self.commit_population()

    def survivor_selection(self) -> None:
        # Only fetch evaluated individuals
        self.fetch_population(already_evaluated=True)

        # If fetch returns an empty population, something went wrong
        if len(self.population) == 0:
            msg = "No individuals found eligible for survivor selection!"
            msg += " Did you forget to evaluate the population?"
            raise ValueError(msg)

        # Apply survivor selection
        self.survivor_func(self.population)
        self.commit_population()

    def get_solution(
        self,
        mode: Literal["best", "median", "worst"] = "best",
        *,
        constrain_alive: bool = True,
    ) -> Individual:
        # Query population
        self.fetch_population(
            only_alive=constrain_alive,
            already_evaluated=True,
            best_comes="first",  # self.population[0]
        )

        # Get requested individual
        match mode:
            case "best":
                return self.population[0]
            case "median":
                return self.population[len(self.population) // 2]
            case "worst":
                return self.population[-1]


def create_individual() -> Individual:
    ind = Individual()
    ind.genotype = IntegersGenerator.integers(low=0, high=10, size=5)
    return ind


def evaluate_func(population: Population) -> None:
    for ind in population:
        ind.fitness = random.random()


def selection_func(population: Population) -> None:
    for ind in population:
        threshold = 0.5
        if ind.fitness < threshold:
            ind.alive = False


def main() -> None:
    """Entry point."""
    # Create initial population
    population_list = [create_individual() for _ in range(100)]

    # Initialize EA
    ea = BasicEA(
        population_list,
        evaluation_function=evaluate_func,
        survivor_selection=selection_func,
    )

    console.log(f"{ea.population_size=}")
    ea.evaluate()
    console.log(f"{ea.population_size=}")
    ea.survivor_selection()
    console.log(f"{ea.population_size=}")

    best = ea.get_solution()
    console.log(f"{best.fitness=}")

    median = ea.get_solution("median")
    console.log(f"{median.fitness=}")

    worst = ea.get_solution("worst")
    console.log(f"{worst.fitness=}")


if __name__ == "__main__":
    main()
