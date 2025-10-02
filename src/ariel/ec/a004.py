"""TODO(jmdm): description of script.

Notes
-----
    * Do we consider survivors to be of the new generation?
"""

# Standard library
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.progress import track
from rich.traceback import install
from sqlalchemy import create_engine
from sqlmodel import Session, SQLModel, col, select

# Local libraries
from ariel.ec.a000 import IntegerMutator, IntegersGenerator
from ariel.ec.a001 import Individual
from ariel.ec.a005 import Crossover

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
        """
        _summary_

        Parameters
        ----------
        db_file_path : str | Path | None, optional
            _description_, by default None
        db_handling : DB_HANDLING_MODES | None, optional
            _description_, by default None

        Raises
        ------
        FileExistsError
            _description_
        """
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


@dataclass
class EAStep:
    name: str
    operation: Callable[[Population], Population]

    def __call__(self, population: Population) -> Population:
        return self.operation(population)


class EA(AbstractEA):
    def __init__(
        self,
        population: Population,
        operations: list[EAStep],
        num_of_generations: int | None = None,
        *,
        first_generation_id: int | None = None,
        quiet: bool | None = None,
    ) -> None:
        """
        Initialize an Evolutionary Algorithm (EA) instance.

        Parameters
        ----------
        population : Population
            Initial population.
        operations : list[EAStep]
            List of operations to be performed in each generation.
        num_of_generations : int | None, optional
            Number of generations to run the EA for, by default None.
            If None, the value from the global config is used.
        first_generation_id : int | None, optional
            ID of the first generation, by default None.
            If None, the value from the global config is used.
        quiet : bool | None, optional
            Whether to suppress console output, by default None.
            If None, the value from the global config is used.
        """
        # Local parameters
        self.operations = operations

        # Flexible global parameters
        self.quiet = quiet or config.quiet
        self.console = Console(quiet=self.quiet)
        self.current_generation = (
            first_generation_id or config.first_generation_id
        )
        self.num_of_generations = (
            num_of_generations or config.num_of_generations
        )

        # Bound to global parameters
        self.is_maximisation = config.is_maximisation
        self.target_population_size = config.target_population_size

        # Initialisation
        self.init_database()
        self.population = population
        self.commit_population()
        self.console.rule("[blue]EA Initialised")

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
        if only_alive is True:
            statement = statement.where(
                (Individual.alive),
            )
        if already_evaluated is not None:
            statement = statement.where(
                (Individual.requires_eval != already_evaluated),
            )
        if custom_logic is not None:
            statement.where(
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
            self.population = list(session.exec(statement).all())

    def get_solution(
        self,
        mode: Literal["best", "median", "worst"] = "best",
        *,
        only_alive: bool = True,
    ) -> Individual:
        # Query population
        self.fetch_population(
            only_alive=only_alive,
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

    def step(self) -> None:
        self.current_generation += 1
        self.fetch_population()
        for operation in self.operations:
            self.population = operation(self.population)
        self.commit_population()

    def run(self) -> None:
        for _ in track(
            range(self.num_of_generations),
            description="Running EA:",
        ):
            self.step()
        self.console.rule("[green]EA Finished Running")


# ------------------------ EA STEPS ------------------------ #
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


def evaluate(population: Population) -> Population:
    for ind in population:
        if ind.requires_eval:
            # Count ones in genotype as fitness
            ind.fitness = sum(1 for gene in ind.genotype if gene == 1)
    return population


def survivor_selection(population: Population) -> Population:
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


def create_individual() -> Individual:
    ind = Individual()
    ind.genotype = IntegersGenerator.integers(low=0, high=10, size=5)
    return ind


def main() -> None:
    """Entry point."""
    # Create initial population
    population_list = [create_individual() for _ in range(10)]
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
