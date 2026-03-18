"""EA class and EAStep."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import computed_field
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from rich.traceback import install
from sqlalchemy import Engine, create_engine
from sqlmodel import Session, SQLModel, col, select

from ariel.ec.individual import Individual
from ariel.ec.population import Population

install()

type DBHandlingMode = Literal["delete", "halt"]


# ── Settings ──────────────────────────────────────────────────────────────────


class EASettings(BaseSettings):
    """
    EASettings class to act as the config of the EA class.

    ```python
    config = EASettings()
    ```

    Parameters
    ----------
    quiet: bool = False
        Weather you want printing each time a generation is completed
    is_maximisation: bool = True
        Weather the task is maximisation of minimisation
    first_generation_id: int = 0
        What the id of the first generation should be
    num_of_steps: int = 100
        Number of steps the EA will run for
    target_population_size: int = 100
        Target population for the EA. Can be used for population size management
    output_folder: Path = Path.cwd() / "__data__"
        Output folder to save results and the database
    db_file_name: str = "database.db"
        The name of the database file
        db_handling: DBHandlingMode | None, optional
            How the database will be handled. Default option is "delete", can be
            changed to "halt". Only works if database name is the same.
            Database name is inherited from EASettings.
            - "delete" will delete a database with the same name if it exists
            - "halt" will stop the code and will not touch other databases
    """

    quiet: bool = False
    is_maximisation: bool = True
    first_generation_id: int = 0
    num_steps: int = 100
    target_population_size: int = 100
    output_folder: Path = Path.cwd() / "__data__"
    db_file_name: str = "database.db"
    db_handling: DBHandlingMode = "delete"

    @computed_field
    @property
    def db_file_path(self) -> Path:
        return self.output_folder / self.db_file_name


config: EASettings = EASettings()


# ── Step wrapper ──────────────────────────────────────────────────────────────
@dataclass
class EAStep:
    """A named, callable wrapper around a single EA pipeline operation."""

    name: str
    operation: Callable[[Population], Population]

    def __call__(self, population: Population) -> Population:
        """Run the operation given to EAStep.

        Parameters
        ----------
        population: Population

        Returns
        -------
        Population
        """
        return self.operation(population)


# ── EA ────────────────────────────────────────────────────────────────────────


class EA:
    """
    Generational evolutionary algorithm engine.

    Each call to ``step()`` / ``run()`` executes the ordered list of
    ``EAStep`` operations against the current population, then persists
    the result to a SQLite database via SQLModel.

    Quick-start
    -----------
    ::

        ops = [
            EAStep("evaluate", evaluate),
            EAStep("survivor_selection", survivor_selection),
        ]
        ea = EA(Population(initial_individuals), ops, num_of_generations=200)
        ea.run()
        winner = ea.best()
    """

    def __init__(
        self,
        population: Population,
        operations: list[EAStep],
        *,
        num_steps: int | None = None,
        first_generation_id: int | None = None,
        quiet: bool | None = None,
        db_file_path: Path | None = None,
        db_handling: DBHandlingMode | None = None,
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
        db_file_path: Path | None, optional
            The file path where the database will be saved and pulled from
        db_handling: DBHandlingMode | None, optional
            How the database will be handled. Default option is "delete", can be
            changed to "halt". Only works if database name is the same.
            Database name is inherited from EASettings.
            - "delete" will delete a database with the same name if it exists
            - "halt" will stop the code and will not touch other databases
        """
        self.operations: list[EAStep] = operations
        self.is_maximisation: bool = config.is_maximisation
        self.target_population_size: int = config.target_population_size
        self.current_generation: int = (
            first_generation_id
            if first_generation_id is not None
            else config.first_generation_id
        )
        self.num_steps: int = (
            num_steps
            if num_steps is not None
            else config.num_steps
        )
        self.console: Console = Console(
            quiet=quiet if quiet is not None else config.quiet
        )
        self.engine: Engine = self._setup_database(
            db_file_path=db_file_path or config.db_file_path,
            db_handling=db_handling or config.db_handling,
        )
        self.population: Population = population
        self._commit()
        self.console.rule("[blue]EA Initialized")

    # ── Database ──────────────────────────────────────────────────────────────

    def _setup_database(
        self, db_file_path: Path,
        db_handling: DBHandlingMode,
    ) -> Engine:
        db_file_path.parent.mkdir(parents=True, exist_ok=True)

        if db_file_path.exists():
            msg = f"Database at {db_file_path!r} — handling: {db_handling!r}"
            match db_handling:
                case "delete":
                    self.console.log(f"⚠️  {msg} → deleting", style="yellow")
                    db_file_path.unlink()
                case "halt":
                    raise FileExistsError(f"⚠️  {msg} → halted")

        engine = create_engine(f"sqlite:///{db_file_path}")
        SQLModel.metadata.create_all(engine)
        return engine

    def _commit(self) -> None:
        with Session(self.engine) as session:
            for ind in self.population:
                if ind.time_of_birth == -1:
                    ind.time_of_birth = self.current_generation
                ind.time_of_death = self.current_generation
                session.add(ind)
            session.commit()

    # ── Query helpers ─────────────────────────────────────────────────────────

    def _fetch(
        self,
        *,
        sort: Literal["asc", "desc"] | None = None,
        only_alive: bool = True,
        requires_eval: bool | None = None,
    ) -> Population:

        statement = select(Individual)

        if only_alive:
            statement = statement.where(Individual.alive)
        if requires_eval is not None:
            statement = statement.where(
                Individual.requires_eval == requires_eval
            )

        match sort:
            case "desc":
                statement = statement.order_by(col(Individual.fitness_).desc())
            case "asc":
                statement = statement.order_by(col(Individual.fitness_).asc())

        with Session(self.engine) as session:
            return Population(list(session.exec(statement).all()))

    def fetch_population(
        self,
        *,
        only_alive: bool = True,
        requires_eval: bool | None = None,
    ) -> None:
        self.population = self._fetch(
            only_alive=only_alive, requires_eval=requires_eval,
        )

    # ── Population stats ──────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        self.fetch_population()
        return self.population.size

    def best(
        self,
        mode: Literal["best", "median", "worst"] = "best",
        *,
        only_alive: bool = True,
    ) -> Individual:
        sort: Literal["asc", "desc"] = "desc" if self.is_maximisation else "asc"
        pop = self._fetch(sort=sort, only_alive=only_alive, requires_eval=False)

        match mode:
            case "best":
                return pop[0]
            case "median":
                return pop[len(pop) // 2]
            case "worst":
                return pop[-1]

    # ── Execution ─────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Perform one cycle of the EAStep Operations."""
        self.current_generation += 1
        self.fetch_population()
        for op in self.operations:
            self.population = op(self.population)
        self._commit()

    def run(self) -> None:
        """Run the EA for num_steps."""
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task(
                "Running EA", total=self.num_steps,
            )
            for _ in range(self.num_steps):
                self.step()
                progress.advance(task)

        self.console.rule("[green]EA Finished")
