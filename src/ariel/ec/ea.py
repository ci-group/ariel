"""EA class and EAStep."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

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


# -- Settings ------------------------------------------------------------------


class EASettings(BaseSettings):
    """
    Global configuration for the EA engine.

    Consumed by both ``EA`` and ``EAStep``. Values can be overridden via
    environment variables or a ``.env`` file thanks to ``pydantic-settings``.

    Parameters
    ----------
    quiet : bool, optional
        Suppress Rich console output during the run. Default is ``False``.
    is_maximisation : bool, optional
        ``True`` for maximisation problems, ``False`` for minimisation.
        Default is ``True``.
    first_generation_id : int, optional
        Generation counter value assigned to the initial population.
        Default is ``0``.
    num_steps : int, optional
        Total number of generational steps ``EA.run`` will execute.
        Default is ``100``.
    target_population_size : int, optional
        Soft target used by survivor-selection operators to decide how many
        individuals to retain each generation. Default is ``100``.
    output_folder : Path, optional
        Directory under which the database file is created. Created
        automatically if it does not exist. Default is
        ``Path.cwd() / "__data__"``.
    db_file_name : str, optional
        File name of the SQLite database. Default is ``"database.db"``.
    db_handling : DBHandlingMode, optional
        Behaviour when a database with the same path already exists.

        ``"delete"``
            Delete the existing file and start fresh.
        ``"halt"``
            Raise ``FileExistsError`` without modifying anything.

        Default is ``"delete"``.
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
        """Absolute path to the SQLite database file.

        Returns
        -------
        Path
            ``output_folder / db_file_name``.
        """
        return self.output_folder / self.db_file_name


config: EASettings = EASettings()


# -- Step wrapper --------------------------------------------------------------
####################################################
#                   OLD VERSION                    #
####################################################

@dataclass
class EAStep:
    """A named, callable wrapper around a single EA pipeline operation.

    Parameters
    ----------
    name : str or None
        Human-readable label for this step. When ``None``, the name is
        inferred from ``operation.__name__`` in ``__post_init__``.
    operation : Callable[[Population], Population]
        The pipeline function to invoke. Must accept a ``Population`` and
        return a ``Population``.
    """

    name: str | None
    operation: Callable[[Population], Population]

    def __post_init__(self) -> None:
        """Derive ``name`` from the wrapped function when not provided."""
        if self.name is None:
            self.name = self.operation.__name__

    def __call__(self, population: Population) -> Population:
        """Execute the wrapped operation on the given population.

        Parameters
        ----------
        population : Population
            The current population to transform.

        Returns
        -------
        Population
            The population returned by the wrapped operation.
        """
        return self.operation(population)

####################################################
#                   NEW VERSION 1                  #
####################################################


def mutate(population: Population,
           probability: float = 0.5,
           scale: float = 2,
           ) -> Population:
    """Apply mutation to the population.

    Parameters
    ----------
    population : Population
        The population whose individuals will be mutated.
    probability : float, optional
        Per-gene mutation probability. Must be in ``[0.0, 1.0]``.
        Default is ``0.5``.
    scale : float, optional
        Magnitude scaling factor applied to each mutation step.
        Default is ``2``.

    Returns
    -------
    Population
        The population after mutation has been applied.
    """
    return population


ops = [
    EAStep("mutate", partial(mutate, probability=0.5, scale=2)),
]

####################################################
#                   NEW VERSION 2                  #
####################################################


class EAStep:
    """A callable wrapper around a single EA pipeline operation.

    Keyword arguments passed at construction time are forwarded to the
    wrapped operation on every call, allowing step-specific parameters to
    be bound once at the point of pipeline definition rather than hard-coded
    inside each operator function.

    Parameters
    ----------
    operation : Callable[[Population], Population]
        The pipeline function to invoke. Must accept a ``Population`` as its
        first positional argument and return a ``Population``.
    name : str or None, optional
        Human-readable label for this step. Defaults to
        ``operation.__name__`` when ``None``. Default is ``None``.
    **kwargs : Any
        Additional keyword arguments forwarded to ``operation`` on every
        ``__call__``.

    Examples
    --------
    >>> EAStep(mutate, probability=0.2, scale=1)
    >>> EAStep(parent_selection, tournament_size=5)
    """

    def __init__(
        self,
        operation: Callable[[Population], Population],
        *,
        name: str | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        self.name = name or operation.__name__
        self.operation = operation
        self.kwargs = kwargs

    def __post_init__(self) -> None:
        """Derive ``name`` from the wrapped function when not provided."""
        if self.name is None:
            self.name = self.operation.__name__

    def __call__(self, population: Population) -> Population:
        """Execute the wrapped operation with the bound keyword arguments.

        Parameters
        ----------
        population : Population
            The current population to transform.

        Returns
        -------
        Population
            The population returned by the wrapped operation.
        """
        return self.operation(population, **self.kwargs)


# -- EA ------------------------------------------------------------------------


class EA:
    """
    Generational evolutionary algorithm engine.

    Each call to ``step()`` / ``run()`` executes the ordered list of
    ``EAStep`` operations against the current population, then persists
    the result to a SQLite database via SQLModel.

    Parameters
    ----------
    population : Population
        Initial population. Committed to the database on construction.
    operations : list[EAStep]
        Ordered pipeline of operations executed once per generation.
    num_steps : int or None, optional
        Total number of generational steps to run. Falls back to
        ``config.num_steps`` when ``None``. Default is ``None``.
    first_generation_id : int or None, optional
        Starting value of the internal generation counter. Falls back to
        ``config.first_generation_id`` when ``None``. Default is ``None``.
    quiet : bool or None, optional
        Suppress Rich console output. Falls back to ``config.quiet`` when
        ``None``. Default is ``None``.
    db_file_path : Path or None, optional
        Path at which the SQLite database is created. Falls back to
        ``config.db_file_path`` when ``None``. Default is ``None``.
    db_handling : DBHandlingMode or None, optional
        Behaviour when a database already exists at ``db_file_path``. Falls
        back to ``config.db_handling`` when ``None``. Default is ``None``.

    Examples
    --------
    >>> ops = [
    ...     EAStep(evaluate),
    ...     EAStep(survivor_selection),
    ... ]
    >>> ea = EA(Population(initial_individuals), ops, num_steps=200)
    >>> ea.run()
    >>> winner = ea.best()
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

    # -- Database --------------------------------------------------------------

    def _setup_database(
        self,
        db_file_path: Path,
        db_handling: DBHandlingMode,
    ) -> Engine:
        """Create (or clear) the SQLite database and return its engine.

        The parent directory is created if it does not already exist. When a
        database file is already present at ``db_file_path``, behaviour is
        controlled by ``db_handling``.

        Parameters
        ----------
        db_file_path : Path
            Destination path for the SQLite database file.
        db_handling : DBHandlingMode
            ``"delete"`` removes any existing file before creating a fresh
            database. ``"halt"`` raises ``FileExistsError`` immediately.

        Returns
        -------
        Engine
            A SQLAlchemy engine connected to the database, with all
            SQLModel tables created.

        Raises
        ------
        FileExistsError
            If a file already exists at ``db_file_path`` and
            ``db_handling`` is ``"halt"``.
        """
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
        """Persist the current population snapshot to the database.

        For each individual in ``self.population``:

        - Sets ``time_of_birth`` to ``current_generation`` if the individual
          has not yet been assigned a birth time (i.e. ``time_of_birth == -1``).
        - Updates ``time_of_death`` to ``current_generation``, reflecting the
          last generation in which the individual was present.
        """
        with Session(self.engine) as session:
            for ind in self.population:
                if ind.time_of_birth == -1:
                    ind.time_of_birth = self.current_generation
                ind.time_of_death = self.current_generation
                session.add(ind)
            session.commit()

    # -- Query helpers ---------------------------------------------------------

    def _fetch(
        self,
        *,
        sort: Literal["asc", "desc"] | None = None,
        only_alive: bool = True,
        requires_eval: bool | None = None,
    ) -> Population:
        """Query individuals from the database and return them as a Population.

        Parameters
        ----------
        sort : {"asc", "desc"} or None, optional
            Sort order by ``fitness_``. ``"desc"`` returns highest fitness
            first; ``"asc"`` returns lowest fitness first. When ``None`` the
            database default ordering is used. Default is ``None``.
        only_alive : bool, optional
            When ``True``, only individuals where ``alive`` is set are
            returned. Default is ``True``.
        requires_eval : bool or None, optional
            Filter by evaluation status. ``True`` returns only unevaluated
            individuals; ``False`` returns only evaluated ones. When ``None``
            no filter is applied. Default is ``None``.

        Returns
        -------
        Population
            A new ``Population`` containing all matching individuals.
        """
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
        """Refresh ``self.population`` from the database.

        Parameters
        ----------
        only_alive : bool, optional
            When ``True``, only individuals where ``alive`` is set are
            fetched. Default is ``True``.
        requires_eval : bool or None, optional
            Filter by evaluation status. ``True`` returns only unevaluated
            individuals; ``False`` returns only evaluated ones. When ``None``
            no filter is applied. Default is ``None``.
        """
        self.population = self._fetch(
            only_alive=only_alive, requires_eval=requires_eval,
        )

    # -- Population stats ------------------------------------------------------

    @property
    def size(self) -> int:
        """Current number of alive individuals in the population.

        Triggers a database fetch on every access.

        Returns
        -------
        int
            Number of alive individuals.
        """
        self.fetch_population()
        return self.population.size

    def best(
        self,
        mode: Literal["best", "median", "worst"] = "best",
        *,
        only_alive: bool = True,
    ) -> Individual:
        """Return a single representative individual by fitness rank.

        The sort direction is determined by ``self.is_maximisation``:
        descending for maximisation problems, ascending for minimisation.
        Only evaluated individuals (``requires_eval=False``) are considered.

        Parameters
        ----------
        mode : {"best", "median", "worst"}, optional
            Which individual to return from the sorted population.

            ``"best"``
                Highest fitness for maximisation; lowest for minimisation.
            ``"median"``
                The individual at the middle index of the sorted population.
            ``"worst"``
                Lowest fitness for maximisation; highest for minimisation.

            Default is ``"best"``.
        only_alive : bool, optional
            When ``True``, restricts the query to alive individuals.
            Default is ``True``.

        Returns
        -------
        Individual
            The selected individual.
        """
        sort: Literal["asc", "desc"] = "desc" if self.is_maximisation else "asc"
        pop = self._fetch(sort=sort, only_alive=only_alive, requires_eval=False)

        match mode:
            case "best":
                return pop[0]
            case "median":
                return pop[len(pop) // 2]
            case "worst":
                return pop[-1]

    # -- Execution -------------------------------------------------------------

    def step(self) -> None:
        """Advance the EA by one generation.

        Increments ``current_generation``, refreshes ``self.population`` from
        the database, passes the population through each ``EAStep`` in
        ``self.operations`` in order, then commits the result.
        """
        self.current_generation += 1
        self.fetch_population()
        for op in self.operations:
            self.population = op(self.population)
        self._commit()

    def run(self) -> None:
        """Run the EA for ``num_steps`` generations.

        Displays a Rich progress bar while iterating. Prints a completion
        rule to the console when finished.
        """
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