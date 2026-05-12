"""Archive: read-only query interface for a persisted EA database."""

import random
from pathlib import Path
from typing import Literal

import numpy as np
from sqlalchemy import Engine, Select, create_engine, func
from sqlmodel import Session, col, select

from ariel.ec.individual import Individual
from ariel.ec.population import Population

type FitnessMode = Literal["min", "max"]
type AgeRange = tuple[int, int] | None
type FitnessRange = tuple[float, float] | None


class Archive:
    """Read-only interface to a persisted EA database.

    Opens a SQLite database written by the EA engine and exposes targeted
    query methods for retrieving archived individuals. Particularly useful
    for the J.E.S.U.S. (Joint Evolution Strategies with Undead Sampling)
    pattern: injecting historically successful individuals back into a
    stagnating population.

    Parameters
    ----------
    db_path : Path or str
        Path to the SQLite ``.db`` file produced by an EA run.

    Examples
    --------
    >>> archive = Archive("__data__/run_1/database.db")
    >>> ind = archive.random_individual(birth_range=(0, 30))
    >>> pop = archive.tournament_population(n=10, tournament_size=3)
    >>> hall = archive.hall_of_fame(n=5, fitness_mode="min")
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            msg = f"Archive database not found: {self.db_path}"
            raise FileNotFoundError(msg)
        self.engine: Engine = create_engine(f"sqlite:///{self.db_path}")

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _base_query(
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
        fitness_range: FitnessRange = None,
        alive_filter: bool | None = None,
        evaluated_only: bool = True,
    ) -> Select[tuple[Individual]]:
        """Construct a base SQL query with optional filters.

        Parameters
        ----------
        birth_range : (int, int) or None, optional
            Inclusive ``(low, high)`` filter on ``time_of_birth``.
        death_range : (int, int) or None, optional
            Inclusive ``(low, high)`` filter on ``time_of_death``.
        fitness_range : (float, float) or None, optional
            Inclusive ``(low, high)`` filter on raw ``fitness_``.
        alive_filter : bool or None, optional
            ``True`` restricts to living individuals, ``False`` to dead ones,
            ``None`` applies no filter. Default is ``None``.
        evaluated_only : bool, optional
            Restrict to individuals that have been evaluated
            (i.e. have a non-``NULL`` fitness). Default ``True``.

        Returns
        -------
        Select[tuple[Individual]]
            SQLModel Select statement with the specified filters applied.
        """
        statement = select(Individual)

        if alive_filter is not None:
            statement = statement.where(
                Individual.alive == alive_filter,
            )
        if evaluated_only:
            statement = statement.where(Individual.requires_eval == False)  # noqa: E712
        if birth_range is not None:
            lo, hi = birth_range
            statement = statement.where(Individual.time_of_birth >= lo).where(
                Individual.time_of_birth <= hi,
            )
        if death_range is not None:
            lo, hi = death_range
            statement = statement.where(Individual.time_of_death >= lo).where(
                Individual.time_of_death <= hi,
            )
        if fitness_range is not None:
            lo, hi = fitness_range
            statement = statement.where(Individual.fitness_ >= lo).where(
                Individual.fitness_ <= hi,
            )

        return statement

    def _fetch_all(self, statement) -> list[Individual]:
        with Session(self.engine) as session:
            return list(session.exec(statement).all())

    # -- Single-individual queries ---------------------------------------------

    def random_individual(
        self,
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
        fitness_range: FitnessRange = None,
        only_dead: bool = False,
    ) -> Individual:
        """Return a single uniformly-random archived individual.

        Parameters
        ----------
        birth_range : (int, int) or None, optional
            Inclusive ``(lo, hi)`` filter on ``time_of_birth``.
        death_range : (int, int) or None, optional
            Inclusive ``(lo, hi)`` filter on ``time_of_death``.
        fitness_range : (float, float) or None, optional
            Inclusive ``(lo, hi)`` filter on raw ``fitness_``.
        only_dead : bool, optional
            Restrict to individuals that are no longer alive. Default ``False``.

        Returns
        -------
        Individual

        Raises
        ------
        ValueError
            If no individuals match the supplied filters.
        """
        statement = (
            self._base_query(
                birth_range=birth_range,
                death_range=death_range,
                fitness_range=fitness_range,
                alive_filter=False if only_dead else None,
            )
            .order_by(func.random())
            .limit(1)
        )

        results = self._fetch_all(statement)
        if not results:
            msg = "No individuals match the given filters."
            raise ValueError(msg)
        return results[0]

    def best_individual(
        self,
        fitness_mode: FitnessMode = "min",
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
    ) -> Individual:
        """Return the single best individual in the archive.

        Parameters
        ----------
        fitness_mode : {"min", "max"}, optional
            ``"min"`` returns the lowest-fitness individual (minimisation);
            ``"max"`` returns the highest (maximisation). Default is ``"min"``.
        birth_range : (int, int) or None, optional
        death_range : (int, int) or None, optional

        Returns
        -------
        Individual

        Raises
        ------
        ValueError
            If the archive contains no evaluated individuals.
        """
        statement = self._base_query(
            birth_range=birth_range,
            death_range=death_range,
        )
        match fitness_mode:
            case "min":
                statement = statement.order_by(col(Individual.fitness_).asc())
            case "max":
                statement = statement.order_by(col(Individual.fitness_).desc())
        statement = statement.limit(1)

        results = self._fetch_all(statement)
        if not results:
            msg = "No evaluated individuals found in the archive."
            raise ValueError(msg)
        return results[0]

    # -- Population queries ----------------------------------------------------

    def random_population(
        self,
        n: int,
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
        fitness_range: FitnessRange = None,
        only_dead: bool = False,
    ) -> Population:
        """Return a uniformly-random sample of up to ``n`` archived individuals.

        Parameters
        ----------
        n : int
            Number of individuals to return.
        birth_range : (int, int) or None, optional
        death_range : (int, int) or None, optional
        fitness_range : (float, float) or None, optional
        only_dead : bool, optional

        Returns
        -------
        Population
            May be smaller than ``n`` if fewer matching individuals exist.
        """
        statement = (
            self._base_query(
                birth_range=birth_range,
                death_range=death_range,
                fitness_range=fitness_range,
                alive_filter=False if only_dead else None,
            )
            .order_by(func.random())
            .limit(n)
        )

        return Population(self._fetch_all(statement))

    def hall_of_fame(
        self,
        n: int = 10,
        fitness_mode: FitnessMode = "min",
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
    ) -> Population:
        """Return the all-time ``n`` best individuals from the archive.

        Parameters
        ----------
        n : int, optional
            Size of the hall of fame. Default is ``10``.
        fitness_mode : {"min", "max"}, optional
            Optimisation direction. Default is ``"min"``.
        birth_range : (int, int) or None, optional
        death_range : (int, int) or None, optional

        Returns
        -------
        Population
        """
        statement = self._base_query(
            birth_range=birth_range,
            death_range=death_range,
        )
        match fitness_mode:
            case "min":
                statement = statement.order_by(col(Individual.fitness_).asc())
            case "max":
                statement = statement.order_by(col(Individual.fitness_).desc())
        statement = statement.limit(n)

        return Population(self._fetch_all(statement))

    def tournament_population(
        self,
        n: int,
        tournament_size: int = 3,
        fitness_mode: FitnessMode = "min",
        *,
        birth_range: AgeRange = None,
        death_range: AgeRange = None,
        pool_multiplier: int = 4,
    ) -> Population:
        """Select ``n`` individuals via repeated tournament selection.

        Fetches a candidate pool of ``n * pool_multiplier`` individuals from
        the database at random, then runs ``n`` tournaments of
        ``tournament_size`` over that pool. This is the core mechanism
        behind the J.E.S.U.S. strategy.

        Parameters
        ----------
        n : int
            Number of tournament winners to return.
        tournament_size : int, optional
            Number of competitors drawn per tournament. Default is ``3``.
        fitness_mode : {"min", "max"}, optional
            ``"min"`` picks the lowest-fitness contestant as the winner
            (minimisation problem). ``"max"`` picks the highest. Default
            is ``"min"``.
        birth_range : (int, int) or None, optional
        death_range : (int, int) or None, optional
        pool_multiplier : int, optional
            Pool size relative to ``n``; larger values increase draw
            diversity at the cost of a bigger DB fetch. Default is ``4``.

        Returns
        -------
        Population
            A population of ``n`` tournament winners.

        Raises
        ------
        ValueError
            If the fetched pool is empty.
        AssertionError
            If ``tournament_size`` exceeds the pool size.
        """
        pool = self.random_population(
            n * pool_multiplier,
            birth_range=birth_range,
            death_range=death_range,
        )
        pool_list = pool.to_list()

        if not pool_list:
            msg = "No individuals match the given filters."
            raise ValueError(msg)

        if tournament_size > len(pool_list):
            msg = f"tournament_size={tournament_size} exceeds pool size={len(pool_list)}."
            raise AssertionError(msg)

        def _key(ind: Individual) -> float:
            return ind.fitness_ if ind.fitness_ is not None else float("inf")

        sample_size = min(tournament_size, len(pool_list))
        if fitness_mode == "min":
            winners = [
                min(random.sample(pool_list, sample_size), key=_key)
                for _ in range(n)
            ]
        else:
            winners = [
                max(random.sample(pool_list, sample_size), key=_key)
                for _ in range(n)
            ]
        return Population(winners)

    # Used in PPSN experiments
    def jesi(
        self,
        median_age: int,
        num_jesi: int = 10,
        tournament_size: int = 3,
        fitness_mode: FitnessMode = "min",
        age_window: int = 10,
    ) -> Population:
        """Retrieve resurrected individuals using the J.E.S.U.S. strategy.

        Selects historically well-performing dead individuals from a specific
        age band — those that lived long enough to be considered "experienced"
        but are no longer in the active population. Designed to inject
        diversity back into a stagnating run.

        The birth and death windows are derived from ``median_age`` and
        ``age_window``:

        - ``death_range = (median_age - age_window, median_age)``
        - ``birth_range = (0, death_range[0] - 5)``

        Parameters
        ----------
        median_age : int
            Target generation of death around which to centre the search.
        num_jesi : int, optional
            Number of individuals to resurrect. Default is ``10``.
        tournament_size : int, optional
            Tournament size for selection within the candidate pool.
            Default is ``3``.
        fitness_mode : {"min", "max"}, optional
            Optimisation direction for tournament selection. Default ``"min"``.
        age_window : int, optional
            Half-width of the death-generation window. Default is ``10``.

        Returns
        -------
        Population
            A population of ``num_jesi`` resurrected individuals.
        """
        death_lo = max(0, median_age - age_window)
        death_hi = median_age
        birth_hi = max(0, death_lo - 5)

        return self.tournament_population(
            n=num_jesi,
            tournament_size=tournament_size,
            fitness_mode=fitness_mode,
            birth_range=(0, birth_hi),
            death_range=(death_lo, death_hi),
        )

    def by_generation(self, generation: int) -> Population:
        """Return all individuals present at a given generation.

        An individual was alive at ``generation`` if it was born at or
        before it and died at or after it.

        Parameters
        ----------
        generation : int
            The generation index to query.

        Returns
        -------
        Population
        """
        statement = (
            select(Individual)
            .where(Individual.requires_eval == False)  # noqa: E712
            .where(Individual.time_of_birth <= generation)
            .where(Individual.time_of_death >= generation)
        )
        return Population(self._fetch_all(statement))

    def fitness_percentile_population(
        self,
        lo_pct: float = 25.0,
        hi_pct: float = 75.0,
        n: int | None = None,
        *,
        fitness_mode: FitnessMode = "min",
        birth_range: AgeRange = None,
    ) -> Population:
        """Return individuals whose fitness falls within a percentile band.

        Useful for sampling "mediocre-but-interesting" individuals rather
        than always drawing from the elite tail, which can help preserve
        morphological diversity in the resurrected pool.

        Parameters
        ----------
        lo_pct : float, optional
            Lower percentile bound (0-100). Default is ``25.0``.
        hi_pct : float, optional
            Upper percentile bound (0-100). Default is ``75.0``.
        n : int or None, optional
            If given, randomly sample ``n`` individuals from the band.
            When ``None`` all matching individuals are returned.
        fitness_mode : {"min", "max"}, optional
            Direction used when interpreting "best". Affects nothing
            functionally in this method but kept for API consistency.
        birth_range : (int, int) or None, optional

        Returns
        -------
        Population
        """
        all_inds = self._fetch_all(self._base_query(birth_range=birth_range))
        if not all_inds:
            return Population.empty()

        fitnesses = np.array([ind.fitness_ for ind in all_inds], dtype=float)
        lo_val = float(np.percentile(fitnesses, lo_pct))
        hi_val = float(np.percentile(fitnesses, hi_pct))

        band = [
            ind for ind in all_inds
            if ind.fitness_ is not None and lo_val <= ind.fitness_ <= hi_val
        ]
        if n is not None:
            band = random.sample(band, min(n, len(band)))
        return Population(band)

    # -- Statistics / introspection --------------------------------------------

    @property
    def size(self) -> int:
        """Total number of evaluated individuals stored in the archive."""
        with Session(self.engine) as session:
            return session.exec(
                select(func.count(Individual.id)).where(
                    Individual.requires_eval == False,  # noqa: E712
                ),
            ).one()

    @property
    def generation_range(self) -> tuple[int, int]:
        """``(earliest_birth, latest_death)`` generation indices in the archive."""
        with Session(self.engine) as session:
            min_birth = session.exec(
                select(func.min(Individual.time_of_birth)),
            ).one()
            max_death = session.exec(
                select(func.max(Individual.time_of_death)),
            ).one()
        return (min_birth or 0, max_death or 0)

    def fitness_stats(self, birth_range: AgeRange = None) -> dict[str, float]:
        """Return summary fitness statistics for archived individuals.

        Parameters
        ----------
        birth_range : (int, int) or None, optional
            Restrict statistics to individuals born in this range.

        Returns
        -------
        dict[str, float]
            Keys: ``"min"``, ``"max"``, ``"mean"``, ``"std"``, ``"median"``.
            All ``NaN`` when no matching individuals exist.
        """
        inds = self._fetch_all(self._base_query(birth_range=birth_range))
        if not inds:
            nan = float("nan")
            return {
                "min": nan,
                "max": nan,
                "mean": nan,
                "std": nan,
                "median": nan,
            }

        fitnesses = np.array([ind.fitness_ for ind in inds], dtype=float)
        return {
            "min": float(np.min(fitnesses)),
            "max": float(np.max(fitnesses)),
            "mean": float(np.mean(fitnesses)),
            "std": float(np.std(fitnesses)),
            "median": float(np.median(fitnesses)),
        }

    def __repr__(self) -> str:
        """Return a string representation of the archive.

        Returns
        -------
        str
            A string representation of the archive.
        """
        return f"Archive(path={self.db_path!r}, size={self.size})"
