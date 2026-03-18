import random
from collections.abc import Callable, Iterator
from typing import Literal, overload

from ariel.ec.individual import Individual


def _safe_attr(individual: Individual, attribute: str) -> float:
    try:
        val = getattr(individual, attribute)
        return float(val) if val is not None else float("-inf")
    except (ValueError, AttributeError, TypeError):
        return float("-inf")


class Population:
    """
    Ordered, mutable container of Individual objects.

    Supports a numpy-inspired chainable query API:

        population
            .alive
            .sample(50)
            .best(sort="max", attribute="fitness", n=10)

    All filter / sort methods return new Population instances — the
    original is never mutated.
    """

    def __init__(self, individuals: list[Individual]) -> None:
        self._population: list[Individual] = list(individuals)

    # ── Core sequence protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._population)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._population)

    def __repr__(self) -> str:
        return f"Population(n={len(self._population)})"

    def __bool__(self) -> bool:
        return bool(self._population)

    def __add__(self, other: "Population") -> "Population":
        return Population(self._population + other._population)

    @overload
    def __getitem__(self, index: int) -> Individual: ...

    @overload
    def __getitem__(self, index: slice) -> "Population": ...

    def __getitem__(self, index: int | slice) -> "Individual | Population":
        if isinstance(index, slice):
            return Population(self._population[index])
        return self._population[index]

    # ── Mutation helpers ──────────────────────────────────────────────────────

    def append(self, individual: Individual) -> None:
        self._population.append(individual)

    def extend(self, other: "Population | list[Individual]") -> None:
        self._population.extend(other._population if isinstance(other, Population) else other)

    def to_list(self) -> list[Individual]:
        return list(self._population)

    # ── Chainable query API ───────────────────────────────────────────────────

    def sample(self, n: int) -> "Population":
        """Return a new Population with *n* randomly drawn individuals (without replacement)."""
        return Population(random.sample(self._population, min(n, len(self._population))))

    def best(
        self,
        *,
        sort: Literal["max", "min"] = "max",
        attribute: str = "fitness_",
        n: int = 1,
    ) -> "Population":
        """
        Return the top *n* individuals sorted by *attribute*.

        Parameters
        ----------
        sort :
            "max" -> highest first (default),
            "min" -> lowest first.
        attribute :
            Any attribute of Individual. Defaults to "fitness_"
            (the raw stored value).  Pass "fitness" to use the
            property (raises for un-evaluated individuals — caught
            internally and treated as -inf / +inf).
        n :
            Number of individuals to return.

        Returns
        -------
        Population : list[Individual]
        """
        reverse: bool = sort == "max"
        key: Callable[[Individual], float] = lambda ind: _safe_attr(ind, attribute)
        return Population(sorted(self._population, key=key, reverse=reverse)[:n])

    def shuffle(self) -> "Population":
        """Return a new Population with the same individuals in random order.

        Returns
        -------
        Population : list[Individual]
        """
        data = list(self._population)
        random.shuffle(data)
        return Population(data)

    def where(self, predicate: Callable[[Individual], bool]) -> "Population":
        """Return a new Population containing only individuals matching *predicate*.

        Parameters
        ----------
        predicate : Callable[[Individual], bool]
            Predicate rule for what to pull from the population
            ```python
                population.where(lambda ind: ind.alive) # Get all alive individuals
            ```

        Returns
        -------
        Population: list[Individual]
        """
        return Population([ind for ind in self._population if predicate(ind)])

    # ── Convenience filter properties ─────────────────────────────────────────

    @property
    def alive(self) -> "Population":
        return self.where(lambda ind: ind.alive)

    @property
    def dead(self) -> "Population":
        return self.where(lambda ind: not ind.alive)

    @property
    def unevaluated(self) -> "Population":
        return self.where(lambda ind: ind.requires_eval)

    @property
    def evaluated(self) -> "Population":
        return self.where(lambda ind: not ind.requires_eval)

    # ── Numerical Properties ─────────────────────────────────────────

    @property
    def size(self) -> int:
        return self.__len__()

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def empty(cls) -> "Population":
        return cls([])
