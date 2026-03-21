"""Population Class for the EC module."""

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
        self.population: list[Individual] = list(individuals)

    # -- Core sequence protocol ------------------------------------------------

    def __len__(self) -> int:
        """Get length of population.

        Returns
        -------
        int
        """
        return len(self.population)

    def __iter__(self) -> Iterator[Individual]:
        """Do iter stuff.

        Returns
        -------
        Iterator[Individual]
        """
        return iter(self.population)

    def __repr__(self) -> str:
        """Show length of population.

        Returns
        -------
        len(Population): str
        """
        return f"Population(n={len(self.population)})"

    def __bool__(self) -> bool:
        """See if Population.

        Does it exist? You never know.

        Returns
        -------
        bool
        """
        return bool(self.population)

    def __add__(self, other: "Population") -> "Population":
        """Add a population to the current population.

        Returns
        -------
        Population
        """
        return Population(self.population + other.population)

    @overload
    def __getitem__(self, index: int) -> Individual: ...

    @overload
    def __getitem__(self, index: slice) -> "Population": ...

    def __getitem__(self, index: int | slice) -> "Individual | Population":
        """Get item method.

        Returns
        -------
        Individual | Population
        """
        if isinstance(index, slice):
            return Population(self.population[index])
        return self.population[index]

    # -- Mutation helpers ------------------------------------------------------

    def append(self, individual: Individual) -> None:
        """Append individual to Population."""
        self.population.append(individual)

    def extend(self, other: "Population | list[Individual]") -> None:
        """Extend Population with another Population."""
        self.population.extend(
            other.population if isinstance(other, Population) else other,
            )

    def to_list(self) -> list[Individual]:
        """Turn population to default Python list object.

        Returns
        -------
        list[Individual]
        """
        return list(self.population)

    # -- Chainable query API ---------------------------------------------------

    def sample(self, n: int) -> "Population":
        """Return a new Population with *n* randomly drawn individuals.

        Parameters
        ----------
        n: int
            Number of individuals to sample

        Returns
        -------
        population: Population
            Sample of n individuals (without replacement) from the population
        """
        return Population(random.sample(self.population,
                                        min(n, len(self.population)),
                                        ),
                                            )

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
            - "max" -> highest first (default),
            - "min" -> lowest first.
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

        def key(ind: Individual) -> float:
            return _safe_attr(ind, attribute)
        return Population(sorted(self.population, key=key, reverse=reverse)[:n])

    def shuffle(self) -> "Population":
        """Return a new Population with the same individuals in random order.

        Returns
        -------
        Population : list[Individual]
        """
        data = list(self.population)
        random.shuffle(data)
        return Population(data)

    def where(self, predicate: Callable[[Individual], bool]) -> "Population":
        """Query the population.

        Return a new Population containing
        only individuals matching *predicate*.

        Parameters
        ----------
        predicate : Callable[[Individual], bool]
            Predicate rule for what to pull from the population
            ```python
                # Get all alive individuals
                population.where(lambda ind: ind.alive)
            ```

        Returns
        -------
        Population: list[Individual]
        """
        return Population([ind for ind in self.population if predicate(ind)])

    # -- Convenience filter properties -----------------------------------------

    @property
    def alive(self) -> "Population":
        """Get all alive individual from the current population."""
        return self.where(lambda ind: ind.alive)

    @property
    def dead(self) -> "Population":
        """Get all dead individual from the current population."""
        return self.where(lambda ind: not ind.alive)

    @property
    def unevaluated(self) -> "Population":
        """Get all unevaluated individuals from the current population."""
        return self.where(lambda ind: ind.requires_eval)

    @property
    def evaluated(self) -> "Population":
        """Get all evaluated individuals  from the current population."""
        return self.where(lambda ind: not ind.requires_eval)

    # -- Numerical Properties -----------------------------------------

    @property
    def size(self) -> int:
        """Population size."""
        return len(self.population)

    # -- Constructors ----------------------------------------------------------

    @classmethod
    def empty(cls) -> "Population":
        """Initialise empty population.

        Returns
        -------
        Population
        """
        return cls([])
