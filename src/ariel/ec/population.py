"""Population Class for the EC module."""

import random
from collections.abc import Callable, Iterator
from typing import Literal, overload

from ariel.ec.individual import Individual


def _safe_attr(individual: Individual, attribute: str) -> float:
    """Safely retrieve a numeric attribute from an individual.

    Attempts to read ``attribute`` from ``individual`` and cast it to
    ``float``. Returns ``float("-inf")`` instead of raising when the
    attribute is missing, ``None``, or non-numeric — allowing callers to
    use this as a sort key without crashing on unevaluated individuals.

    Parameters
    ----------
    individual : Individual
        The individual to inspect.
    attribute : str
        Name of the attribute to retrieve (e.g. ``"fitness_"``).

    Returns
    -------
    float
        The attribute value cast to ``float``, or ``float("-inf")`` if
        the attribute is absent, ``None``, or cannot be cast.
    """
    try:
        val = getattr(individual, attribute)
        return float(val) if val is not None else float("-inf")
    except (ValueError, AttributeError, TypeError):
        return float("-inf")


class Population:
    """
    Ordered, mutable container of ``Individual`` objects.

    Supports a chainable query API modelled after NumPy / pandas idioms.
    All filter and sort methods return new ``Population`` instances —
    the original is never mutated.

    Parameters
    ----------
    individuals : list[Individual]
        The initial list of individuals. A shallow copy is stored
        internally so that mutations to the source list do not affect the
        population.

    Examples
    --------
    >>> population.alive.sample(50).best(sort="max", attribute="fitness", n=10)
    """

    def __init__(self, individuals: list[Individual]) -> None:
        self.population: list[Individual] = list(individuals)

    # -- Core sequence protocol ------------------------------------------------

    def __len__(self) -> int:
        """Return the number of individuals in the population.

        Returns
        -------
        int
            Number of individuals currently stored.
        """
        return len(self.population)

    def __iter__(self) -> Iterator[Individual]:
        """Iterate over individuals in insertion order.

        Returns
        -------
        Iterator[Individual]
            An iterator over the underlying individual list.
        """
        return iter(self.population)

    def __repr__(self) -> str:
        """Return a concise string representation showing population size.

        Returns
        -------
        str
            A string of the form ``Population(n=<size>)``.
        """
        return f"Population(n={len(self.population)})"

    def __bool__(self) -> bool:
        """Return ``True`` when the population is non-empty.

        Returns
        -------
        bool
            ``False`` if the population contains no individuals,
            ``True`` otherwise.
        """
        return bool(self.population)

    def __add__(self, other: "Population") -> "Population":
        """Concatenate two populations into a new one.

        Parameters
        ----------
        other : Population
            The population to append to the current one.

        Returns
        -------
        Population
            A new population containing all individuals from both operands,
            with ``self``'s individuals preceding ``other``'s.
        """
        return Population(self.population + other.population)

    @overload
    def __getitem__(self, index: int) -> Individual: ...

    @overload
    def __getitem__(self, index: slice) -> "Population": ...

    def __getitem__(self, index: int | slice) -> "Individual | Population":
        """Retrieve an individual or a slice of the population.

        Parameters
        ----------
        index : int or slice
            An integer index returns a single ``Individual``; a slice
            returns a new ``Population``.

        Returns
        -------
        Individual
            When ``index`` is an ``int``.
        Population
            When ``index`` is a ``slice``.
        """
        if isinstance(index, slice):
            return Population(self.population[index])
        return self.population[index]

    # -- Mutation helpers ------------------------------------------------------

    def append(self, individual: Individual) -> None:
        """Append a single individual to the population in place.

        Parameters
        ----------
        individual : Individual
            The individual to add.
        """
        self.population.append(individual)

    def extend(self, other: "Population | list[Individual]") -> None:
        """Extend the population in place with another population or list.

        Parameters
        ----------
        other : Population or list[Individual]
            Individuals to append. Both ``Population`` instances and plain
            lists of ``Individual`` objects are accepted.
        """
        self.population.extend(
            other.population if isinstance(other, Population) else other,
            )

    def to_list(self) -> list[Individual]:
        """Return a plain Python list of the individuals.

        Returns
        -------
        list[Individual]
            A shallow copy of the internal individual list.
        """
        return list(self.population)

    # -- Chainable query API ---------------------------------------------------

    def sample(self, n: int) -> "Population":
        """Return a new population of up to ``n`` randomly drawn individuals.

        Sampling is performed without replacement. If ``n`` exceeds the
        current population size the entire population is returned in a
        random order.

        Parameters
        ----------
        n : int
            Maximum number of individuals to sample.

        Returns
        -------
        Population
            A new population containing ``min(n, len(self))`` individuals
            drawn uniformly at random without replacement.
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
        """Return the top ``n`` individuals sorted by a numeric attribute.

        Uses ``_safe_attr`` to retrieve values, so unevaluated individuals
        (with ``None`` or missing attributes) sort to the bottom regardless
        of ``sort`` direction.

        Parameters
        ----------
        sort : {"max", "min"}, optional
            Sorting direction.

            ``"max"``
                Highest attribute value ranks first. Default.
            ``"min"``
                Lowest attribute value ranks first.
        attribute : str, optional
            Name of the ``Individual`` attribute to sort by. Pass
            ``"fitness_"`` (default) to use the raw stored value, or
            ``"fitness"`` to use the property (which raises for
            unevaluated individuals — caught internally and treated as
            ``-inf``).
        n : int, optional
            Number of top individuals to return. Default is ``1``.

        Returns
        -------
        Population
            A new population containing the ``n`` highest- (or lowest-)
            ranked individuals.
        """
        reverse: bool = sort == "max"

        def key(ind: Individual) -> float:
            return _safe_attr(ind, attribute)
        return Population(sorted(self.population, key=key, reverse=reverse)[:n])

    def shuffle(self) -> "Population":
        """Return a new population with individuals in a random order.

        The original population is not modified.

        Returns
        -------
        Population
            A new population containing the same individuals in a randomly
            shuffled order.
        """
        data = list(self.population)
        random.shuffle(data)
        return Population(data)

    def where(self, predicate: Callable[[Individual], bool]) -> "Population":
        """Query the current population.

        Return a new population containing only
        individuals that satisfy ``predicate``.

        Parameters
        ----------
        predicate : Callable[[Individual], bool]
            A function that takes an ``Individual`` and returns ``True``
            for individuals that should be included.

        Returns
        -------
        Population
            A filtered population of all individuals for which
            ``predicate(ind)`` is ``True``.

        Examples
        --------
        >>> population.where(lambda ind: ind.alive)
        >>> population.where(lambda ind: ind.tags.get("selected", False))
        """
        return Population([ind for ind in self.population if predicate(ind)])

    # -- Convenience filter properties -----------------------------------------

    @property
    def alive(self) -> "Population":
        """All individuals with ``alive=True``.

        Returns
        -------
        Population
            A filtered population of living individuals.
        """
        return self.where(lambda ind: ind.alive)

    @property
    def dead(self) -> "Population":
        """All individuals with ``alive=False``.

        Returns
        -------
        Population
            A filtered population of dead individuals.
        """
        return self.where(lambda ind: not ind.alive)

    @property
    def unevaluated(self) -> "Population":
        """All individuals where ``requires_eval=True``.

        Returns
        -------
        Population
            A filtered population of unevaluated individuals.
        """
        return self.where(lambda ind: ind.requires_eval)

    @property
    def evaluated(self) -> "Population":
        """All individuals where ``requires_eval=False``.

        Returns
        -------
        Population
            A filtered population of evaluated individuals.
        """
        return self.where(lambda ind: not ind.requires_eval)

    # -- Numerical properties --------------------------------------------------

    @property
    def size(self) -> int:
        """Number of individuals currently in the population.

        Returns
        -------
        int
            Equivalent to ``len(self)``.
        """
        return len(self.population)

    # -- Constructors ----------------------------------------------------------

    @classmethod
    def empty(cls) -> "Population":
        """Construct an empty population.

        Returns
        -------
        Population
            A new population with no individuals.
        """
        return cls([])
