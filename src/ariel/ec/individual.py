"""ARIEL Individual."""
from collections.abc import Hashable, Sequence

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

type JSONPrimitive = str | int | float | bool
type JSONType = JSONPrimitive | Sequence[JSONType] | dict[Hashable, JSONType]
type JSONIterable = Sequence[JSONType] | dict[Hashable, JSONType]


class Individual(SQLModel, table=True):
    """
    A single candidate solution in an evolutionary algorithm.

    Persisted as a SQLite row via SQLModel. All mutable state — fitness,
    genotype, and tags — is accessed through properties that enforce
    consistency invariants (e.g. clearing ``requires_eval`` on fitness
    assignment).

    Parameters
    ----------
    id : int or None, optional
        Primary key assigned by the database on first commit. ``None``
        before the individual is persisted. Default is ``None``.
    alive : bool, optional
        Whether the individual is still active in the population.
        Indexed for fast survivor-selection queries. Default is ``True``.
    time_of_birth : int, optional
        Generation in which this individual was created. Set to ``-1``
        until assigned by the EA engine on first commit. Default is ``-1``.
    time_of_death : int, optional
        Last generation in which this individual was present. Updated on
        every commit by the EA engine. Default is ``-1``.
    requires_eval : bool, optional
        ``True`` when the individual's fitness has not yet been computed.
        Automatically set to ``False`` by the ``fitness`` setter. Indexed
        for fast unevaluated-individual queries. Default is ``True``.
    fitness_ : float or None, optional
        Raw stored fitness value. Access via the ``fitness`` property in
        user code; use ``fitness_`` only for direct database queries.
        Default is ``None``.
    requires_init : bool, optional
        ``True`` when the genotype has not yet been assigned. Automatically
        updated by the ``genotype`` setter. Indexed for fast queries.
        Default is ``True``.
    genotype_ : JSONIterable or None, optional
        Raw stored genotype. Access via the ``genotype`` property in user
        code. Stored as a JSON column to support arbitrary nested
        structures. Default is ``None``.
    tags_ : dict[JSONType, JSONType], optional
        Raw stored tag dictionary. Access and update via the ``tags``
        property. Stored as a JSON column. Default is ``{}``.
    """

    id: int | None = Field(default=None, primary_key=True)

    # -- Lifetime --------------------------------------------------------------
    alive: bool = Field(default=True, index=True)
    time_of_birth: int = Field(default=-1, index=True)
    time_of_death: int = Field(default=-1, index=True)

    # -- Fitness ---------------------------------------------------------------
    requires_eval: bool = Field(default=True, index=True)
    fitness_: float | None = Field(default=None, index=True)

    # -- Genotype --------------------------------------------------------------
    requires_init: bool = Field(default=True, index=True)
    genotype_: JSONIterable | None = Field(default=None, sa_column=Column(JSON))

    # -- Tags ------------------------------------------------------------------
    tags_: dict[JSONType, JSONType] = Field(default={}, sa_column=Column(JSON))

    # -- Fitness property ------------------------------------------------------

    @property
    def fitness(self) -> float:
        """Evaluated fitness of the individual.

        Returns
        -------
        float
            The stored fitness value.

        Raises
        ------
        ValueError
            If accessed before the individual has been evaluated
            (i.e. ``fitness_`` is ``None``).
        """
        if self.fitness_ is None:
            raise ValueError(f"fitness accessed before evaluation: {self.id=}")
        return self.fitness_

    @fitness.setter
    def fitness(self, value: float) -> None:
        """Assign the fitness and mark the individual as evaluated.

        Sets ``requires_eval`` to ``False`` and stores ``value`` as a
        ``float`` in ``fitness_``.

        Parameters
        ----------
        value : int or float
            The fitness score to assign.
        """
        self.requires_eval = False
        self.fitness_ = float(value)

    # -- Genotype property -----------------------------------------------------

    @property
    def genotype(self) -> JSONIterable:
        """Genotype of the individual.

        Returns
        -------
        JSONIterable
            The stored genotype sequence or mapping.

        Raises
        ------
        ValueError
            If accessed before the genotype has been initialised
            (i.e. ``genotype_`` is ``None``).
        """
        if self.genotype_ is None:
            raise ValueError(f"genotype accessed before initialization: {self.id=}")
        return self.genotype_

    @genotype.setter
    def genotype(self, value: JSONIterable) -> None:
        """Assign the genotype and update the initialisation flag.

        Sets ``requires_init`` to ``False`` when ``value`` is non-empty,
        ``True`` when it is empty or falsy.

        Parameters
        ----------
        value : JSONIterable
            The genotype to assign. May be any JSON-serialisable sequence
            or mapping.
        """
        self.requires_init = not bool(value)
        self.genotype_ = value

    # -- Tags property ---------------------------------------------------------

    @property
    def tags(self) -> dict[JSONType, JSONType]:
        """Metadata tags attached to this individual.

        Returns
        -------
        dict[JSONType, JSONType]
            The current tag dictionary.
        """
        return self.tags_

    @tags.setter
    def tags(self, update: dict[JSONType, JSONType]) -> None:
        """Merge new tags into the existing tag dictionary.

        Existing keys are overwritten by keys present in ``update``;
        keys absent from ``update`` are preserved.

        Parameters
        ----------
        update : dict[JSONType, JSONType]
            Key-value pairs to merge into ``tags_``.
        """
        self.tags_ = {**self.tags_, **update}
