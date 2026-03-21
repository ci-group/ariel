"""ARIEL Individual."""
from collections.abc import Hashable, Sequence

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel

type JSONPrimitive = str | int | float | bool
type JSONType = JSONPrimitive | Sequence[JSONType] | dict[Hashable, JSONType]
type JSONIterable = Sequence[JSONType] | dict[Hashable, JSONType]


class Individual(SQLModel, table=True):
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
        if self.fitness_ is None:
            raise ValueError(f"fitness accessed before evaluation: {self.id=}")
        return self.fitness_

    @fitness.setter
    def fitness(self, value: int | float) -> None:
        self.requires_eval = False
        self.fitness_ = float(value)

    # -- Genotype property -----------------------------------------------------

    @property
    def genotype(self) -> JSONIterable:
        if self.genotype_ is None:
            raise ValueError(f"genotype accessed before initialization: {self.id=}")
        return self.genotype_

    @genotype.setter
    def genotype(self, value: JSONIterable) -> None:
        self.requires_init = not bool(value)
        self.genotype_ = value

    # -- Tags property ---------------------------------------------------------

    @property
    def tags(self) -> dict[JSONType, JSONType]:
        return self.tags_

    @tags.setter
    def tags(self, update: dict[JSONType, JSONType]) -> None:
        self.tags_ = {**self.tags_, **update}
