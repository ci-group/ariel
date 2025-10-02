"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Hashable, Sequence
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install
from sqlalchemy import JSON, Column, Engine
from sqlmodel import Field, Session, SQLModel, create_engine

# Local libraries
from ariel.ec.a000 import IntegersGenerator

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)
DB_NAME = "database.db"
DB_PATH: Path = DATA / DB_NAME
SEED = 42

# Global functions
install(width=180)
console = Console()
prnt = console.print
rlg = console.log
RNG = np.random.default_rng(SEED)

# Typing aliases
type JSONPrimitive = str | int | float | bool
type JSONType = JSONPrimitive | Sequence[JSONType] | dict[Hashable, JSONType]
type JSONIterable = Sequence[JSONType] | dict[Hashable, JSONType]


def init_database() -> Engine:
    """
    Initialize a database with a JSON data model.

    Returns
    -------
    Engine
        The SQLAlchemy engine instance.
    """
    # Delete the database file if it exists
    if DB_PATH.exists():
        DB_PATH.unlink()

    # Create the database engine
    engine = create_engine(
        f"sqlite:///{DB_PATH}",
    )
    SQLModel.metadata.create_all(engine)
    return engine


class Individual(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)

    # ------------------------ LIFE TIME ------------------------
    alive: bool = Field(default=True, index=True)

    time_of_birth: int = Field(default=-1, index=True)
    time_of_death: int = Field(default=-1, index=True)

    # parents_id removed for now; reintroduce later if needed.

    # ------------------------ FITNESS ------------------------
    requires_eval: bool = Field(default=True, index=True)
    fitness_: float | None = Field(default=None, index=True)

    @property
    def fitness(self) -> float:
        if self.fitness_ is None:
            msg = "Trying to fetch uninitialized data in fitness!\n"
            msg += f"--> {self.fitness_=}"
            raise ValueError(msg)
        return self.fitness_

    @fitness.setter
    def fitness(self, fitness_value: float | None) -> None:
        if fitness_value is None:
            msg = "Trying to assign `None` to fitness!\n"
            msg += f"--> {self.fitness_value=}"
            raise ValueError(msg)
        self.requires_eval = False
        self.fitness_ = fitness_value

    # ------------------------ GENOTYPE ------------------------
    requires_init: bool = Field(default=True, index=True)
    genotype_: JSONIterable | None = Field(default=None, sa_column=Column(JSON))

    @property
    def genotype(self) -> JSONIterable:
        if self.genotype_ is None:
            msg = "Trying to fetch uninitialized data in genotype!"
            raise ValueError(msg)
        return self.genotype_

    @genotype.setter
    def genotype(self, individual_genotype: JSONIterable) -> None:
        self.requires_init = not bool(individual_genotype)
        self.genotype_ = individual_genotype

    # ------------------------ TAGS ------------------------
    tags_: dict[JSONType, JSONType] = Field(
        default={},
        sa_column=Column(JSON),
    )

    @property
    def tags(self) -> dict[JSONType, JSONType]:
        return self.tags_

    @tags.setter
    def tags(self, tag: dict[JSONType, JSONType]) -> None:
        self.tags_ = {**self.tags_, **tag}


def main() -> None:
    """Entry point."""
    # Initialize the database
    engine = init_database()

    # Save data
    with Session(engine) as session:
        ind = Individual()

        # Generators
        ind.genotype = IntegersGenerator.integers(low=0, high=10, size=5)

        # Tags
        ind.tags = {"a": ["1", 2, 3]}
        ind.tags = {"b": ("1", 2, 3)}
        ind.tags = {"c": 1}
        ind.tags = {"d": True}

        prnt(ind)
        session.add(ind)
        session.commit()
        prnt(ind)
        session.refresh(ind)
        prnt(ind)


if __name__ == "__main__":
    main()
