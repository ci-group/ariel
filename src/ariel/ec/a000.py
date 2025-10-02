"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Sequence
from pathlib import Path
from typing import cast

# Third-party libraries
import numpy as np
from pydantic_settings import BaseSettings
from rich.console import Console
from rich.traceback import install

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
RNG = np.random.default_rng(SEED)

# Type Aliases
type Integers = Sequence[int]
type Floats = Sequence[float]


class IntegersGeneratorSettings(BaseSettings):
    integers_endpoint: bool = True
    choice_replace: bool = True
    choice_shuffle: bool = False


config = IntegersGeneratorSettings()


class IntegersGenerator:
    @staticmethod
    def integers(
        low: int,
        high: int,
        size: int | Sequence[int] | None = 1,
        *,
        endpoint: bool | None = None,
    ) -> Integers:
        endpoint = endpoint or config.integers_endpoint
        generated_values = RNG.integers(
            low=low,
            high=high,
            size=size,
            endpoint=endpoint,
        )
        return cast("Integers", generated_values.astype(int).tolist())

    @staticmethod
    def choice(
        value_set: int | Integers,
        size: int | Sequence[int] | None = 1,
        probabilities: Sequence[float] | None = None,
        axis: int = 0,
        *,
        replace: bool | None = None,
        shuffle: bool | None = None,
    ) -> Integers:
        replace = replace or config.choice_replace
        shuffle = shuffle or config.choice_shuffle
        generated_values = np.array(
            RNG.choice(
                a=value_set,
                size=size,
                replace=replace,
                p=probabilities,
                axis=axis,
                shuffle=shuffle,
            ),
        )
        return cast("Integers", generated_values.astype(int).tolist())


class IntegerMutator:
    @staticmethod
    def random_swap(
        individual: Integers,
        low: int,
        high: int,
        mutation_probability: float,
    ) -> Integers:
        shape = np.asarray(individual).shape
        mutator = RNG.integers(
            low=low,
            high=high,
            size=shape,
            endpoint=True,
        )
        mask = RNG.choice(
            [True, False],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        new_genotype = np.where(mask, mutator, individual).astype(int).tolist()
        return cast("Integers", new_genotype.astype(int).tolist())

    @staticmethod
    def integer_creep(
        individual: Integers,
        span: int,
        mutation_probability: float,
    ) -> Integers:
        # Prep
        ind_arr = np.array(individual)
        shape = ind_arr.shape

        # Generate mutation values
        mutator = RNG.integers(
            low=1,
            high=span,
            size=shape,
            endpoint=True,
        )

        # Include negative mutations
        sub_mask = RNG.choice(
            [-1, 1],
            size=shape,
        )

        # Determine which positions to mutate
        do_mask = RNG.choice(
            [1, 0],
            size=shape,
            p=[mutation_probability, 1 - mutation_probability],
        )
        mutation_mask = mutator * sub_mask * do_mask
        new_genotype = ind_arr + mutation_mask
        return cast("Integers", new_genotype.astype(int).tolist())
                    
def main() -> None:
    """Entry point."""
    console.log(IntegersGenerator.integers(-5, 5, 5))
    example = IntegersGenerator.choice([1, 3, 4], (2, 5))
    console.log(example)
    example2 = IntegerMutator.integer_creep(
        example,
        span=1,
        mutation_probability=1,
    )
    console.log(example2)


if __name__ == "__main__":
    main()
