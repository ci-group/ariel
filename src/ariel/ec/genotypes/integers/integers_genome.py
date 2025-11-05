from collections.abc import Sequence
from typing import cast
import numpy as np
from ariel.ec.genotypes.genotype import Genotype
from ariel.ec.mutations import IntegerMutator
from ariel.ec.crossovers import IntegerCrossover
from pydantic import BaseSettings

SEED = 42
RNG = np.random.default_rng(SEED)
# Type Aliases
type Integers = Sequence[int]
type Floats = Sequence[float]

class IntegersGenome(Genotype):

    @staticmethod
    def get_crossover_object() -> "IntegerCrossover":
        return IntegerCrossover()
    
    @staticmethod
    def get_mutator_object() -> "IntegerMutator":
        return IntegerMutator()

    @staticmethod
    def create_individual(
        low: int,
        high: int,
        size: int | Sequence[int] | None = 1,
        *,
        endpoint: bool | None = None,
    ) -> Integers:
        endpoint = endpoint
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
        replace = replace
        shuffle = shuffle
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
