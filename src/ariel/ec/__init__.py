"""EC module for ARIEL."""

from ariel.ec.crossover import Crossover
from ariel.ec.ea import EA, DBHandlingMode, EASettings, EAStep, config
from ariel.ec.generators import (
    SEED,
    FloatMutator,
    Floats,
    FloatsGenerator,
    IntegerMutator,
    Integers,
    IntegersGenerator,
)
from ariel.ec.individual import (
    Individual,
    JSONIterable,
    JSONPrimitive,
    JSONType,
)
from ariel.ec.population import Population

__all__: list[str] = [

    # EA engine
    "EA",

    # Shared
    "SEED",

    # Crossover
    "Crossover",
    "DBHandlingMode",
    "EASettings",
    "EAStep",
    "FloatMutator",
    "Floats",

    # Float generators / mutators
    "FloatsGenerator",

    # Data layer
    "Individual",
    "IntegerMutator",
    "Integers",

    # Integer generators / mutators
    "IntegersGenerator",
    "JSONIterable",
    "JSONPrimitive",
    "JSONType",

    # Population
    "Population",
    "config",
]
