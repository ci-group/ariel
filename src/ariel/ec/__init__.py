"""EC module for ARIEL.

Symbols are exposed lazily via ``__getattr__`` so that importing a
submodule (for example
``ariel.ec.drone.genome_handlers.mounting_points``) does not eagerly
pull in the EA orchestration's heavy optional dependencies.

Concretely: ``Archive`` / ``EA`` need ``sqlalchemy`` + ``sqlmodel``,
and ``Individual`` needs ``sqlmodel`` too. In the Isaac Lab
integration env (built via the Phase 2.5 Option A recipe with
``pip install -e . --no-deps``), those packages are not installed
because the env's binary stack is owned by Isaac Lab and pip is told
to skip dependency resolution. Eagerly importing them from this
``__init__`` made every import under ``ariel.ec.*`` fail with
``ModuleNotFoundError: No module named 'sqlalchemy'`` in that env.
Lazy resolution keeps the package-level namespace API stable while
letting unrelated submodules import cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ariel.ec.archive import Archive
    from ariel.ec.crossover import Crossover
    from ariel.ec.ea import EA, DBHandlingMode, EAOperation, EASettings, config
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
    "Archive",
    "Crossover",
    "DBHandlingMode",
    "EA",
    "EAOperation",
    "EASettings",
    "FloatMutator",
    "Floats",
    "FloatsGenerator",
    "Individual",
    "IntegerMutator",
    "Integers",
    "IntegersGenerator",
    "JSONIterable",
    "JSONPrimitive",
    "JSONType",
    "Population",
    "SEED",
    "config",
]


# Map each lazily-exported name to its source submodule. Using a single
# table (instead of an `if`-cascade) keeps the dispatch readable and the
# set easy to audit against `__all__`.
_LAZY_IMPORTS: dict[str, str] = {
    "Archive": "ariel.ec.archive",
    "Crossover": "ariel.ec.crossover",
    "DBHandlingMode": "ariel.ec.ea",
    "EA": "ariel.ec.ea",
    "EAOperation": "ariel.ec.ea",
    "EASettings": "ariel.ec.ea",
    "FloatMutator": "ariel.ec.generators",
    "Floats": "ariel.ec.generators",
    "FloatsGenerator": "ariel.ec.generators",
    "Individual": "ariel.ec.individual",
    "IntegerMutator": "ariel.ec.generators",
    "Integers": "ariel.ec.generators",
    "IntegersGenerator": "ariel.ec.generators",
    "JSONIterable": "ariel.ec.individual",
    "JSONPrimitive": "ariel.ec.individual",
    "JSONType": "ariel.ec.individual",
    "Population": "ariel.ec.population",
    "SEED": "ariel.ec.generators",
    "config": "ariel.ec.ea",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib  # noqa: PLC0415
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
