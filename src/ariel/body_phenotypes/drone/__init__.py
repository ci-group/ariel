"""ARIEL drone body phenotype package.

Bridges airevolve's drone genome handlers and physics simulator into ARIEL's
EA engine. Genomes are stored as JSON in the SQLite database; airevolve's
custom ODE dynamics are used for fitness evaluation.

Symbols are exposed lazily via ``__getattr__`` so that importing a
submodule (for example ``ariel.body_phenotypes.drone.blueprint``) does
not eagerly drag in the EA operators or anything they transitively
require (sqlalchemy, sqlmodel, pydantic_settings, …). This keeps the
Blueprint / decoders / backends importable in lightweight envs (the
Isaac Lab integration env built via the Phase 2.5 Option A recipe is
the motivating case).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ariel.body_phenotypes.drone.genome import (
        deserialize_cppn_genome,
        deserialize_genome,
        serialize_cppn_genome,
        serialize_genome,
    )
    from ariel.body_phenotypes.drone.operations import (
        crossover_cppn_drones,
        crossover_drones,
        evaluate_drones,
        evaluate_drones_hover_mujoco,
        initialize_cppn_drones,
        initialize_drones,
        mutate_cppn_drones,
        mutate_drones,
        parent_tag,
        truncation_select,
    )

__all__ = [
    "crossover_cppn_drones",
    "crossover_drones",
    "deserialize_cppn_genome",
    "deserialize_genome",
    "evaluate_drones",
    "evaluate_drones_hover_mujoco",
    "initialize_cppn_drones",
    "initialize_drones",
    "mutate_cppn_drones",
    "mutate_drones",
    "parent_tag",
    "serialize_cppn_genome",
    "serialize_genome",
    "truncation_select",
]


# Map each lazily-exported name to its source submodule. Single table
# (instead of an `if`-cascade) keeps the dispatch easy to audit against
# ``__all__``.
_LAZY_IMPORTS: dict[str, str] = {
    "crossover_cppn_drones": "ariel.body_phenotypes.drone.operations",
    "crossover_drones": "ariel.body_phenotypes.drone.operations",
    "deserialize_cppn_genome": "ariel.body_phenotypes.drone.genome",
    "deserialize_genome": "ariel.body_phenotypes.drone.genome",
    "evaluate_drones": "ariel.body_phenotypes.drone.operations",
    "evaluate_drones_hover_mujoco": "ariel.body_phenotypes.drone.operations",
    "initialize_cppn_drones": "ariel.body_phenotypes.drone.operations",
    "initialize_drones": "ariel.body_phenotypes.drone.operations",
    "mutate_cppn_drones": "ariel.body_phenotypes.drone.operations",
    "mutate_drones": "ariel.body_phenotypes.drone.operations",
    "parent_tag": "ariel.body_phenotypes.drone.operations",
    "serialize_cppn_genome": "ariel.body_phenotypes.drone.genome",
    "serialize_genome": "ariel.body_phenotypes.drone.genome",
    "truncation_select": "ariel.body_phenotypes.drone.operations",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib  # noqa: PLC0415
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
