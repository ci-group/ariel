"""Types of robotic controllers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ariel.simulation.controllers.cmaes_learner import CMAESLearner
from ariel.simulation.controllers.distributed_mlp import (
    EMPTY_NODE,
    DistributedMLP,
    NodeObservation,
    StandardMLP,
)

if TYPE_CHECKING:
    from ariel.simulation.controllers.morphology_adapter import MorphologyAdapter
    from ariel.simulation.controllers.na_cpg import NaCPG

__all__ = [
    "CMAESLearner",
    "DistributedMLP",
    "EMPTY_NODE",
    "MorphologyAdapter",
    "NaCPG",
    "NodeObservation",
    "StandardMLP",
]

_LAZY = {
    "MorphologyAdapter": "ariel.simulation.controllers.morphology_adapter",
    "NaCPG": "ariel.simulation.controllers.na_cpg",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
