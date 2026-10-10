"""ARIEL simulation package.

Symbols are exposed lazily via ``__getattr__`` so that importing a
submodule (for example
``ariel.simulation.tasks.drone_gate_env``) does not eagerly pull in
heavy optional dependencies that the consumer may not need.

Concretely: ``mujoco_worker`` requires the ``mujoco`` Python package.
In the Isaac Lab integration env (built via the Phase 2.5 Option A
recipe with ``pip install -e . --no-deps``), ``mujoco`` is not
installed because the env's binary stack is owned by Isaac Lab.
Eagerly importing ``mujoco_worker`` from ``__init__`` made every
import under ``ariel.simulation.*`` fail with ``ModuleNotFoundError:
No module named 'mujoco'`` in that env. Lazy resolution keeps the
namespace API the same while letting unrelated submodules import
cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase

__all__ = ["EvalConfig", "MuJoCoWorkerBase"]


def __getattr__(name: str) -> Any:
    if name in ("EvalConfig", "MuJoCoWorkerBase"):
        from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase
        return {"EvalConfig": EvalConfig, "MuJoCoWorkerBase": MuJoCoWorkerBase}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
