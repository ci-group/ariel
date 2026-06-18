"""ARIEL simulation package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ariel.simulation.mujoco_worker import EvalConfig, MuJoCoWorkerBase

__all__ = ["EvalConfig", "MuJoCoWorkerBase"]


def __getattr__(name: str):
    if name in ("EvalConfig", "MuJoCoWorkerBase"):
        from ariel.simulation import mujoco_worker
        return getattr(mujoco_worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
