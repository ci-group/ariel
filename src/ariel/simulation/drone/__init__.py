"""ARIEL drone simulation package.

Symbols are exposed lazily via ``__getattr__`` so that importing a
submodule (for example ``ariel.simulation.drone.propeller_data``,
which is pure NumPy and has no heavy deps) does not eagerly pull in
``DroneSimulator`` (which needs ``sympy``) or ``DroneInterface``.

Motivating case: the Phase 2.5 Option A Isaac Lab integration env
installs ariel with ``pip install -e . --no-deps``, so sympy is not
present. Touching just ``propeller_data`` would otherwise fail with
``ModuleNotFoundError: No module named 'sympy'`` via the eager
``DroneSimulator`` import. Lazy resolution keeps the package-level
namespace API stable while letting unrelated submodules import
cleanly.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ariel.simulation.drone.drone_interface import DroneInterface
    from ariel.simulation.drone.drone_simulator import (
        DroneSimulator,
        create_hexarotor,
        create_octorotor,
        create_quadrotor,
        create_tricopter,
    )
    from ariel.simulation.drone.propeller_data import (
        GRAVITY,
        create_standard_propeller_config,
        get_extended_prop_params,
    )

__all__ = [
    "DroneInterface",
    "DroneSimulator",
    "GRAVITY",
    "create_hexarotor",
    "create_octorotor",
    "create_quadrotor",
    "create_standard_propeller_config",
    "create_tricopter",
    "get_extended_prop_params",
]


_LAZY_IMPORTS: dict[str, str] = {
    "DroneInterface": "ariel.simulation.drone.drone_interface",
    "DroneSimulator": "ariel.simulation.drone.drone_simulator",
    "GRAVITY": "ariel.simulation.drone.propeller_data",
    "create_hexarotor": "ariel.simulation.drone.drone_simulator",
    "create_octorotor": "ariel.simulation.drone.drone_simulator",
    "create_quadrotor": "ariel.simulation.drone.drone_simulator",
    "create_standard_propeller_config": "ariel.simulation.drone.propeller_data",
    "create_tricopter": "ariel.simulation.drone.drone_simulator",
    "get_extended_prop_params": "ariel.simulation.drone.propeller_data",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib  # noqa: PLC0415
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
