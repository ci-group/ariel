"""Isaac Lab backend for the drone-gate-passing task — *stub*.

Implementation deferred to Phase 2 of the pluggable-simulator effort.
The Protocol contract lives in
:mod:`ariel.simulation.tasks.blueprint_gate_env`; this module reserves
the import path and pins the planned API so collaborators can see the
shape from both sides of the seam.

Phase 2 will:

1. Convert the blueprint to URDF in-process via
   ``ariel.body_phenotypes.drone.backends.blueprint_to_urdf``.
2. Run ``isaaclab.sim.converters.UrdfConverter`` to produce a USD
   (no subprocess — both halves live in the unified ariel/isaaclab
   conda env now; see DRONE_BLUEPRINT_PLAN.md §6 entry 15).
3. Spawn ``num_envs`` parallel articulation instances in Isaac Sim.
4. Apply per-motor thrust each step using a first-order motor model
   lifted from
   ``soft_airframe_optimization/src/morphy_simulator.py``
   (matched to the NumPy backend's reference dynamics so trained
   policies have a chance at sim-to-sim transfer).
5. Spawn gate prims at the same positions as the NumPy env so the
   task reward shaping is identical.
6. Return observations / rewards / dones in the standard VecEnv
   format, satisfying the
   :class:`~ariel.simulation.tasks.blueprint_gate_env.BlueprintGateEnv`
   Protocol.
"""
from __future__ import annotations

from typing import Any

from stable_baselines3.common.vec_env import VecEnv

from ariel.body_phenotypes.drone.blueprint import DroneBlueprint


class IsaacLabBlueprintGateEnv(VecEnv):
    """Isaac Lab gate-passing env — *not yet implemented*.

    Carries the planned constructor signature so the call site in the
    tutorial (and any future production callers) compiles against the
    intended API. Calling it raises ``NotImplementedError`` until
    Phase 2 lands.
    """

    blueprint: DroneBlueprint
    num_envs: int

    def __init__(
        self,
        *,
        blueprint: DroneBlueprint,
        num_envs: int,
        headless: bool = True,
        device: str = "cuda:0",
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError(
            "IsaacLabBlueprintGateEnv is the Phase 2 deliverable of the "
            "pluggable-simulator effort. Use NumpyBlueprintGateEnv for now."
        )

    # The abstract VecEnv methods would be implemented in Phase 2; we
    # leave them as stubs so the class can at least be imported.
    def reset(self):  # type: ignore[override]
        raise NotImplementedError

    def step_async(self, actions):  # type: ignore[override]
        raise NotImplementedError

    def step_wait(self):  # type: ignore[override]
        raise NotImplementedError

    def close(self) -> None:  # type: ignore[override]
        raise NotImplementedError

    def get_attr(self, attr_name, indices=None):  # type: ignore[override]
        raise NotImplementedError

    def set_attr(self, attr_name, value, indices=None):  # type: ignore[override]
        raise NotImplementedError

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):  # type: ignore[override]
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):  # type: ignore[override]
        raise NotImplementedError


__all__ = ["IsaacLabBlueprintGateEnv"]
