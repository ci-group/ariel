"""Pluggable simulator backends for the drone-gate-passing task.

This module exposes the ``BlueprintGateEnv`` *Protocol* — ariel's
contract for a simulator-backend-agnostic gymnasium ``VecEnv`` that
runs the gate-racing task on a drone described by a
:class:`~ariel.body_phenotypes.drone.blueprint.DroneBlueprint`.

The PPO trainer (and any other consumer) accepts *any* implementation
of this Protocol; collaborators can plug in their own simulator by
satisfying the contract. v1 ships:

* :class:`NumpyBlueprintGateEnv` — wraps the existing
  :class:`~ariel.simulation.tasks.drone_gate_env.DroneGateEnv`, which
  drives the pure-NumPy ``DroneSimulator``.
* :class:`~ariel.simulation.tasks.isaaclab_gate_env.IsaacLabBlueprintGateEnv`
  — stub (next phase). Will wrap Isaac Lab parallel envs spawned from
  a USD generated via ``blueprint_to_urdf``.

Usage:

    from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
    from ariel.simulation.tasks.blueprint_gate_env import NumpyBlueprintGateEnv

    bp = spherical_angular_to_blueprint(genome, propsize=5)
    env = NumpyBlueprintGateEnv(blueprint=bp, num_envs=64)
    # env is a gymnasium VecEnv — hand to PPO unchanged.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from stable_baselines3.common.vec_env import VecEnv

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv


# ---------- Protocol ----------

@runtime_checkable
class BlueprintGateEnv(Protocol):
    """Construction contract for simulator-backend-agnostic gate-racing envs.

    Implementations are gymnasium VecEnvs (stable-baselines3 style)
    constructed from a :class:`DroneBlueprint`. A PPO trainer treats
    any conforming instance as an opaque ``VecEnv`` — it never has to
    know which simulator backend produced it.

    A conforming implementation must:

    1. Accept a ``blueprint: DroneBlueprint`` argument at construction
       (alongside ``num_envs`` and any backend-specific kwargs).
    2. Expose ``self.blueprint`` and ``self.num_envs`` as public
       attributes.
    3. Implement the standard ``VecEnv`` methods (``reset``,
       ``step_async``, ``step_wait``, ``close``, ``get_attr``,
       ``set_attr``, ``env_method``, ``env_is_wrapped``) — most
       implementations inherit these from
       ``stable_baselines3.common.vec_env.VecEnv``.

    To add a new simulator backend, write a class that satisfies this
    Protocol; ariel's EA + PPO pipeline will accept it directly.
    """

    blueprint: DroneBlueprint
    num_envs: int


# ---------- v1 backend: NumPy DroneSimulator via existing DroneGateEnv ----------

class NumpyBlueprintGateEnv(DroneGateEnv):
    """Pluggable-Protocol wrapper around the existing NumPy-physics env.

    Composes ``blueprint_to_propellers`` (Blueprint → propellers list)
    with the existing :class:`DroneGateEnv` (propellers → VecEnv).
    Inheriting from ``DroneGateEnv`` is a deliberate choice: every
    VecEnv method, the gate definitions, the reward shaping, and the
    observation/action space are reused unchanged. The wrapper's only
    job is to be the Protocol-conforming entry point and stash
    ``self.blueprint``.

    Subclassing instead of composition means consumers can isinstance-
    check either ``NumpyBlueprintGateEnv`` or ``DroneGateEnv``, and
    ``VecEnv`` introspection (``env.num_envs``, ``env.observation_space``)
    works without forwarding boilerplate.
    """

    def __init__(
        self,
        *,
        blueprint: DroneBlueprint,
        num_envs: int,
        convention: str = "ned",
        **drone_gate_env_kwargs: Any,
    ) -> None:
        # `DroneSimulator` (the NumPy backend underneath DroneGateEnv) is
        # NED — z-down, Lee-controller convention. blueprint_to_propellers
        # supports both "z_up" (default) and "ned"; we ask for NED here so
        # the propellers list matches DroneSimulator's frame without an
        # extra inversion in the env layer.
        propellers = blueprint_to_propellers(blueprint, convention=convention)
        super().__init__(
            num_envs=num_envs,
            propellers=propellers,
            **drone_gate_env_kwargs,
        )
        self.blueprint = blueprint
        # Protocol also requires `num_envs`; DroneGateEnv -> VecEnv already
        # sets that, but reassigning makes the contract visible at this
        # class's source.
        self.num_envs = num_envs


__all__ = [
    "BlueprintGateEnv",
    "NumpyBlueprintGateEnv",
]
