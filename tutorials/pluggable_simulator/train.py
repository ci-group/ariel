"""Train a PPO gate-passing policy through a pluggable simulator backend.

Demonstrates ariel's backend-agnostic EA+PPO loop: the same trainer
runs against any backend that satisfies the
:class:`~ariel.simulation.tasks.blueprint_gate_env.BlueprintGateEnv`
Protocol. Today there are two backends:

  --simulator numpy      → NumpyBlueprintGateEnv     (shipped)
  --simulator isaaclab   → IsaacLabBlueprintGateEnv  (stub; Phase 2)

The morphology is built from a small set of presets via
``spherical_angular_to_blueprint``; the PPO training loop is identical
across backends.

Examples:

    # Short smoke run on the NumPy backend (a few thousand steps).
    python tutorials/pluggable_simulator/train.py --simulator numpy \\
        --total-timesteps 5000 --num-envs 4

    # Full Isaac Lab path (not yet implemented).
    python tutorials/pluggable_simulator/train.py --simulator isaaclab \\
        --total-timesteps 50000 --num-envs 64
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.tasks.blueprint_gate_env import (
    BlueprintGateEnv,
    NumpyBlueprintGateEnv,
)


# ---------- preset morphologies -------------------------------------------------

PRESETS = {
    "quad": np.array([
        [0.18, 0.0,            0.0, 0.0, 0.0, 1.0],
        [0.18, np.pi / 2.0,    0.0, 0.0, 0.0, 0.0],
        [0.18, np.pi,          0.0, 0.0, 0.0, 1.0],
        [0.18, 3 * np.pi / 2,  0.0, 0.0, 0.0, 0.0],
    ]),
    "hex": np.array([
        [0.18, i * np.pi / 3.0, 0.0, 0.0, 0.0, float(i % 2)]
        for i in range(6)
    ]),
}


# ---------- backend dispatch ----------------------------------------------------

def make_env(simulator: str, blueprint, num_envs: int, **kwargs) -> BlueprintGateEnv:
    """Return a BlueprintGateEnv for the requested backend.

    Adding a new simulator means adding one branch here and one
    implementation file under ``ariel.simulation.tasks`` that
    satisfies the BlueprintGateEnv Protocol.
    """
    if simulator == "numpy":
        return NumpyBlueprintGateEnv(blueprint=blueprint, num_envs=num_envs, **kwargs)
    if simulator == "isaaclab":
        from ariel.simulation.tasks.isaaclab_gate_env import IsaacLabBlueprintGateEnv
        return IsaacLabBlueprintGateEnv(blueprint=blueprint, num_envs=num_envs, **kwargs)
    raise ValueError(
        f"Unknown simulator {simulator!r}. Known: 'numpy', 'isaaclab'."
    )


# ---------- main ----------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--simulator", choices=["numpy", "isaaclab"], default="numpy",
                        help="Which simulator backend to plug behind the PPO loop.")
    parser.add_argument("--preset", choices=list(PRESETS), default="quad",
                        help="Built-in morphology to evolve a policy on.")
    parser.add_argument("--propsize", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=8,
                        help="Vectorized parallel envs.")
    parser.add_argument("--total-timesteps", type=int, default=10_000,
                        help="PPO total environment steps (a few thousand for "
                             "a smoke test; tens of millions for actual training).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"=== pluggable-simulator demo ===")
    print(f"  simulator        : {args.simulator}")
    print(f"  preset / propsize: {args.preset} / {args.propsize}")
    print(f"  num_envs         : {args.num_envs}")
    print(f"  total_timesteps  : {args.total_timesteps}")

    blueprint = spherical_angular_to_blueprint(
        PRESETS[args.preset], propsize=args.propsize,
    )
    print(f"  blueprint        : {blueprint.g.number_of_nodes()} nodes")

    env = make_env(
        args.simulator, blueprint=blueprint, num_envs=args.num_envs, seed=args.seed,
    )
    # Sanity check the Protocol — runtime_checkable lets us assert here.
    assert isinstance(env, BlueprintGateEnv), (
        f"{type(env).__name__} does not satisfy BlueprintGateEnv Protocol"
    )
    env = VecMonitor(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps)
    dt = time.time() - t0

    print(f"\n=== training complete ===")
    print(f"  wall time : {dt:.1f} s")
    print(f"  steps     : {args.total_timesteps}")
    print(f"  steps/sec : {args.total_timesteps / dt:.0f}")


if __name__ == "__main__":
    main()
