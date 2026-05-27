"""Train a PPO policy through a pluggable simulator backend.

Demonstrates ariel's backend-agnostic EA + PPO loop: the same outer
training script accepts any backend that satisfies its plug point.
Each backend brings its own simulator AND its own RL library, because
that matches how real heterogeneous simulator ecosystems work:

  --simulator numpy      → NumpyBlueprintGateEnv (gymnasium VecEnv)
                           + stable-baselines3 PPO   (gate-passing task)

  --simulator isaaclab   → IsaacLabBlueprintHoverEnv (DirectRLEnv)
                           + random-action env stepping  (hover-to-goal task;
                                                         rl_games PPO wiring
                                                         pending Phase 2.5)

The morphology is built from a small set of presets via
``spherical_angular_to_blueprint``; the EA above this script never
sees the simulator choice — it just gets a fitness scalar back.

Examples:

    # NumPy backend, gate task, sb3 PPO (short smoke run).
    python tutorials/pluggable_simulator/train.py --simulator numpy \\
        --total-timesteps 5000 --num-envs 4

    # Isaac Lab backend, hover task, rsl_rl PPO (headless, short run).
    python tutorials/pluggable_simulator/train.py --simulator isaaclab \\
        --headless --max-iterations 20 --num-envs 64
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint


def _log(msg: str) -> None:
    """stderr-flushed progress messages (Isaac Sim sometimes swallows stdout)."""
    sys.stderr.write(f"[train] {msg}\n")
    sys.stderr.flush()


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


# ---------- shared base parser --------------------------------------------------

def _base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--simulator", choices=["numpy", "isaaclab"], default="numpy",
                   help="Which simulator backend (and matching RL library) to use.")
    p.add_argument("--preset", choices=list(PRESETS), default="quad")
    p.add_argument("--propsize", type=int, default=5)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    return p


# ---------- numpy backend: sb3 PPO + NumpyBlueprintGateEnv ----------------------

def main_numpy(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--total-timesteps", type=int, default=10_000,
                        help="sb3 PPO total environment steps.")
    args = parser.parse_args()

    from ariel.simulation.tasks.blueprint_gate_env import (
        BlueprintGateEnv,
        NumpyBlueprintGateEnv,
    )
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor

    print(f"=== pluggable-simulator demo: numpy + stable-baselines3 PPO ===")
    print(f"  preset / propsize : {args.preset} / {args.propsize}")
    print(f"  num_envs          : {args.num_envs}")
    print(f"  total_timesteps   : {args.total_timesteps}")

    blueprint = spherical_angular_to_blueprint(
        PRESETS[args.preset], propsize=args.propsize,
    )
    env = NumpyBlueprintGateEnv(
        blueprint=blueprint, num_envs=args.num_envs, seed=args.seed,
    )
    assert isinstance(env, BlueprintGateEnv), "Protocol mismatch"
    env = VecMonitor(env)

    model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=args.total_timesteps)
    dt = time.time() - t0

    print(f"\n=== training complete ===")
    print(f"  wall time : {dt:.1f} s")
    print(f"  steps/sec : {args.total_timesteps / dt:.0f}")


# ---------- isaaclab backend: rsl_rl PPO + DirectRLEnv --------------------------

def main_isaaclab(parser: argparse.ArgumentParser) -> None:
    # Add Isaac-Lab-specific args + AppLauncher args, then parse.
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="rl_games PPO max_epochs.")
    parser.add_argument("--device-override", type=str, default=None,
                        help="Override the auto-selected device "
                             "(e.g., 'cpu' or 'cuda:0').")

    # AppLauncher must be imported early so its args can be added to the
    # parser. Importing it triggers Isaac Sim's launcher setup; this is
    # the cost of going down the isaaclab path.
    from isaaclab.app import AppLauncher  # noqa: PLC0415
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.device_override is not None:
        args.device = args.device_override

    _log(f"=== pluggable-simulator demo: isaaclab + env-stepping smoke "
         f"(rl_games PPO wiring deferred to Phase 2.5; see DRONE_BLUEPRINT_PLAN.md §6 entry 17) ===")
    _log(f"  preset / propsize : {args.preset} / {args.propsize}")
    _log(f"  num_envs          : {args.num_envs}")
    _log(f"  max_iterations    : {args.max_iterations}")

    # Launch Isaac Sim. The env module imports isaaclab.* at module level,
    # so we delay the import until *after* the launcher is running.
    _log("launching Isaac Sim...")
    app_launcher = AppLauncher(args, multi_gpu=False)
    simulation_app = app_launcher.app
    _log("Isaac Sim launched")

    import traceback  # noqa: PLC0415
    try:
        _log("importing IsaacLabBlueprintHoverEnv...")
        from ariel.simulation.tasks.isaaclab_hover_env import (  # noqa: PLC0415
            IsaacLabBlueprintHoverEnv,
            IsaacLabBlueprintHoverEnvCfg,
        )
        import torch  # noqa: PLC0415

        _log("building blueprint...")
        blueprint = spherical_angular_to_blueprint(
            PRESETS[args.preset], propsize=args.propsize,
        )

        _log("building cfg from blueprint (generates URDF + USD)...")
        env_cfg = IsaacLabBlueprintHoverEnvCfg.from_blueprint(
            blueprint, num_envs=args.num_envs,
        )
        env_cfg.seed = args.seed
        env_cfg.sim.device = args.device

        _log(f"constructing env (num_envs={args.num_envs}, device={args.device})...")
        env = IsaacLabBlueprintHoverEnv(cfg=env_cfg)

        # Phase 2 demonstrates env construction + scene spawn + per-step
        # physics via a random-action stepping loop. Real PPO training
        # via `rl_games.torch_runner.Runner` is structurally wired (see
        # IsaacLabBlueprintHoverEnv's `make_rl_games_agent_cfg`) but
        # blocked on a Phase 2.5 follow-up: Isaac Sim's bundled torch
        # tensorboard transitively imports an older TF/jax/numpy stack
        # that conflicts with the conda env's numpy 2. See
        # DRONE_BLUEPRINT_PLAN.md §6 entry 17 for the full story.
        n_steps = max(1, args.max_iterations) * 24  # iterations × horizon
        _log(f"stepping env with random actions for {n_steps} steps "
             f"(env-construction smoke; PPO is Phase 2.5)...")

        env.reset()
        action_dim = env.cfg.action_space
        rewards_seen: list[float] = []
        t0 = time.time()
        for step in range(n_steps):
            actions = (torch.rand(args.num_envs, action_dim, device=args.device) * 2.0 - 1.0)
            _obs_dict, reward, _terminated, _truncated, _info = env.step(actions)
            rewards_seen.append(float(reward.mean().item()))
            if (step + 1) % 24 == 0:
                _log(f"  step {step+1:4d} | mean reward (last 24): "
                     f"{sum(rewards_seen[-24:])/24:.4f}")
        dt = time.time() - t0

        _log(f"=== env-stepping smoke complete ===")
        _log(f"  wall time      : {dt:.1f} s")
        _log(f"  steps          : {n_steps}")
        _log(f"  steps/sec      : {n_steps / dt:.0f}")
        _log(f"  mean reward    : {sum(rewards_seen) / len(rewards_seen):.4f}")
    except Exception as exc:
        _log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        _log("closing simulation app")
        simulation_app.close()


# ---------- entry point ---------------------------------------------------------

def main() -> None:
    # Peek at --simulator without committing to a fully-parsed namespace yet.
    # This lets us avoid importing isaaclab when the user only wants numpy.
    peek_parser = argparse.ArgumentParser(add_help=False)
    peek_parser.add_argument("--simulator", choices=["numpy", "isaaclab"], default="numpy")
    peek_args, _ = peek_parser.parse_known_args(sys.argv[1:])

    parser = _base_parser()
    if peek_args.simulator == "numpy":
        main_numpy(parser)
    elif peek_args.simulator == "isaaclab":
        main_isaaclab(parser)
    else:
        raise SystemExit(f"Unknown simulator: {peek_args.simulator!r}")


if __name__ == "__main__":
    main()
