"""Train a PPO policy through a pluggable simulator backend.

Demonstrates ariel's backend-agnostic EA + PPO loop: the same outer
training script accepts any backend that satisfies its plug point.
Each backend brings its own simulator AND its own RL library, because
that matches how real heterogeneous simulator ecosystems work:

  --simulator numpy      → NumpyBlueprintGateEnv (gymnasium VecEnv)
                           + stable-baselines3 PPO   (gate-passing task)

  --simulator isaaclab   → IsaacLabBlueprintHoverEnv (DirectRLEnv)
                           + rl_games PPO (default) or random-action env
                             stepping (with `--mode step`)  (hover-to-goal task)

The morphology is built from a small set of presets via
``spherical_angular_to_blueprint``; the EA above this script never
sees the simulator choice — it just gets a fitness scalar back.

Examples:

    # NumPy backend, gate task, sb3 PPO (short smoke run).
    python tutorials/pluggable_simulator/train.py --simulator numpy \\
        --total-timesteps 5000 --num-envs 4

    # Isaac Lab backend, hover task, rl_games PPO (headless, short run).
    # `--mode train` is the default; PPO trains for --max-iterations
    # epochs.
    python tutorials/pluggable_simulator/train.py --simulator isaaclab \\
        --headless --max-iterations 20 --num-envs 64

    # Same backend, but `--mode step` skips PPO and runs a random-action
    # env-stepping loop. Faster sanity check for env construction and
    # the Isaac-Lab-side env-stack.
    python tutorials/pluggable_simulator/train.py --simulator isaaclab \\
        --mode step --headless --max-iterations 3 --num-envs 16
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


# ---------- isaaclab backend: rl_games PPO + DirectRLEnv ------------------------

def _isaaclab_step_smoke(env, args) -> None:
    """Random-action stepping loop for fast env-construction validation.

    Useful when iterating on env code or env-stack health; avoids PPO's
    setup cost. The recipe's Option A acceptance prefers
    `--mode train` (real rl_games PPO), but `--mode step` remains the
    fastest sanity check.
    """
    import torch  # noqa: PLC0415

    n_steps = max(1, args.max_iterations) * 24  # iterations × horizon
    _log(f"stepping env with random actions for {n_steps} steps "
         f"(env-construction smoke)...")

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


def _isaaclab_rl_games_train(env, args) -> None:
    """Real PPO training via rl_games + Isaac Lab's adapter.

    Reaches the last Option A acceptance box: ``one short `rl_games` PPO
    smoke run completes in this env``. Mirrors Isaac Lab's official
    rl_games training pattern from
    ``IsaacLab/scripts/reinforcement_learning/rl_games/train.py``.
    """
    from ariel.simulation.tasks.isaaclab_hover_env import (  # noqa: PLC0415
        make_rl_games_agent_cfg,
    )
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: PLC0415
    from rl_games.common import env_configurations, vecenv  # noqa: PLC0415
    from rl_games.common.algo_observer import IsaacAlgoObserver  # noqa: PLC0415
    from rl_games.torch_runner import Runner  # noqa: PLC0415

    _log("building rl_games agent config...")
    agent_cfg_kwargs = dict(
        max_epochs=args.max_iterations,
        minibatch_size=24 * args.num_envs,
        device=args.device,
    )
    if args.experiment_prefix:
        agent_cfg_kwargs["experiment_name"] = args.experiment_prefix
    agent_cfg = make_rl_games_agent_cfg(**agent_cfg_kwargs)
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", float("inf"))
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", float("inf"))
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    _log("wrapping env for rl_games...")
    env = RlGamesVecEnvWrapper(
        env, args.device, clip_obs, clip_actions, obs_groups, concate_obs_groups,
    )

    _log("registering env in rl_games global registry...")
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(
            config_name, num_actors, **kwargs,
        ),
    )
    env_configurations.register(
        "rlgpu",
        {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env},
    )
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    _log("building rl_games Runner...")
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)
    runner.reset()

    _log(f"starting PPO training for max_epochs={args.max_iterations}...")
    t0 = time.time()
    runner.run({"train": True, "play": False, "sigma": None})
    dt = time.time() - t0

    _log(f"=== PPO training complete ===")
    _log(f"  wall time   : {dt:.1f} s")
    _log(f"  max_epochs  : {args.max_iterations}")


def main_isaaclab(parser: argparse.ArgumentParser) -> None:
    # Add Isaac-Lab-specific args + AppLauncher args, then parse.
    parser.add_argument("--mode", choices=["train", "step"], default="train",
                        help="`train` (default): run real rl_games PPO. "
                             "`step`: random-action env-stepping smoke "
                             "(faster, useful for env-construction sanity).")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="rl_games PPO max_epochs (--mode train) OR "
                             "number of 24-step horizons of random actions "
                             "(--mode step).")
    parser.add_argument("--device-override", type=str, default=None,
                        help="Override the auto-selected device "
                             "(e.g., 'cpu' or 'cuda:0').")
    parser.add_argument("--blueprint-json", type=str, default=None,
                        help="Path to a DroneBlueprint JSON file. When set, "
                             "the blueprint is loaded from disk and --preset "
                             "is ignored. Lets an outer EA loop hand in a "
                             "per-individual morphology.")
    parser.add_argument("--experiment-prefix", type=str, default=None,
                        help="rl_games experiment_name (also the runs/<X>_* "
                             "directory prefix). Set per individual when "
                             "driven by an outer EA loop so checkpoints are "
                             "discoverable.")

    # AppLauncher must be imported early so its args can be added to the
    # parser. Importing it triggers Isaac Sim's launcher setup; this is
    # the cost of going down the isaaclab path.
    from isaaclab.app import AppLauncher  # noqa: PLC0415
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if args.device_override is not None:
        args.device = args.device_override

    _log(f"=== pluggable-simulator demo: isaaclab + {args.mode} mode ===")
    _log(f"  preset / propsize : {args.preset} / {args.propsize}")
    _log(f"  num_envs          : {args.num_envs}")
    _log(f"  max_iterations    : {args.max_iterations}")

    # Launch Isaac Sim. The env module imports isaaclab.* at module level,
    # so we delay the import until *after* the launcher is running.
    _log("launching Isaac Sim...")
    app_launcher = AppLauncher(args, multi_gpu=False)
    simulation_app = app_launcher.app
    _log("Isaac Sim launched")

    import os  # noqa: PLC0415
    import traceback  # noqa: PLC0415
    exit_code = 0
    try:
        _log("importing IsaacLabBlueprintHoverEnv...")
        from ariel.simulation.tasks.isaaclab_hover_env import (  # noqa: PLC0415
            IsaacLabBlueprintHoverEnv,
            IsaacLabBlueprintHoverEnvCfg,
        )

        if args.blueprint_json:
            _log(f"loading blueprint from {args.blueprint_json}...")
            from ariel.body_phenotypes.drone.blueprint import (  # noqa: PLC0415
                DroneBlueprint,
            )
            blueprint = DroneBlueprint.load_json(args.blueprint_json)
        else:
            _log(f"building blueprint from preset {args.preset!r}...")
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

        if args.mode == "step":
            _isaaclab_step_smoke(env, args)
        elif args.mode == "train":
            _isaaclab_rl_games_train(env, args)
        else:
            raise SystemExit(f"unknown --mode {args.mode!r}")
    except Exception as exc:
        _log(f"ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc(file=sys.stderr)
        exit_code = 1

    # Skip simulation_app.close(): in current Isaac Sim builds it can
    # leave Isaac Sim's app threads spinning at ~120% CPU after a
    # successful run. Checkpoints are already on disk by this point,
    # so we exit hard via os._exit() to release control back to any
    # outer EA driver. See README §3c.
    _log(f"exiting (code {exit_code})")
    os._exit(exit_code)


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
