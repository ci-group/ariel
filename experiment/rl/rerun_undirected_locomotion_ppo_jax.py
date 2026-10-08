"""Re-run the rollout from a prior ``undirected_locomotion_ppo_jax.py`` run.

Loads ``params.pkl`` saved by the training script, rebuilds the PPO policy
with the same network config, and renders a fresh rollout video + trajectory
plot. Useful for producing a new recording from a different seed or episode
length without retraining.

Must be run from the same ``mujoco_playground`` virtualenv as training.

Example
-------
``/home/user/Desktop/EvoDevo/mujoco_playground/.venv/bin/python \
    experiment/rerun_undirected_locomotion_ppo_jax.py \
    --run-dir __data__/undirected_locomotion_ppo_jax/insect_undirected-20261007-162432``
"""

from __future__ import annotations

# XLA/MuJoCo configuration — must happen before jax and brax import.
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import json
from datetime import datetime
from pathlib import Path

from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from undirected_locomotion_ppo_jax import (
    NACCDMAX_PER_ENV,
    NACONMAX_PER_ENV,
    UndirectedLocomotionInsect,
    default_config,
    render_rollout,
)


def _build_make_inference_fn(run_dir: Path):
    """Rebuild the PPO policy factory from the saved ppo_config.json."""
    ppo_cfg = json.loads((run_dir / "ppo_config.json").read_text())
    network_factory_kwargs = ppo_cfg["network_factory"]
    normalize_observations = ppo_cfg.get("normalize_observations", False)

    # Single-env sizing (matches render_rollout's internal env).
    rollout_cfg = default_config()
    rollout_cfg.naconmax = NACONMAX_PER_ENV
    rollout_cfg.naccdmax = NACCDMAX_PER_ENV
    env = UndirectedLocomotionInsect(rollout_cfg)

    normalize = running_statistics.normalize if normalize_observations else (lambda x, y: x)
    ppo_network = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=normalize,
        **network_factory_kwargs,
    )
    return ppo_networks.make_inference_fn(ppo_network)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Prior training-run directory containing params.pkl and ppo_config.json.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Where to write rollout.mp4 and trajectory.png. "
        "Defaults to <run-dir>/rerun-<timestamp>/.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    params_path = run_dir / "params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(
            f"No params.pkl in {run_dir}. "
            "The training script must have saved params for this run."
        )

    out_dir = (
        Path(args.outdir).expanduser().resolve()
        if args.outdir
        else run_dir / f"rerun-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing rollout artefacts to {out_dir}")

    make_inference_fn = _build_make_inference_fn(run_dir)
    params = model.load_params(str(params_path))
    render_rollout(make_inference_fn, params, out_dir, args.episode_length, args.seed)


if __name__ == "__main__":
    main()
