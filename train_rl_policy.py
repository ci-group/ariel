#!/usr/bin/env python3
"""Simple RL training script with parametrizable run length.

Usage:
    uv run python train_rl_policy.py --steps 10000
    uv run python train_rl_policy.py --steps 50000 --morphology random --seed 42
    uv run python train_rl_policy.py --steps 5000 --batch-size 32 --n-epochs 2
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI, sample_valid_morphology
from generalist.policy import GeneralistDronePolicy
from generalist.training.rl_loop import RLLoop


def main():
    parser = argparse.ArgumentParser(
        description="Train generalist drone controller with custom RL loop"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Total training steps (default: 10000)",
    )
    parser.add_argument(
        "--morphology",
        choices=["nominal", "random"],
        default="nominal",
        help="Morphology type (default: nominal)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for morphology/env (default: 0)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=256,
        help="Rollout length per iteration (default: 256)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Minibatch size (default: 64)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Number of epochs per update (default: 3)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE smoothing (default: 0.95)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO clip range (default: 0.2)",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.001,
        help="Entropy coefficient (default: 0.001)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Checkpoint save directory (default: auto)",
    )

    args = parser.parse_args()

    # ---- Setup ----
    print("\n" + "=" * 70)
    print("GENERALIST DRONE CONTROLLER — RL TRAINING")
    print("=" * 70)

    # Morphology
    print(f"\n📦 Morphology: {args.morphology}")
    if args.morphology == "nominal":
        phi = NOMINAL_PHI
    else:
        rng = np.random.default_rng(args.seed)
        phi = sample_valid_morphology(rng)
    print(f"   φ shape: {phi.shape}")

    # Environment
    print(f"\n🌍 Environment")
    env = DroneTrackingEnv(phi=phi, seed=args.seed)
    print(f"   Observation dim: {env.observation_space.shape}")
    print(f"   Action dim: {env.action_space.shape}")

    # Policy
    print(f"\n🧠 Policy: GeneralistDronePolicy")
    policy = GeneralistDronePolicy()
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"   Parameters: {total_params:,}")

    # Training config
    print(f"\n⚙️  Training Configuration")
    print(f"   Total steps: {args.steps:,}")
    print(f"   Rollout length: {args.n_steps}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Epochs per update: {args.n_epochs}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Discount (γ): {args.gamma}")
    print(f"   GAE λ: {args.gae_lambda}")
    print(f"   PPO clip range: {args.clip_range}")
    print(f"   Entropy coef: {args.ent_coef}")

    # Checkpoint directory
    if args.save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        morph_str = args.morphology[:3]  # "nom" or "ran"
        save_dir = f"checkpoints/rl_{morph_str}_{timestamp}"
    else:
        save_dir = args.save_dir

    print(f"   Save directory: {save_dir}")

    # ---- Create RL Loop ----
    print(f"\n🚀 Creating RLLoop...")
    loop = RLLoop(
        policy=policy,
        env=env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
    )
    print(f"✓ RLLoop created")

    # ---- Train ----
    print(f"\n{'=' * 70}")
    print(f"TRAINING IN PROGRESS")
    print(f"{'=' * 70}\n")

    checkpoint_path = loop.train(
        total_timesteps=args.steps,
        log_interval=max(args.n_steps, 256),
        save_dir=save_dir,
    )

    # ---- Summary ----
    print(f"\n{'=' * 70}")
    print(f"TRAINING COMPLETE ✓")
    print(f"{'=' * 70}")
    print(f"\n📊 Results:")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Mean episode return: {np.mean(loop.episode_returns):.2f}")
    print(f"   Mean episode length: {np.mean(loop.episode_lengths):.0f}")

    # Visualize command
    print(f"\n🎬 To visualize the trained policy:")
    print(f"   PYTHONPATH=. uv run python generalist/examples/run_and_visualize_mujoco.py \\")
    print(f"     --rl-checkpoint {checkpoint_path} \\")
    print(f"     --duration 10.0 \\")
    print(f"     --morphology {args.morphology}")

    print(f"\n✨ Done!")


if __name__ == "__main__":
    main()
