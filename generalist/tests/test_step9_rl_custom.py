"""Test custom RL loop with proper LSTM state management."""
from __future__ import annotations

import numpy as np
import torch

from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI
from generalist.policy import GeneralistDronePolicy
from generalist.trajectories import generate_circle
from generalist.training.rl_loop import RLLoop


def test_rl_loop_basic():
    """Test RLLoop instantiation and basic rollout."""
    phi = NOMINAL_PHI
    env = DroneTrackingEnv(phi=phi, seed=0)
    policy = GeneralistDronePolicy()

    loop = RLLoop(
        policy=policy,
        env=env,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=2,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Rollout
    obs, acts, rets, advs, vals = loop.rollout(128)

    assert obs.shape == (128, 73)
    assert acts.shape == (128, 4)
    assert rets.shape == (128,)
    assert advs.shape == (128,)
    assert vals.shape == (128,)
    print("✓ Rollout shapes correct")


def test_rl_loop_update():
    """Test PPO update step."""
    phi = NOMINAL_PHI
    env = DroneTrackingEnv(phi=phi, seed=0)
    policy = GeneralistDronePolicy()

    loop = RLLoop(
        policy=policy,
        env=env,
        learning_rate=1e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=2,
    )

    # Rollout and update
    obs, acts, rets, advs, vals = loop.rollout(128)
    stats = loop.update(obs, acts, rets, advs)

    assert "loss" in stats
    assert "mean_episode_return" in stats
    assert "mean_episode_length" in stats
    print(f"✓ Update successful: loss={stats['loss']:.4f}")


def test_rl_loop_train_short():
    """Test short training run (convergence check)."""
    phi = NOMINAL_PHI
    env = DroneTrackingEnv(phi=phi, seed=0)
    policy = GeneralistDronePolicy()

    loop = RLLoop(
        policy=policy,
        env=env,
        learning_rate=1e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=3,
    )

    # Train for 2 iterations (512 steps total)
    print("Training for 512 steps...")
    checkpoint_path = loop.train(
        total_timesteps=512,
        log_interval=256,
        save_dir="/tmp/test_rl_checkpoint",
    )

    assert checkpoint_path is not None
    assert "final_policy.pt" in checkpoint_path
    print(f"✓ Training completed, checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    test_rl_loop_basic()
    test_rl_loop_update()
    test_rl_loop_train_short()
    print("\n✓ All RL loop tests passed")
