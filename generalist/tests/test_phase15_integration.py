"""Integration test for Phase 1.5: RL loop + MuJoCo visualization."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from generalist.envs import DroneTrackingEnv
from generalist.morphology.genome import NOMINAL_PHI
from generalist.policy import GeneralistDronePolicy
from generalist.trajectories import generate_circle
from generalist.training.rl_loop import RLLoop


def test_phase15_rl_to_visualization():
    """Test full Phase 1.5: train RL policy and prepare for visualization."""
    # Create temporary directory for checkpoints
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # ---- Train RL policy (short run) ----
        print("Step 1: Training RL policy...")
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

        checkpoint_path = loop.train(
            total_timesteps=1024,
            log_interval=512,
            save_dir=tmpdir,
        )
        print(f"✓ RL training complete: {checkpoint_path}")

        # ---- Verify checkpoint ----
        print("\nStep 2: Verifying checkpoint...")
        assert Path(checkpoint_path).exists()
        print(f"✓ Checkpoint exists: {checkpoint_path}")

        # ---- Load policy and do rollout ----
        print("\nStep 3: Loading policy and doing rollout...")
        policy2 = GeneralistDronePolicy()
        policy2.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        policy2.eval()

        env2 = DroneTrackingEnv(phi=phi, seed=1)
        reference = generate_circle(radius=2.0, speed=2.0, altitude=1.5, dt=0.02, T_total=5.0)
        obs, _ = env2.reset(options={"reference": reference})

        lstm_states = None
        total_reward = 0.0
        ep_length = 0
        for t in range(min(len(reference) - 1, 100)):
            obs_t = torch.from_numpy(obs[np.newaxis, :]).float()
            with torch.no_grad():
                action_mean, _, lstm_states = policy2(obs_t, lstm_states=lstm_states)
                action = np.tanh(action_mean.squeeze(0).numpy())

            obs, reward, term, trunc, info = env2.step(action.astype(np.float32))
            total_reward += reward
            ep_length += 1

            if term or trunc:
                lstm_states = None
                break

        print(f"✓ Rollout complete: {ep_length} steps, total_reward={total_reward:.2f}")

        # ---- Verify policy can be saved/loaded for visualization ----
        print("\nStep 4: Verifying policy can be used in visualization...")
        # Just verify the checkpoint path can be used
        checkpoint_str = str(checkpoint_path)
        assert checkpoint_str.endswith("final_policy.pt")
        print(f"✓ Checkpoint path suitable for visualization: {checkpoint_str}")


if __name__ == "__main__":
    test_phase15_rl_to_visualization()
    print("\n✓ Phase 1.5 integration test passed")
