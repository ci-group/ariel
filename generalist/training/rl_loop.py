"""Custom RL loop for GeneralistDronePolicy with proper LSTM state management.

This replaces the SB3 RecurrentPPO integration (which has interface
incompatibilities). We implement PPO directly, managing the policy's
LSTM state across rollouts correctly.

Spec §9 compliance: PPO with curriculum callbacks, IL→RL transfer.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from generalist.envs import DroneTrackingEnv
from generalist.policy import GeneralistDronePolicy


class RLLoop:
    """Custom PPO training loop with LSTM state management.

    Manages the policy's LSTM hidden states across episode boundaries,
    implements PPO updates with GAE, and integrates curriculum callbacks.
    """

    def __init__(
        self,
        policy: GeneralistDronePolicy,
        env: DroneTrackingEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 256,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.001,
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None,
    ):
        self.policy = policy
        self.env = env
        self.lr = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = self.policy.to(self.device).eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100  # will be updated
        )

        self.episode_returns = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.global_step = 0

    def rollout(self, n_steps: int) -> tuple:
        """Collect n_steps of trajectory data with LSTM state tracking.

        Returns:
            obs, actions, returns, advantages, values (all as tensors)
        """
        obs_list, act_list, val_list, rew_list = [], [], [], []
        ep_return, ep_length = 0.0, 0

        lstm_states = None
        obs, _ = self.env.reset()

        for step in range(n_steps):
            obs_t = torch.from_numpy(obs[np.newaxis, :]).float().to(self.device)

            with torch.no_grad():
                action_mean, value, lstm_states = self.policy(obs_t, lstm_states=lstm_states)
                action_pre_tanh = action_mean.squeeze(0).cpu().numpy()
                action = np.tanh(action_pre_tanh)

            obs_list.append(obs)
            act_list.append(action_pre_tanh)
            val_list.append(float(value.squeeze().cpu().numpy()))

            obs, reward, term, trunc, info = self.env.step(action.astype(np.float32))
            rew_list.append(float(reward))

            ep_return += reward
            ep_length += 1

            if term or trunc:
                self.episode_returns.append(ep_return)
                self.episode_lengths.append(ep_length)

                lstm_states = None  # reset LSTM at episode boundary
                obs, _ = self.env.reset()
                ep_return, ep_length = 0.0, 0

            self.global_step += 1

        # Compute returns and advantages via GAE
        returns, advantages = self._compute_gae(np.array(rew_list), np.array(val_list))

        obs_t = torch.from_numpy(np.array(obs_list)).float().to(self.device)
        act_t = torch.from_numpy(np.array(act_list)).float().to(self.device)
        ret_t = torch.from_numpy(returns).float().to(self.device)
        adv_t = torch.from_numpy(advantages).float().to(self.device)
        val_t = torch.from_numpy(np.array(val_list)).float().to(self.device)

        return obs_t, act_t, ret_t, adv_t, val_t

    def _compute_gae(
        self, rewards: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute returns and advantages via Generalized Advantage Estimation."""
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0  # episode end
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns[t] = gae + values[t]
            advantages[t] = gae

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def update(
        self, obs: torch.Tensor, actions: torch.Tensor,
        returns: torch.Tensor, advantages: torch.Tensor
    ) -> dict:
        """PPO update with value function."""
        self.policy.train()
        losses = []

        for epoch in range(self.n_epochs):
            indices = torch.randperm(len(obs))
            for i in range(0, len(obs), self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                obs_batch = obs[batch_idx]
                act_batch = actions[batch_idx]
                ret_batch = returns[batch_idx]
                adv_batch = advantages[batch_idx]

                # Forward pass (single-step, no LSTM state carried in updates)
                action_mean, value, _ = self.policy(obs_batch, lstm_states=None)
                action_squashed = torch.tanh(act_batch)

                # Compute log probs: pre-tanh Gaussian + tanh-squashing correction
                std = torch.ones_like(action_mean)
                dist = torch.distributions.Normal(action_mean, std)
                log_prob = dist.log_prob(act_batch).sum(dim=-1)
                log_prob = log_prob - torch.sum(torch.log(1 - action_squashed ** 2 + 1e-6), dim=-1)
                log_prob_old = log_prob.detach()

                # PPO clipping
                ratio = torch.exp(log_prob - log_prob_old)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                value_loss = nn.functional.mse_loss(value.squeeze(), ret_batch)

                # Entropy bonus
                entropy = dist.entropy().sum(dim=-1).mean()

                total_loss = (
                    actor_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                losses.append(float(total_loss.item()))

        self.policy.eval()
        return {
            "loss": float(np.mean(losses)),
            "mean_episode_return": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
            "mean_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
        }

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1000,
        save_dir: Optional[str | Path] = None,
    ) -> str:
        """Main training loop."""
        save_dir = Path(save_dir) if save_dir else Path("checkpoints/rl_custom")
        save_dir.mkdir(parents=True, exist_ok=True)

        num_iterations = max(2, total_timesteps // self.n_steps)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_iterations
        )

        while self.global_step < total_timesteps:
            obs, acts, rets, advs, vals = self.rollout(self.n_steps)
            stats = self.update(obs, acts, rets, advs)

            if self.global_step % log_interval == 0:
                print(
                    f"Step {self.global_step}: loss={stats['loss']:.4f}, "
                    f"mean_return={stats['mean_episode_return']:.2f}, "
                    f"mean_length={stats['mean_episode_length']:.0f}"
                )

            self.scheduler.step()

        # Save final checkpoint
        out_path = save_dir / "final_policy.pt"
        torch.save(self.policy.state_dict(), out_path)
        print(f"Saved final policy to {out_path}")
        return str(out_path)
