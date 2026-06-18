"""MTRL v4 for hexacopter — six changes over v3 to fix the slow,
risk-averse policy revealed by the 80M v3 eval:

    1. VELOCITY-TOWARD-GATE REWARD. v3 only had telescoping distance
       reward, so slow approach == fast approach. v4 adds an explicit
       +α·max(0, v·dir_to_gate) per step via the env's new
       `velocity_reward_coef` kwarg. Expected impact: slalom forward
       speed 0.5 → 2-3 m/s.

    2. PER-TASK GATE DENSITY. v3 densified figure8 from 8 to 24 gates,
       which made yaw-transitions awkward and the small loop confusing.
       v4 keeps figure8 at density 1 (8 original gates) while still
       densifying shuttle-run (4 → 8 after dedup).

    3. RELAXED TILT TERMINATION. cos threshold 0.25 (75°) → 0.10 (≈85°),
       matching the forest task. Allows pitch-forward racing dives.

    4. FIXED-START FOR FIGURE8. The tight figure-8 loop benefits from a
       canonical start direction; random-gate init had the drone learn
       nothing committal. Slalom / shuttle-run keep random init.

    5. SOFT ALTITUDE FLOOR. v3's shuttle-run policy was dipping to within
       8 cm of the ground. v4 enables the env's `altitude_floor_coef`
       which softly punishes z > -0.5 NED (below 0.5 m altitude).

    6. STRONGER HOVER XY-PULL. v3's hover policy drifted y by 1.2 m.
       v4 doubles the pos_xy coefficient in `_hover_reward` (0.2 → 0.4).

Run:
    uv run examples/spear/27_train_rl_hex_mtrl_v4.py --device cuda:0
    uv run examples/spear/27_train_rl_hex_mtrl_v4.py --steps 80000000

NOTE: do NOT add `from __future__ import annotations` to this file.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv


TASK_NAMES = ("figure8", "slalom", "shuttle-run", "hover")
NUM_TASKS = len(TASK_NAMES)
HOVER_ID = TASK_NAMES.index("hover")
GATES_AHEAD = 2
NUM_MOTORS = 6
SHARED_OBS_DIM = 12 + NUM_MOTORS       # 18
TASK_OBS_DIM = 4 * GATES_AHEAD         # 8
BASE_OBS_DIM = SHARED_OBS_DIM + TASK_OBS_DIM   # 26
FULL_OBS_DIM = BASE_OBS_DIM + NUM_TASKS        # 30
ENCODER_LATENT = 32
ENCODER_HIDDEN = 128
ACTOR_HIDDEN = 256

# Env shaping coefficients (tuned so a clean racing episode and a clean
# hover episode have comparable episodic return ~ +15 each).
UPRIGHT_BONUS = 0.01           # per step; 1200 steps × 0.01 ≈ +12 max
TILT_TERMINATE_COS = 0.10      # v4: ≈85° (was 0.25 in v3) — allow racing dives
EXTRA_YAW_RATE_PEN = 0.005     # on |ω_z|
VELOCITY_REWARD_COEF = 0.005   # v4: +α·max(0, v·dir_to_gate) per step
ALTITUDE_FLOOR_Z = -0.5        # v4: NED, soft floor at 0.5 m altitude
ALTITUDE_FLOOR_COEF = 0.5      # v4: penalty slope below the floor


class _HoverGateConfig:
    """Single stationary 'gate' at hover target. Reward is overridden in the
    multi-task wrapper to use a hover-specific shape."""
    gate_pos = np.array([[0.0, 0.0, -1.5]], dtype=np.float64)
    gate_yaw = np.array([0.0], dtype=np.float64)
    x_bounds = np.array([-3.0, 3.0], dtype=np.float64)
    y_bounds = np.array([-3.0, 3.0], dtype=np.float64)
    z_bounds = np.array([-3.0, 0.5], dtype=np.float64)
    starting_pos = np.array([0.0, 0.0, -1.5], dtype=np.float64)


HOVER_TARGET_NED = torch.tensor([0.0, 0.0, -1.5], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Gate densification
# ─────────────────────────────────────────────────────────────────────────────

def _densify_gates(gate_pos, gate_yaw, factor: int, periodic: bool,
                   min_segment: float = 0.5):
    """Insert (factor-1) interpolated gates between each consecutive pair.

    Segments shorter than `min_segment` are skipped (avoids duplicating the
    same xy position when the original config has back-to-back gates with
    matching coords, e.g. shuttle-run's [2,0,0]→[8,0,0]→[8,0,0]).

    Yaw interpolation uses shortest-angle path. Returns (new_pos, new_yaw).
    """
    if factor <= 1:
        return gate_pos.copy(), gate_yaw.copy()
    pos = np.asarray(gate_pos, dtype=np.float64)
    yaw = np.asarray(gate_yaw, dtype=np.float64)
    n = len(pos)
    out_pos, out_yaw = [], []
    last_idx = n if periodic else n - 1
    for i in range(n):
        out_pos.append(pos[i])
        out_yaw.append(yaw[i])
        if i >= last_idx:
            continue
        j = (i + 1) % n
        seg = pos[j] - pos[i]
        if np.linalg.norm(seg) < min_segment:
            continue
        dyaw = (yaw[j] - yaw[i] + math.pi) % (2 * math.pi) - math.pi
        for k in range(1, factor):
            t = k / factor
            out_pos.append(pos[i] + t * seg)
            out_yaw.append(yaw[i] + t * dyaw)
    return np.asarray(out_pos, dtype=np.float64), np.asarray(out_yaw, dtype=np.float64)


def _task_config(name: str, density: int = 1):
    """Returns (gate_pos, gate_yaw, starting_pos) with optional densification.

    v4: figure8 is pinned to density=1 because densified figure8 had
    awkward intermediate yaws that confused the policy. Shuttle-run still
    obeys `density` because its 4-gate base is too sparse. Slalom is
    always density=1 (it already has 100 gates).
    """
    if name == "figure8":
        cfg = GATE_CONFIGS["figure8"]
        pos, yaw = _densify_gates(cfg.gate_pos, cfg.gate_yaw, 1, True)   # v4: always 1
        return pos, yaw, np.asarray(cfg.starting_pos, dtype=np.float64)
    if name == "slalom":
        cfg = GATE_CONFIGS["slalom"]
        pos, yaw = _densify_gates(cfg.gate_pos, cfg.gate_yaw, 1, False)
        return pos, yaw, np.asarray(cfg.starting_pos, dtype=np.float64)
    if name == "shuttle-run":
        cfg = GATE_CONFIGS["backandforth"]
        pos, yaw = _densify_gates(cfg.gate_pos, cfg.gate_yaw, density, True)
        return pos, yaw, np.asarray(cfg.starting_pos, dtype=np.float64)
    if name == "hover":
        cfg = _HoverGateConfig
        return (np.asarray(cfg.gate_pos, dtype=np.float64),
                np.asarray(cfg.gate_yaw, dtype=np.float64),
                np.asarray(cfg.starting_pos, dtype=np.float64))
    raise KeyError(name)


def build_hex_genome(arm_length: float = 0.2) -> np.ndarray:
    az = np.arange(6) * (np.pi / 3.0)
    g = np.zeros((6, 6), dtype=np.float64)
    g[:, 0] = arm_length
    g[:, 1] = az
    g[:, 3] = az
    g[:, 5] = np.array([1, 0, 1, 0, 1, 0])
    return g


# ─────────────────────────────────────────────────────────────────────────────
# Per-task reward helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hover_reward(world_states: torch.Tensor, target: torch.Tensor,
                  diverged: torch.Tensor) -> torch.Tensor:
    """Hover reward with an explicit upright term. Magnitude-matched so a
    clean 1200-step hover yields ~+15, similar to a 5-gate racing episode."""
    pos = world_states[:, 0:3]
    vel = world_states[:, 3:6]
    phi = world_states[:, 6]
    theta = world_states[:, 7]
    rates = world_states[:, 9:12]
    pos_err = pos - target.unsqueeze(0)

    dist_xy = torch.norm(pos_err[:, :2], dim=1)
    dist_z = torch.abs(pos_err[:, 2])
    pos_xy = torch.exp(-0.5 * dist_xy)
    pos_z = torch.exp(-2.5 * dist_z)

    roll_pitch_rates = torch.norm(rates[:, :2], dim=1)
    yaw_rate = rates[:, 2].abs()
    rate_term = torch.exp(-0.5 * roll_pitch_rates) * torch.exp(-0.5 * yaw_rate)

    downward_vel = vel[:, 2]
    climb_term = torch.where(
        downward_vel > 0.0, torch.exp(-1.5 * downward_vel), torch.ones_like(downward_vel)
    )

    upright = (phi.cos() * theta.cos()).clamp(min=0.0)

    # v4: pos_xy 0.2 → 0.4 to cure the 1.2 m y-drift seen in the v3 eval.
    # Other coefficients rebalanced to keep total ≈ 1.25 (so per-step
    # max ≈ 0.0125 and a 1200-step clean hover still yields ~+15).
    rew = ((pos_z * 0.30) + (pos_xy * 0.40) + (rate_term * 0.10)
           + (climb_term * 0.05) + (upright * 0.15)) * 0.0125
    rew = torch.where(diverged, torch.full_like(rew, -1.0), rew)
    return rew


# ─────────────────────────────────────────────────────────────────────────────
# Per-task running reward normalizer
# ─────────────────────────────────────────────────────────────────────────────

class _PerTaskRewardNormalizer:
    """Welford-style running std per task. Divides each env's reward by the
    sqrt of its task's discounted-return variance (matching VecNormalize's
    reward-norm formulation but with per-task statistics)."""

    def __init__(self, num_tasks: int, task_ids: np.ndarray, gamma: float = 0.99,
                 eps: float = 1e-8, clip: float = 10.0):
        self.num_tasks = num_tasks
        self.task_ids = task_ids
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.returns = np.zeros(len(task_ids), dtype=np.float64)
        self.mean = np.zeros(num_tasks, dtype=np.float64)
        self.var = np.ones(num_tasks, dtype=np.float64)
        self.count = np.full(num_tasks, eps, dtype=np.float64)

    def update_and_normalize(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        for ti in range(self.num_tasks):
            mask = self.task_ids == ti
            batch = self.returns[mask]
            batch_n = len(batch)
            if batch_n == 0:
                continue
            batch_mean = batch.mean()
            batch_var = batch.var()
            delta = batch_mean - self.mean[ti]
            tot = self.count[ti] + batch_n
            new_mean = self.mean[ti] + delta * batch_n / tot
            m_a = self.var[ti] * self.count[ti]
            m_b = batch_var * batch_n
            new_var = (m_a + m_b + delta**2 * self.count[ti] * batch_n / tot) / tot
            self.mean[ti] = new_mean
            self.var[ti] = new_var
            self.count[ti] = tot
        scale = np.sqrt(self.var[self.task_ids] + self.eps)
        out = np.clip(rewards / scale, -self.clip, self.clip)
        self.returns[dones] = 0.0
        return out.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-task VecEnv
# ─────────────────────────────────────────────────────────────────────────────

class MultiTaskHexVecEnv(VecEnv):
    def __init__(
        self,
        propellers,
        num_envs: int,
        device: str = "cpu",
        dt: float = 0.01,
        seed: int = 0,
        gate_density: int = 3,
    ) -> None:
        if num_envs % NUM_TASKS != 0:
            raise ValueError(f"num_envs ({num_envs}) must be divisible by {NUM_TASKS}")
        per_task = num_envs // NUM_TASKS

        self.task_names = list(TASK_NAMES)
        self.num_tasks = NUM_TASKS
        self.per_task = per_task
        # Loose training bounds (per-task GATE_CONFIG bounds are too tight).
        TRAIN_X = (-20.0, 20.0)
        TRAIN_Y = (-20.0, 20.0)
        TRAIN_Z = (-5.0, 0.5)   # NED: floor at +0.5, ceiling at -5

        self.envs = []
        self.task_gate_pos = []
        self.task_gate_yaw = []
        self.task_start_pos = []
        for ti, name in enumerate(self.task_names):
            gpos, gyaw, spos = _task_config(name, density=gate_density)
            self.task_gate_pos.append(gpos)
            self.task_gate_yaw.append(gyaw)
            self.task_start_pos.append(spos)
            # Slalom's x extent is ~200m; widen its x bound only.
            xb = TRAIN_X
            if name == "slalom":
                xb = (-5.0, float(gpos[:, 0].max()) + 5.0)
            is_hover = (name == "hover")
            # v4: figure8 gets a canonical start (its tight closed loop
            # benefits from a committed initial direction); slalom and
            # shuttle-run keep random-gate init for robustness.
            random_init = (not is_hover) and (name != "figure8")
            env = TorchDroneGateEnv(
                num_envs=per_task,
                propellers=propellers,
                gates_pos=gpos,
                gate_yaw=gyaw,
                start_pos=spos,
                x_bounds=xb,
                y_bounds=TRAIN_Y,
                z_bounds=TRAIN_Z,
                gates_ahead=GATES_AHEAD,
                device=device,
                dt=dt,
                seed=seed + ti,
                initialize_at_random_gates=random_init,
                upright_bonus=0.0 if is_hover else UPRIGHT_BONUS,
                tilt_terminate_cos=TILT_TERMINATE_COS,
                extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
                # v4 additions (skip for hover — wrapper overrides its reward):
                velocity_reward_coef=0.0 if is_hover else VELOCITY_REWARD_COEF,
                altitude_floor_z=ALTITUDE_FLOOR_Z,
                altitude_floor_coef=0.0 if is_hover else ALTITUDE_FLOOR_COEF,
            )
            self.envs.append(env)

        self._hover_env = self.envs[HOVER_ID]
        self._hover_target = HOVER_TARGET_NED.to(self._hover_env.dev)

        base_dim = self.envs[0].state_len
        if base_dim != BASE_OBS_DIM:
            raise RuntimeError(
                f"env state_len={base_dim} ≠ expected {BASE_OBS_DIM}"
            )
        observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FULL_OBS_DIM,), dtype=np.float32
        )
        action_space = self.envs[0].action_space
        VecEnv.__init__(self, num_envs, observation_space, action_space)

        ids = np.repeat(np.arange(self.num_tasks), per_task)
        self.task_ids = ids.astype(np.int64)
        self.task_one_hot = np.eye(self.num_tasks, dtype=np.float32)[ids]
        self.num_motors = self.envs[0].num_motors

        # Per-task reward normalizer
        self.reward_normalizer = _PerTaskRewardNormalizer(
            num_tasks=self.num_tasks, task_ids=self.task_ids,
        )

    def _stack_obs(self, sub_obs_list):
        base = np.concatenate(sub_obs_list, axis=0).astype(np.float32)
        return np.concatenate([base, self.task_one_hot], axis=1)

    def reset(self):
        return self._stack_obs([e.reset() for e in self.envs])

    def step_async(self, actions):
        per = self.per_task
        for i, e in enumerate(self.envs):
            e.step_async(actions[i * per:(i + 1) * per])

    def step_wait(self):
        sub_o, sub_r, sub_d, infos = [], [], [], []
        for ti, e in enumerate(self.envs):
            o, r, d, info = e.step_wait()

            if ti == HOVER_ID:
                diverged = ~torch.isfinite(e.world_states).all(dim=1)
                r_hover = _hover_reward(e.world_states, self._hover_target, diverged)
                r = r_hover.cpu().numpy().astype(np.float32)

            sub_o.append(o); sub_r.append(r); sub_d.append(d)
            for item in info:
                item["task"] = self.task_names[ti]
                item["task_id"] = ti
                if "terminal_observation" in item:
                    pad = np.zeros(self.num_tasks, dtype=np.float32)
                    pad[ti] = 1.0
                    item["terminal_observation"] = np.concatenate(
                        [item["terminal_observation"].astype(np.float32), pad]
                    )
                infos.append(item)

        raw_rewards = np.concatenate(sub_r).astype(np.float32)
        dones_np = np.concatenate(sub_d)
        norm_rewards = self.reward_normalizer.update_and_normalize(raw_rewards, dones_np)

        return (
            self._stack_obs(sub_o),
            norm_rewards,
            dones_np,
            infos,
        )

    def close(self):
        for e in self.envs:
            e.close()

    def seed(self, seed=None):
        return []

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode="human"):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# MTRL policy  (unchanged from v2)
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2,
         act=nn.Tanh) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class MTRLActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_tasks: int = NUM_TASKS, **kwargs):
        self.num_tasks = num_tasks
        kwargs.pop("net_arch", None)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.shared_encoder = _mlp(SHARED_OBS_DIM, ENCODER_HIDDEN, ENCODER_LATENT, n_hidden=2)
        self.task_encoders = nn.ModuleList([
            _mlp(TASK_OBS_DIM, ENCODER_HIDDEN, ENCODER_LATENT, n_hidden=2)
            for _ in range(self.num_tasks)
        ])
        self.actor_trunk = _mlp(2 * ENCODER_LATENT, ACTOR_HIDDEN, ACTOR_HIDDEN, n_hidden=1)
        self.action_mean = nn.Linear(ACTOR_HIDDEN, int(np.prod(self.action_space.shape)))
        self.critics = nn.ModuleList([
            _mlp(BASE_OBS_DIM, ACTOR_HIDDEN, 1, n_hidden=2)
            for _ in range(self.num_tasks)
        ])
        self.log_std = nn.Parameter(
            torch.full((int(np.prod(self.action_space.shape)),), float(self.log_std_init))
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _split(self, obs):
        drone = obs[:, :SHARED_OBS_DIM]
        task_obs = obs[:, SHARED_OBS_DIM:BASE_OBS_DIM]
        one_hot = obs[:, BASE_OBS_DIM:BASE_OBS_DIM + self.num_tasks]
        full_base = obs[:, :BASE_OBS_DIM]
        return drone, task_obs, one_hot, full_base

    def _actor_latent(self, drone, task_obs, one_hot):
        shared_lat = self.shared_encoder(drone)
        all_task_lats = torch.stack(
            [enc(task_obs) for enc in self.task_encoders], dim=1
        )
        task_lat = (all_task_lats * one_hot.unsqueeze(-1)).sum(dim=1)
        return self.actor_trunk(torch.cat([shared_lat, task_lat], dim=1))

    def _gated_value(self, full_base, one_hot):
        vs = torch.cat([c(full_base) for c in self.critics], dim=1)
        return (vs * one_hot).sum(dim=1)

    def _distribution(self, latent_pi):
        mean = self.action_mean(latent_pi)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def forward(self, obs, deterministic: bool = False):
        drone, task_obs, one_hot, full_base = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot)
        distribution = self._distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._gated_value(full_base, one_hot)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def predict_values(self, obs):
        _, _, one_hot, full_base = self._split(obs)
        return self._gated_value(full_base, one_hot)

    def evaluate_actions(self, obs, actions):
        drone, task_obs, one_hot, full_base = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot)
        distribution = self._distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._gated_value(full_base, one_hot)
        return values, log_prob, entropy

    def get_distribution(self, obs):
        drone, task_obs, one_hot, _ = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot)
        return self._distribution(latent_pi)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy annealing callback (restored in v3)
# ─────────────────────────────────────────────────────────────────────────────

class EntCoefAnneal(BaseCallback):
    def __init__(self, start: float, end: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-task evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _eval_per_task(env: MultiTaskHexVecEnv, model, n_steps: int):
    """Roll out for `n_steps` and collect both completed-episode stats and
    a live snapshot of every env's current state. The live snapshot avoids
    the bias where stable policies (no early termination) produce n=0
    completed episodes inside short rollouts.

    Returns (ep_r, ep_g, live_g, live_steps) where:
        ep_r[t], ep_g[t]   — completed episodes' total reward / final gates
        live_g[t]          — gates-passed counts visible *during* the rollout
                             (per env, summed: episodic resets + current)
        live_steps[t]      — total env-steps elapsed for task t
    """
    obs = env.reset()
    per = env.per_task
    cur_r = np.zeros(env.num_envs, dtype=np.float64)
    cur_g = np.zeros(env.num_envs, dtype=np.int64)
    ep_r = [[] for _ in range(env.num_tasks)]
    ep_g = [[] for _ in range(env.num_tasks)]
    # Total gate-passes observed across the rollout (across resets, all envs).
    total_g = np.zeros(env.num_tasks, dtype=np.int64)
    prev_g = np.zeros(env.num_envs, dtype=np.int64)
    for _ in range(n_steps):
        if model is None:
            action = np.stack(
                [env.action_space.sample() for _ in range(env.num_envs)], axis=0
            )
        else:
            action, _ = model.predict(obs, deterministic=False)
        obs, rews, dones, infos = env.step(action)
        cur_r += rews
        for i, info in enumerate(infos):
            gp = info.get("num_gates_passed", None)
            if gp is not None:
                cur_g[i] = int(np.asarray(gp)[i % per])
            # Count gate-passes between snapshots (handles within-episode
            # increments AND episodic resets to 0).
            delta = cur_g[i] - prev_g[i]
            if delta > 0:
                total_g[i // per] += delta
            prev_g[i] = cur_g[i]
        for i, done in enumerate(dones):
            if done:
                ti = i // per
                ep_r[ti].append(float(cur_r[i]))
                ep_g[ti].append(int(cur_g[i]))
                cur_r[i] = 0.0
                cur_g[i] = 0
                prev_g[i] = 0
    live_steps = np.full(env.num_tasks, n_steps * per, dtype=np.int64)
    return ep_r, ep_g, total_g, live_steps


def _format_eval(ep_r, ep_g, total_g, live_steps, dt: float = 0.01):
    lines = []
    for ti, name in enumerate(TASK_NAMES):
        r = float(np.mean(ep_r[ti])) if ep_r[ti] else float("nan")
        g = float(np.mean(ep_g[ti])) if ep_g[ti] else float("nan")
        n = len(ep_r[ti])
        # Gates per second across the entire eval window (robust to n=0).
        gps = total_g[ti] / max(live_steps[ti] * dt, 1e-9)
        lines.append(
            f"  {name:>12}: reward/ep={r:+8.3f}  gates/ep={g:6.2f}  (n={n})  "
            f"gates/sec={gps:6.3f}  (live total={int(total_g[ti])})"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80_000_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--ent-start", type=float, default=0.01)
    parser.add_argument("--ent-end", type=float, default=1e-4)
    parser.add_argument("--gate-density", type=int, default=3,
                        help="Subdivide each gate segment by this factor (1 = original).")
    parser.add_argument("--warm-start-dir", default=None,
                        help="Path to a directory containing policy.zip + vecnormalize.pkl "
                             "from a previous run. Initializes PPO weights + obs-norm stats "
                             "from those files instead of training from scratch. Use for "
                             "per-morphology fine-tuning in the EA inner loop.")
    parser.add_argument("--genome-npy", default=None,
                        help="Path to a .npy file with a (6, 6) hex genome array. Defaults "
                             "to the standard symmetric hexacopter.")
    parser.add_argument(
        "--out-dir",
        default=f"__data__/spear_rl_hex_mtrl_v4/{time.strftime('%Y%m%d_%H%M%S')}",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.genome_npy is not None:
        genome = np.load(args.genome_npy)
        if genome.shape != (6, 6):
            raise ValueError(
                f"--genome-npy must be a (6, 6) array; got shape {genome.shape}. "
                f"Variable-motor morphologies are not supported by this script."
            )
        print(f"morphology: loaded genome from {args.genome_npy}")
    else:
        genome = build_hex_genome()
    bp = spherical_angular_to_blueprint(genome)
    propellers = blueprint_to_propellers(bp, convention="ned")
    print(f"morphology: {len(propellers)} propellers")

    _raw_env = MultiTaskHexVecEnv(
        propellers=propellers,
        num_envs=args.num_envs,
        device=args.device,
        dt=0.01,
        seed=args.seed,
        gate_density=args.gate_density,
    )
    for ti, name in enumerate(TASK_NAMES):
        print(f"  task {name:>12}: {len(_raw_env.task_gate_pos[ti])} gates")
    print(
        f"num_envs={_raw_env.num_envs}  per_task={_raw_env.per_task}  "
        f"obs_dim={_raw_env.observation_space.shape[0]}  "
        f"(shared={SHARED_OBS_DIM}, task={TASK_OBS_DIM}, one_hot={NUM_TASKS})"
    )
    # Obs-only VecNormalize — reward normalization is done per-task inside the wrapper.
    if args.warm_start_dir is not None:
        ws_dir = Path(args.warm_start_dir)
        ws_vn = ws_dir / "vecnormalize.pkl"
        ws_policy = ws_dir / "policy.zip"
        if not ws_vn.exists() or not ws_policy.exists():
            raise FileNotFoundError(
                f"--warm-start-dir {ws_dir} must contain policy.zip + vecnormalize.pkl"
            )
        env = VecNormalize.load(str(ws_vn), _raw_env)
        env.training = True
        env.norm_reward = False
        print(f"warm-started obs-norm stats from {ws_vn}")
    else:
        env = VecNormalize(_raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    batch_size = (args.n_steps * args.num_envs) // 8
    if args.warm_start_dir is not None:
        # Reload PPO with the new env. SB3 requires matching obs/action shapes;
        # since we restrict --genome-npy to 6 motors this is guaranteed.
        model = PPO.load(
            str(ws_policy), env=env, device=args.device,
            custom_objects={
                "policy_class": MTRLActorCriticPolicy,
                # Override the base run's hyperparams so this fine-tune uses
                # the args supplied on this command line (n_steps, batch, etc).
                "n_steps": args.n_steps,
                "batch_size": batch_size,
                "n_epochs": 10,
                "learning_rate": 3e-4,
                "clip_range": 0.2,
                "ent_coef": args.ent_start,
            },
        )
        # The annealer drives ent_coef linearly from this start value, so make
        # sure we begin at the requested ent_start regardless of what the base
        # checkpoint ended at.
        model.ent_coef = args.ent_start
        print(f"warm-started policy weights from {ws_policy}")
    else:
        model = PPO(
            MTRLActorCriticPolicy,
            env,
            policy_kwargs=dict(log_std_init=-0.5),
            n_steps=args.n_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            learning_rate=3e-4,
            ent_coef=args.ent_start,
            clip_range=0.2,
            max_grad_norm=0.5,
            device=args.device,
            verbose=1,
        )

    # Eval long enough that each per-task env can hit its 1200-step timeout
    # at least once — otherwise the only completed episodes are early
    # crashes, biasing the printed numbers downward.
    eval_steps = 1500
    print(f"\n[before training] random-policy rollout ({eval_steps} steps):")
    rand_r, rand_g, rand_total, rand_live = _eval_per_task(env, model=None, n_steps=eval_steps)
    print(_format_eval(rand_r, rand_g, rand_total, rand_live))

    callbacks = [EntCoefAnneal(args.ent_start, args.ent_end, args.steps)]
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)
    elapsed = time.time() - t0
    print(f"\ntrained {args.steps:,} steps in {elapsed:.1f}s ({args.steps / elapsed:.0f} sps)")

    print(f"\n[after training] trained-policy rollout ({eval_steps} steps):")
    trained_r, trained_g, trained_total, trained_live = _eval_per_task(
        env, model=model, n_steps=eval_steps,
    )
    print(_format_eval(trained_r, trained_g, trained_total, trained_live))

    save_path = out_dir / "policy.zip"
    model.save(str(save_path))
    vecnorm_path = out_dir / "vecnormalize.pkl"
    env.save(str(vecnorm_path))
    print(f"\nsaved policy → {save_path}")
    print(f"saved vecnormalize → {vecnorm_path}")
    print(f"\nVisualize with:\n"
          f"  uv run examples/spear/27_eval_rl_hex_mtrl_v4.py --policy {save_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
