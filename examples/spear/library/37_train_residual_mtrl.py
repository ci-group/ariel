"""Stage 3 — multi-task PPO residual on top of the analytical hover prior,
with per-worker (morph, task) rotation.

Each VecEnv worker holds one (morph, task) pair for the whole run: morphs
come from `__data__/hex_library/v1` (minus a stratified held-out split),
tasks are assigned round-robin across all 5 (hover, figure8, slalom,
shuttle-run, circle). Observations are
``(gate_obs[26], task_one_hot[5], morph_features[22],
   cmaes_params[11], median_score[1]) = 65d``
where the trailing 12d prior descriptor is **critic-only** (the actor
never sees it — see `_split`). Actions are residual in ``[-1, 1]^N``; the
env adds ``α_task · residual`` to the analytical prior internally.

One remaining limit vs the full Stage 3 design (intentionally deferred):
each worker holds ONE morph for the entire run. True per-episode morph
rotation needs a refactor of `ResidualDroneEnv` to swap `cmaes_params` +
`morph_features` on `reset`; for now diversity comes from N workers.

Gate (IMPLEMENTATION_PLAN.md step 7):
  * Step-0 mean reward should be high — the prior alone hovers.
  * First 1M steps don't crash or diverge.

Run:
    uv run examples/spear/library/37_train_residual_mtrl.py \\
        --steps 20_000_000 --num-envs 20

Resume from checkpoint:
    uv run examples/spear/library/37_train_residual_mtrl.py \\
        --resume __data__/library_residual_mtrl/<timestamp> --out-dir <new_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: F401
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).parent))
from envs.residual_drone_env import (  # noqa: E402
    MORPH_FEAT_DIM, NUM_TASKS, PRIOR_TAIL_DIM, ResidualDroneEnv, TASK_NAMES,
)
from hex_sampler import _stratify_key, sample_feasible  # noqa: E402

# Obs layout (must match ResidualDroneEnv expansion):
#   base = TorchDroneGateEnv.state_len   (gate_obs + drone + motor_w)
#   tail = task_oh (5) + morph_features (22)
#          + cmaes_params (11) + median_score (1)   ← critic-only
# state_len for hex (6 motors, 1 hover gate, 2 gates_ahead) = 12+6+4*2 = 26
DRONE_OBS_DIM = 18                                # state + motor_w (shared)
TASK_OBS_DIM = 8                                  # 2 gates × (xyz + yaw)
BASE_OBS_DIM = DRONE_OBS_DIM + TASK_OBS_DIM       # 26 — matches parent
TAIL_DIM = NUM_TASKS + MORPH_FEAT_DIM + PRIOR_TAIL_DIM  # 39
FULL_OBS_DIM = BASE_OBS_DIM + TAIL_DIM            # 65
SHARED_IN_DIM = DRONE_OBS_DIM + MORPH_FEAT_DIM    # 40 — drone+morph to shared
# Per-task critic additionally sees the 12d prior descriptor
# (cmaes_params + median_score) — "how trustworthy is my prior here."
# The actor path never touches it, so it cannot learn to undo the prior.
CRITIC_IN_DIM = BASE_OBS_DIM + MORPH_FEAT_DIM + PRIOR_TAIL_DIM  # 60

ENCODER_LATENT = 32
ENCODER_HIDDEN = 128
ACTOR_HIDDEN = 256


# ─────────────────────────────────────────────────────────────────────────────
# Morph rotation: load library + pair with sampler to get propellers/inertia
# ─────────────────────────────────────────────────────────────────────────────

def _load_morph_library(path: Path, n_morphs: int) -> list[dict]:
    """Return list of morph dicts ready for ResidualDroneEnv."""
    d = np.load(path)
    seeds_in_lib = set(int(s) for s in d["morph_seed"])
    # Re-sample to recover propeller + inertia objects (genome alone is not
    # quite enough to round-trip cleanly through derive_reference_params).
    sampled = sample_feasible(200, seed=42, stratify=True)
    by_seed = {m.seed: m for m in sampled}

    morphs = []
    for i in range(min(n_morphs, len(d["morph_seed"]))):
        seed = int(d["morph_seed"][i])
        if seed not in by_seed:
            continue
        m = by_seed[seed]
        morphs.append({
            "morph_seed":     seed,
            "propellers":     m.propellers,
            "mass":           float(m.mass),
            "inertia":        m.inertia,
            "prop_size":      int(m.prop_size),
            "twr":            float(m.twr),
            "cmaes_params":   d["cmaes_params"][i].astype(np.float32),
            "morph_features": d["morph_features"][i].astype(np.float32),
            "median_score":   float(d["median_score"][i]),
            "cell":           _stratify_key(m),
        })
    if not morphs:
        raise RuntimeError(
            f"could not pair any library morph_seeds with the sampler "
            f"({len(seeds_in_lib)} in library)"
        )
    return morphs


def _select_held_out(morphs: list[dict], n_held_out: int, seed: int) -> list[int]:
    """Pick `n_held_out` morph indices stratified across the sampler's
    27 (arm, prop, asym) cells: shuffle the occupied cells, then take one
    morph per cell round-robin so the held-out set spans the coverage
    grid rather than clustering. Deterministic in `seed`."""
    if n_held_out <= 0:
        return []
    if n_held_out >= len(morphs):
        raise ValueError(
            f"held-out size {n_held_out} >= library size {len(morphs)}"
        )
    rng = np.random.RandomState(seed)
    by_cell: dict[tuple, list[int]] = {}
    for i, m in enumerate(morphs):
        by_cell.setdefault(m["cell"], []).append(i)
    cells = sorted(by_cell)
    rng.shuffle(cells)
    for idxs in by_cell.values():
        rng.shuffle(idxs)
    held: list[int] = []
    while len(held) < n_held_out:
        progressed = False
        for c in cells:
            if by_cell[c] and len(held) < n_held_out:
                held.append(by_cell[c].pop())
                progressed = True
        if not progressed:
            break
    return sorted(held)


# ─────────────────────────────────────────────────────────────────────────────
# Morph-rotating VecEnv: N workers, each a ResidualDroneEnv with one morph
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Per-task reward normalizer — Welford-style running discounted-return std,
# one stream per task. Mirrors 27_v4. Why this matters: hover's reward is
# ~+0.0125/step (continuous shaping) while figure8 has +1 spikes at gate
# passes — sharing one VecNormalize scale washes hover out and tanks PPO's
# value function on the smoothed task. Per-task scaling normalises each
# task to ~unit variance independently.
# ─────────────────────────────────────────────────────────────────────────────

class _PerTaskRewardNormalizer:
    def __init__(self, num_tasks: int, task_ids: np.ndarray,
                 gamma: float = 0.99, eps: float = 1e-8, clip: float = 10.0):
        self.num_tasks = num_tasks
        self.task_ids = task_ids
        self.gamma = gamma
        self.eps = eps
        self.clip = clip
        self.returns = np.zeros(len(task_ids), dtype=np.float64)
        self.mean = np.zeros(num_tasks, dtype=np.float64)
        self.var = np.ones(num_tasks, dtype=np.float64)
        self.count = np.full(num_tasks, eps, dtype=np.float64)

    def update_and_normalize(self, rewards: np.ndarray,
                             dones: np.ndarray) -> np.ndarray:
        self.returns = self.returns * self.gamma + rewards
        for ti in range(self.num_tasks):
            mask = self.task_ids == ti
            batch = self.returns[mask]
            n = len(batch)
            if n == 0:
                continue
            bm, bv = batch.mean(), batch.var()
            delta = bm - self.mean[ti]
            tot = self.count[ti] + n
            new_mean = self.mean[ti] + delta * n / tot
            m_a = self.var[ti] * self.count[ti]
            m_b = bv * n
            new_var = (m_a + m_b + delta**2 * self.count[ti] * n / tot) / tot
            self.mean[ti] = new_mean
            self.var[ti] = new_var
            self.count[ti] = tot
        scale = np.sqrt(self.var[self.task_ids] + self.eps)
        out = np.clip(rewards / scale, -self.clip, self.clip)
        self.returns[dones] = 0.0
        return out.astype(np.float32)


class MorphRotatingVecEnv(VecEnv):
    """One `ResidualDroneEnv` per slot. Slots are assigned a (morph, task)
    pair so every task gets at least one worker and morphs rotate across
    tasks. Observations are the expanded 65d vector (incl. the critic-only
    prior descriptor); the residual action space matches the per-env motor
    count (hex = 6).
    """

    def __init__(self, morphs: list[dict], tasks: list[str] | str = "hover",
                 alpha: float | None = None, device: str = "cpu",
                 max_steps: int = 1200, seed: int = 0):
        if isinstance(tasks, str):
            tasks = [tasks] * len(morphs)
        if len(tasks) != len(morphs):
            raise ValueError(
                f"tasks ({len(tasks)}) must match morphs ({len(morphs)})"
            )
        self.num_slots = len(morphs)
        self.slot_tasks = list(tasks)
        self.envs: list[ResidualDroneEnv] = []
        for i, (m, t) in enumerate(zip(morphs, tasks)):
            if len(m["propellers"]) != 6:
                raise ValueError(
                    "MVP only supports hex morphs (6 motors); got "
                    f"{len(m['propellers'])} on morph index {i}"
                )
            self.envs.append(ResidualDroneEnv(
                m, task=t, alpha=alpha, num_envs=1, max_steps=max_steps,
                device=device, seed=seed + i,
            ))
        obs_space = self.envs[0].observation_space
        act_space = self.envs[0].action_space
        VecEnv.__init__(self, self.num_slots, obs_space, act_space)
        self._actions_buffer = None

        # Per-task reward normaliser (one running std per task).
        self.task_ids = np.array(
            [TASK_NAMES.index(t) for t in tasks], dtype=np.int64,
        )
        self.reward_normalizer = _PerTaskRewardNormalizer(
            num_tasks=NUM_TASKS, task_ids=self.task_ids,
        )

    def reset(self):
        obs_list = [e.reset()[0] for e in self.envs]
        return np.stack(obs_list, axis=0).astype(np.float32)

    def step_async(self, actions):
        self._actions_buffer = actions

    def step_wait(self):
        obs_list, r_list, d_list, infos = [], [], [], []
        for i, e in enumerate(self.envs):
            e.step_async(self._actions_buffer[i:i+1])
            o, r, d, info = e.step_wait()
            obs_list.append(o[0]); r_list.append(float(r[0])); d_list.append(bool(d[0]))
            info[0]["task"] = self.slot_tasks[i]
            info[0]["task_id"] = int(self.task_ids[i])
            info[0]["mean_abs_residual"] = float(np.abs(self._actions_buffer[i]).mean())
            infos.append(info[0])
        raw_rewards = np.asarray(r_list, dtype=np.float32)
        dones_np = np.asarray(d_list, dtype=bool)
        norm_rewards = self.reward_normalizer.update_and_normalize(
            raw_rewards, dones_np,
        )
        return (
            np.stack(obs_list, axis=0).astype(np.float32),
            norm_rewards,
            dones_np,
            infos,
        )

    def close(self):
        pass

    def seed(self, seed=None):
        return []

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return []

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def render(self, mode="human"):
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Simple residual policy (MLP). Stage 3 full version uses the v4 MTRL
# multi-encoder architecture; for the MVP a single MLP is sufficient
# because there is only one task active.
# ─────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: int, out_dim: int, n_hidden: int = 2,
         act=nn.Tanh) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden), act()]
    for _ in range(n_hidden - 1):
        layers += [nn.Linear(hidden, hidden), act()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


# ─────────────────────────────────────────────────────────────────────────────
# MTRL actor-critic — direct port of 27_v4 with morph_features added to the
# shared encoder + each per-task critic. Architecture:
#   shared_enc(drone + morph)        → latent_s
#   task_enc_t(gates_ahead)          → latent_t, one per task
#   actor_trunk([latent_s, gated_lt]) → mean residual
#   critic_t(base_obs + morph)        → per-task value, gated by one_hot
# Action is residual (the env adds the analytical prior).
# ─────────────────────────────────────────────────────────────────────────────

class MTRLActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, num_tasks: int = NUM_TASKS, **kwargs):
        self.num_tasks = num_tasks
        kwargs.pop("net_arch", None)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.shared_encoder = _mlp(
            SHARED_IN_DIM, ENCODER_HIDDEN, ENCODER_LATENT, n_hidden=2,
        )
        self.task_encoders = nn.ModuleList([
            _mlp(TASK_OBS_DIM, ENCODER_HIDDEN, ENCODER_LATENT, n_hidden=2)
            for _ in range(self.num_tasks)
        ])
        self.actor_trunk = _mlp(
            2 * ENCODER_LATENT, ACTOR_HIDDEN, ACTOR_HIDDEN, n_hidden=1,
        )
        self.action_mean = nn.Linear(
            ACTOR_HIDDEN, int(np.prod(self.action_space.shape)),
        )
        self.critics = nn.ModuleList([
            _mlp(CRITIC_IN_DIM, ACTOR_HIDDEN, 1, n_hidden=2)
            for _ in range(self.num_tasks)
        ])
        self.log_std = nn.Parameter(
            torch.full(
                (int(np.prod(self.action_space.shape)),),
                float(self.log_std_init),
            )
        )
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _split(self, obs):
        """Split 65d obs → (drone, task_obs, one_hot, morph, critic_in).

        The trailing 12d prior descriptor (cmaes_params + median_score) is
        routed to `critic_in` ONLY — the actor paths (drone, task_obs,
        morph) must never include it."""
        morph_end = BASE_OBS_DIM + self.num_tasks + MORPH_FEAT_DIM    # 53
        drone = obs[:, :DRONE_OBS_DIM]                                # 18
        task_obs = obs[:, DRONE_OBS_DIM:BASE_OBS_DIM]                 # 8
        one_hot = obs[:, BASE_OBS_DIM:BASE_OBS_DIM + self.num_tasks]  # 5
        morph = obs[:, BASE_OBS_DIM + self.num_tasks:morph_end]       # 22
        prior_desc = obs[:, morph_end:]                               # 12
        critic_in = torch.cat(
            [obs[:, :BASE_OBS_DIM], morph, prior_desc], dim=1,
        )                                                             # 60
        return drone, task_obs, one_hot, morph, critic_in

    def _actor_latent(self, drone, task_obs, one_hot, morph):
        shared_lat = self.shared_encoder(torch.cat([drone, morph], dim=1))
        all_task_lats = torch.stack(
            [enc(task_obs) for enc in self.task_encoders], dim=1
        )
        task_lat = (all_task_lats * one_hot.unsqueeze(-1)).sum(dim=1)
        return self.actor_trunk(torch.cat([shared_lat, task_lat], dim=1))

    def _gated_value(self, critic_in, one_hot):
        vs = torch.cat([c(critic_in) for c in self.critics], dim=1)
        return (vs * one_hot).sum(dim=1)

    def _distribution(self, latent_pi):
        mean = self.action_mean(latent_pi)
        return self.action_dist.proba_distribution(mean, self.log_std)

    def forward(self, obs, deterministic: bool = False):
        drone, task_obs, one_hot, morph, critic_in = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot, morph)
        distribution = self._distribution(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._gated_value(critic_in, one_hot)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def predict_values(self, obs):
        _, _, one_hot, _, critic_in = self._split(obs)
        return self._gated_value(critic_in, one_hot)

    def evaluate_actions(self, obs, actions):
        drone, task_obs, one_hot, morph, critic_in = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot, morph)
        distribution = self._distribution(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._gated_value(critic_in, one_hot)
        return values, log_prob, entropy

    def get_distribution(self, obs):
        drone, task_obs, one_hot, morph, _ = self._split(obs)
        latent_pi = self._actor_latent(drone, task_obs, one_hot, morph)
        return self._distribution(latent_pi)


# ─────────────────────────────────────────────────────────────────────────────
# Entropy annealing — mirrors 27_v4. Without this `std` drifts UP during
# training (saw 0.50 → 0.825 over 5M steps); the policy becomes more
# exploratory rather than converging, which kills MTRL performance.
# ─────────────────────────────────────────────────────────────────────────────

class EntCoefAnneal(BaseCallback):
    def __init__(self, start: float, end: float, total_timesteps: int):
        super().__init__(verbose=0)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self.start + (self.end - self.start) * frac
        return True


class GradientCosineCallback(BaseCallback):
    """Log pairwise per-task actor gradient cosine similarities.

    Fires at `_on_rollout_end` (after rollout collection, before PPO's train())
    when the rollout buffer is still shaped (buffer_size, n_envs, ...).

    PCGrad escalation trigger (not enacted here, logging only):
      cosine < 0 in >50% of logged updates AFTER 20M steps.
    """

    def __init__(self, raw_env: "MorphRotatingVecEnv", log_every: int = 250_000):
        super().__init__(verbose=0)
        self.raw_env = raw_env
        self.log_every = log_every
        self._next_log = log_every

    def _on_rollout_end(self) -> bool:
        if self.num_timesteps < self._next_log:
            return True
        self._next_log += self.log_every

        rb = self.model.rollout_buffer
        if not rb.full:
            return True

        policy = self.model.policy
        clip_range = float(self.model.clip_range(self.model._current_progress_remaining))
        actor_params = (
            list(policy.shared_encoder.parameters())
            + list(policy.task_encoders.parameters())
            + list(policy.actor_trunk.parameters())
            + list(policy.action_mean.parameters())
        )

        prev_training = policy.training
        policy.set_training_mode(True)
        policy.optimizer.zero_grad()

        task_grads: dict[str, torch.Tensor] = {}
        for task in TASK_NAMES:
            env_indices = [
                i for i, t in enumerate(self.raw_env.slot_tasks) if t == task
            ]
            if not env_indices:
                continue

            # rb arrays are (buffer_size, n_envs, ...) before get() flattens them
            obs_t = torch.as_tensor(
                rb.observations[:, env_indices].reshape(-1, rb.observations.shape[-1]),
                device=self.model.device, dtype=torch.float32,
            )
            acts_t = torch.as_tensor(
                rb.actions[:, env_indices].reshape(-1, rb.actions.shape[-1]),
                device=self.model.device, dtype=torch.float32,
            )
            logp_old = torch.as_tensor(
                rb.log_probs[:, env_indices].reshape(-1),
                device=self.model.device, dtype=torch.float32,
            )
            adv = torch.as_tensor(
                rb.advantages[:, env_indices].reshape(-1),
                device=self.model.device, dtype=torch.float32,
            )
            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

            policy.optimizer.zero_grad()
            _values, log_prob, _entropy = policy.evaluate_actions(obs_t, acts_t)
            ratio = torch.exp(log_prob - logp_old)
            loss = -torch.min(
                ratio * adv_norm,
                ratio.clamp(1.0 - clip_range, 1.0 + clip_range) * adv_norm,
            ).mean()
            loss.backward()

            grads = [p.grad.flatten() for p in actor_params if p.grad is not None]
            if grads:
                task_grads[task] = torch.cat(grads).detach().clone()

        policy.optimizer.zero_grad()
        policy.set_training_mode(prev_training)

        if len(task_grads) < 2:
            return True

        tasks_with_grads = list(task_grads.keys())
        lines = [f"\n[grad_cosine @ {self.num_timesteps:,} steps]"]
        for i, t1 in enumerate(tasks_with_grads):
            for j, t2 in enumerate(tasks_with_grads):
                if j <= i:
                    continue
                g1, g2 = task_grads[t1], task_grads[t2]
                cos = torch.nn.functional.cosine_similarity(
                    g1.unsqueeze(0), g2.unsqueeze(0)
                ).item()
                lines.append(f"  {t1:>12} × {t2:<12}: {cos:+.3f}")
        print("\n".join(lines), flush=True)
        return True

    def _on_step(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-task eval — completed-episode stats + live gate counts per task
# ─────────────────────────────────────────────────────────────────────────────

def _eval_per_task(env, raw_env: "MorphRotatingVecEnv", model, n_steps: int):
    """Roll out for `n_steps`; group rewards/gates/residual norms by `slot_tasks`."""
    obs = env.reset()
    cur_r = np.zeros(env.num_envs, dtype=np.float64)
    cur_g = np.zeros(env.num_envs, dtype=np.int64)
    prev_g = np.zeros(env.num_envs, dtype=np.int64)
    ep_r = {t: [] for t in TASK_NAMES}
    ep_g = {t: [] for t in TASK_NAMES}
    total_g = {t: 0 for t in TASK_NAMES}
    n_steps_per_task = {t: 0 for t in TASK_NAMES}
    abs_res_sum = {t: 0.0 for t in TASK_NAMES}
    abs_res_cnt = {t: 0 for t in TASK_NAMES}

    slot_tasks = raw_env.slot_tasks
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
            t = slot_tasks[i]
            n_steps_per_task[t] += 1
            gp = info.get("num_gates_passed", None)
            if gp is not None:
                cur_g[i] = int(np.asarray(gp)[0]) if hasattr(gp, "__len__") else int(gp)
            delta = cur_g[i] - prev_g[i]
            if delta > 0:
                total_g[t] += delta
            prev_g[i] = cur_g[i]
            mar = info.get("mean_abs_residual", None)
            if mar is not None:
                abs_res_sum[t] += mar
                abs_res_cnt[t] += 1
        for i, done in enumerate(dones):
            if done:
                t = slot_tasks[i]
                ep_r[t].append(float(cur_r[i]))
                ep_g[t].append(int(cur_g[i]))
                cur_r[i] = 0.0
                cur_g[i] = 0
                prev_g[i] = 0
    mean_abs_res = {
        t: abs_res_sum[t] / max(abs_res_cnt[t], 1) for t in TASK_NAMES
    }
    return ep_r, ep_g, total_g, n_steps_per_task, mean_abs_res


def _format_eval(ep_r, ep_g, total_g, n_steps_per_task, mean_abs_res,
                 dt: float = 0.01):
    lines = []
    for t in TASK_NAMES:
        r = float(np.mean(ep_r[t])) if ep_r[t] else float("nan")
        g = float(np.mean(ep_g[t])) if ep_g[t] else float("nan")
        n = len(ep_r[t])
        gps = total_g[t] / max(n_steps_per_task[t] * dt, 1e-9)
        res = mean_abs_res.get(t, float("nan"))
        lines.append(
            f"  {t:>12}: reward/ep={r:+8.3f}  gates/ep={g:6.2f}  (n={n:>2})  "
            f"gates/sec={gps:6.3f}  |res|={res:.3f}  (live total={int(total_g[t])})"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", default="__data__/hex_library/v1/library.npz")
    p.add_argument("--num-envs", type=int, default=20)
    p.add_argument("--steps", type=int, default=20_000_000)
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=None,
                   help="Override per-task default residual scale. By default "
                        "hover uses 0.10 (prior dominates) and racing tasks "
                        "use 0.40. Set to a single float to override all.")
    p.add_argument("--ent-start", type=float, default=0.005)
    p.add_argument("--ent-end", type=float, default=1e-4)
    p.add_argument("--log-std-init", type=float, default=-1.0,
                   help="Initial action log_std. -1.0 (σ≈0.37) tighter than "
                        "the previous -0.5 (σ≈0.61); pairs with entropy "
                        "annealing for a converging policy.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--held-out", type=int, default=10,
                   help="Morphs reserved for eval only (stratified across "
                        "the 27 sampler cells). 0 disables the split.")
    p.add_argument("--held-out-seed", type=int, default=0,
                   help="RNG seed for the held-out selection; keep fixed "
                        "across runs so the eval set is stable.")
    p.add_argument("--out-dir",
                   default=f"__data__/library_residual_mtrl/{time.strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--resume", default=None,
                   help="Path to a previous out_dir; auto-loads the latest "
                        "checkpoint + vecnormalize.pkl and continues training. "
                        "Step count carries over (reset_num_timesteps=False).")
    p.add_argument("--morph-seed", type=int, default=None,
                   help="If set, restrict training to the single morph with "
                        "this seed. Useful for single-morph debugging runs.")
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    all_morphs = _load_morph_library(Path(args.library), n_morphs=10**9)
    print(f"loaded {len(all_morphs)} morphs from {args.library}")
    if args.morph_seed is not None:
        all_morphs = [m for m in all_morphs if m["morph_seed"] == args.morph_seed]
        if not all_morphs:
            raise SystemExit(f"morph_seed {args.morph_seed} not found in library")
        print(f"restricted to morph_seed={args.morph_seed}")

    # Held-out split: reserved morphs never appear in a training worker.
    # Persisted next to the checkpoint so Stage-4 eval uses the same set.
    held_idx = _select_held_out(all_morphs, args.held_out, args.held_out_seed)
    train_morphs = [m for i, m in enumerate(all_morphs) if i not in set(held_idx)]
    split = {
        "held_out_seed": args.held_out_seed,
        "held_out_indices": held_idx,
        "held_out_morph_seeds": [all_morphs[i]["morph_seed"] for i in held_idx],
        "train_morph_seeds": [m["morph_seed"] for m in train_morphs],
        "library": str(args.library),
    }
    (out_dir / "held_out.json").write_text(json.dumps(split, indent=2))
    print(f"held out {len(held_idx)} morphs "
          f"(seeds {split['held_out_morph_seeds']}); "
          f"training on {len(train_morphs)}")

    # One (morph, task) pair per worker: morphs cycle through the train
    # split, tasks round-robin so every task gets num_envs/5 workers.
    morphs = [train_morphs[i % len(train_morphs)] for i in range(args.num_envs)]
    tasks = [TASK_NAMES[i % NUM_TASKS] for i in range(len(morphs))]
    print("task distribution: "
          + ", ".join(f"{t}={tasks.count(t)}" for t in TASK_NAMES))

    raw_env = MorphRotatingVecEnv(
        morphs=morphs, tasks=tasks, alpha=args.alpha,
        device=args.device, seed=args.seed,
    )
    # norm_reward=False: rewards are already per-task normalised inside
    # MorphRotatingVecEnv. Obs normalisation stays global.
    env = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # ── Stage 3 gate: step-0 per-task reward shows the prior is already
    # holding (hover task strongest; trajectory tasks should pass gates).
    print(f"\n[before training] random-residual rollout (1500 steps):")
    ep_r, ep_g, total_g, nspt, mar = _eval_per_task(env, raw_env, model=None, n_steps=1500)
    print(_format_eval(ep_r, ep_g, total_g, nspt, mar))

    # ── PPO ----------------------------------------------------------------
    batch_size = (args.n_steps * args.num_envs) // 8
    model = PPO(
        MTRLActorCriticPolicy,
        env,
        policy_kwargs=dict(log_std_init=args.log_std_init),
        n_steps=args.n_steps,
        batch_size=max(batch_size, 64),
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

    # ── Resume from checkpoint --------------------------------------------
    _reset_num_timesteps = True
    if args.resume:
        resume_dir = Path(args.resume)
        ckpts = sorted((resume_dir / "checkpoints").glob("ppo_*_steps.zip"))
        if ckpts:
            latest_ckpt = ckpts[-1]
            # Load weights + optimizer state into our fresh model so the
            # architecture (policy_kwargs, n_steps, etc.) stays correct.
            saved = PPO.load(str(latest_ckpt), device=args.device)
            model.policy.load_state_dict(saved.policy.state_dict())
            model.policy.optimizer.load_state_dict(
                saved.policy.optimizer.state_dict()
            )
            model.num_timesteps = saved.num_timesteps
            del saved
            # Restore VecNormalize running stats
            vn_path = resume_dir / "vecnormalize.pkl"
            if vn_path.exists():
                saved_vn = VecNormalize.load(str(vn_path), raw_env)
                env.obs_rms = saved_vn.obs_rms
                env.ret_rms = saved_vn.ret_rms
                print(f"loaded VecNormalize stats from {vn_path.name}")
            print(f"resumed from {latest_ckpt.name} ({model.num_timesteps:,} steps)")
            _reset_num_timesteps = False
        else:
            print(f"no checkpoints in {resume_dir}/checkpoints — starting fresh")

    t0 = time.time()
    # Checkpoint every ~250k steps. Transient CUDA errors have killed runs
    # before; checkpoints make those recoverable without losing progress.
    ckpt_freq = max(args.n_steps, 250_000 // env.num_envs)
    callbacks = [
        EntCoefAnneal(args.ent_start, args.ent_end, args.steps),
        CheckpointCallback(
            save_freq=ckpt_freq, save_path=str(out_dir / "checkpoints"),
            name_prefix="ppo", save_vecnormalize=True,
        ),
        GradientCosineCallback(raw_env, log_every=250_000),
    ]
    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False,
                reset_num_timesteps=_reset_num_timesteps)
    elapsed = time.time() - t0
    print(f"\ntrained {args.steps:,} steps in {elapsed:.1f}s "
          f"({args.steps/elapsed:.0f} sps)")

    # Final eval
    print(f"\n[after training] trained-policy rollout (1500 steps):")
    ep_r, ep_g, total_g, nspt, mar = _eval_per_task(env, raw_env, model=model, n_steps=1500)
    print(_format_eval(ep_r, ep_g, total_g, nspt, mar))

    model.save(str(out_dir / "policy.zip"))
    env.save(str(out_dir / "vecnormalize.pkl"))
    print(f"\nsaved → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
