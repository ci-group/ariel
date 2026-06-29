"""Stage 2 — TorchDroneGateEnv wrapper that applies the analytical hover
prior on top of a learned residual action.

  action_total = clamp( prior_action(state; cmaes_params) + α · residual )

The PPO actor in Stage 3 only ever sees / produces ``residual``. The prior
is owned by the env, so PPO is unchanged from 27_v4 except for obs-space
dimensions:

  obs = (gate_drone_obs, task_one_hot[5], morph_features[22])

For Stage 2 we ship a single-morph variant (hover task only). Stage 3's
MultiTaskHexVecEnv aggregator will rotate morphs across workers and add
the other four tasks; the per-worker prior+residual logic lives here.

Plan gate (per IMPLEMENTATION_PLAN.md step 6):
    With α=0 and any residual, env behaves identically to the prior
    alone.  With α=0.4 and a zero residual, prior alone still hovers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone.dynamics_params import derive_reference_params
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from prior_controller import HoverPrior  # noqa: E402

TASK_NAMES = ("hover", "figure8", "slalom", "shuttle-run", "circle")
NUM_TASKS = len(TASK_NAMES)
MORPH_FEAT_DIM = 22

# Per-task env shaping, mirroring 27_v4 (so trajectory tasks see the same
# reward landscape as the non-residual baseline; only the action plumbing
# changes).
UPRIGHT_BONUS = 0.01
# Tilt-termination disabled (0.0). At the previous 0.10 (≈85°) threshold,
# random-residual diagnostics showed ~97.5% of all episode terminations
# across every task were tilt-induced — the threshold was the binding
# safety net, even on racing tasks where banking past 90° is desirable.
# NaN-divergence + altitude floor + OOB bounds remain as kill conditions.
TILT_TERMINATE_COS = 0.0
EXTRA_YAW_RATE_PEN = 0.005
VELOCITY_REWARD_COEF = 0.005
ALTITUDE_FLOOR_Z = -0.5
ALTITUDE_FLOOR_COEF = 0.5


def _task_gate_config(task: str):
    """Return (gates_pos, gate_yaw, start_pos, x_bounds, random_init).

    Hover is a single stationary gate at the hover target. The other four
    pull from `GATE_CONFIGS`. Figure8 uses fixed start; the rest use
    random-gate init (matches 27_v4)."""
    if task == "hover":
        gpos = np.array([[0.0, 0.0, -1.5]], dtype=np.float64)
        gyaw = np.array([0.0], dtype=np.float64)
        spos = np.array([0.0, 0.0, -1.5], dtype=np.float64)
        return gpos, gyaw, spos, (-3.0, 3.0), False
    key = "backandforth" if task == "shuttle-run" else task
    cfg = GATE_CONFIGS[key]
    gpos = np.asarray(cfg.gate_pos, dtype=np.float64)
    gyaw = np.asarray(cfg.gate_yaw, dtype=np.float64)
    spos = np.asarray(cfg.starting_pos, dtype=np.float64)
    if task == "slalom":
        xb = (-5.0, float(gpos[:, 0].max()) + 5.0)
    else:
        xb = (-20.0, 20.0)
    random_init = (task != "figure8")
    return gpos, gyaw, spos, xb, random_init


class ResidualDroneEnv(TorchDroneGateEnv):
    """Per-morph residual env. The actor outputs a residual in [-1, 1]^N;
    the env adds the analytical hover prior internally.

    Construction
    ------------
    morph
        Dict with keys ``propellers``, ``mass``, ``inertia``, ``prop_size``,
        ``cmaes_params`` (N+5 numpy array), ``morph_features`` (22d numpy
        array). Typically one row of ``__data__/hex_library/v1/library.npz``.
    task
        One of ``TASK_NAMES``. For Stage 2 only ``"hover"`` is implemented;
        the other tasks raise until Stage 3 adds their gate configs.
    alpha
        Residual scaling. Default 0.4 matches plan v2.
    num_envs
        Per-worker batch (this is one VecEnv; Stage 3 will run several).
    """

    HOVER_TARGET_NED = (0.0, 0.0, -1.5)

    # Per-task residual scaling. Hover gets a tiny α because the analytical
    # prior already solves it perfectly; a full-strength residual just
    # destabilises hover and drags the shared actor to a "high-entropy"
    # equilibrium that hurts both. Racing tasks keep α=0.4 so the residual
    # has room to learn banking corrections on top of the prior.
    # v4: per-task α. Hover keeps a small budget (prior is perfect).
    # Racing tasks get the full ±0.4 swing now that the prior's attitude
    # leveler no longer fights them (see TASK_PRIOR_GAIN_SCALE below).
    TASK_ALPHA = {
        "hover":       0.10,
        "figure8":     0.40,
        "slalom":      0.40,
        "shuttle-run": 0.40,
        "circle":      0.40,
    }
    # v4: per-task prior gain scaling. The hover-tuned CMA gains include
    # active attitude leveling (k_tilt) that actively *resists* banking —
    # exactly the motion racing requires. For trajectory tasks we keep
    # altitude tracking (k_alt_p, k_alt_d) and damp rates lightly, but
    # zero `k_tilt` so the residual can steer freely.
    # Indices in cmaes_params after the N trims: [k_alt_p, k_alt_d, k_tilt,
    # k_rate, k_yaw_rate].
    TASK_PRIOR_GAIN_SCALE = {
        "hover":       np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        # k_tilt_scale=0.3: keeps ~30% of attitude leveling, enough that
        # the drone doesn't tumble from random PPO exploration but a
        # deliberate bank command from the residual can still produce
        # sustained tilt. k_rate / k_yaw_rate stay at 0.5 — light damping
        # to keep rates bounded.
        "figure8":     np.array([1.0, 1.0, 0.3, 0.5, 0.5], dtype=np.float32),
        "slalom":      np.array([1.0, 1.0, 0.3, 0.5, 0.5], dtype=np.float32),
        "shuttle-run": np.array([1.0, 1.0, 0.3, 0.5, 0.5], dtype=np.float32),
        "circle":      np.array([1.0, 1.0, 0.3, 0.5, 0.5], dtype=np.float32),
    }
    RESIDUAL_L2_PENALTY = 0.0   # v4: removed — prior no longer fights residual

    def __init__(
        self,
        morph: dict,
        *,
        task: str = "hover",
        alpha: float | None = None,
        num_envs: int = 1,
        max_steps: int = 600,
        device: str = "cpu",
        seed: int | None = None,
    ):
        if task not in TASK_NAMES:
            raise ValueError(f"task {task!r} not in {TASK_NAMES}")
        self.task_name = task
        self.task_id = TASK_NAMES.index(task)
        # If alpha not supplied, use the per-task default.
        self.alpha = float(self.TASK_ALPHA[task] if alpha is None else alpha)
        gpos, gyaw, spos, xb, random_init = _task_gate_config(task)
        self._task_target = torch.tensor(gpos[0], dtype=torch.float32)  # hover target if applicable

        # Stash morph + derived params; HoverPrior built post-super().__init__
        # because we need self.dev / self.dtype after super().
        self._morph = morph
        self._cmaes_params_np = np.asarray(morph["cmaes_params"], dtype=np.float32)
        self._morph_features_np = np.asarray(morph["morph_features"], dtype=np.float32)
        if self._morph_features_np.shape[0] != MORPH_FEAT_DIM:
            raise ValueError(
                f"morph_features must be {MORPH_FEAT_DIM}d, got "
                f"{self._morph_features_np.shape[0]}d"
            )

        # Build the parent gate env. For hover we use a single "gate" at
        # HOVER_TARGET_NED so the parent's gate-distance reward becomes a
        # hover-tracking signal. Other tasks load their GATE_CONFIGS entry.
        is_hover = (task == "hover")
        super().__init__(
            num_envs=num_envs,
            propellers=morph["propellers"],
            gates_pos=gpos,
            gate_yaw=gyaw,
            start_pos=spos,
            x_bounds=xb,
            y_bounds=(-20.0, 20.0),
            z_bounds=(-5.0, 0.5),
            initialize_at_random_gates=random_init,
            seed=seed,
            device=device,
            max_steps=max_steps,
            motor_limit=1.0,
            upright_bonus=0.0 if is_hover else UPRIGHT_BONUS,
            tilt_terminate_cos=TILT_TERMINATE_COS,
            extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
            velocity_reward_coef=0.0 if is_hover else VELOCITY_REWARD_COEF,
            altitude_floor_z=ALTITUDE_FLOOR_Z,
            altitude_floor_coef=0.0 if is_hover else ALTITUDE_FLOOR_COEF,
        )

        # ---- build the analytical prior on this morph ------------------
        params_dict = derive_reference_params(
            propellers=morph["propellers"], mass=morph["mass"],
            inertia=morph["inertia"], prop_size=morph["prop_size"],
            gravity=self.drone_sim.g,
        )
        twr = morph.get("twr", None)
        self.prior = HoverPrior(
            propellers=morph["propellers"], params=params_dict,
            target_ned=self.HOVER_TARGET_NED,
            gravity=self.drone_sim.g, action_scale=0.4, twr=twr,
            device=self.dev, dtype=self.dtype,
        )
        # Apply per-task gain scaling. Trim stays as-trained; the 5 PD
        # gains (k_alt_p, k_alt_d, k_tilt, k_rate, k_yaw_rate) are scaled
        # so the prior provides only the stability terms the task needs.
        N = len(morph["propellers"])
        scaled = self._cmaes_params_np.copy()
        scaled[N:N + 5] *= self.TASK_PRIOR_GAIN_SCALE[task]
        self._cmaes_params_t = torch.as_tensor(
            scaled, device=self.dev, dtype=self.dtype,
        ).unsqueeze(0)  # (1, N+5) — broadcasts against (B, ...)

        # Hover-equivalent normalised motor state for resets — same fix
        # that took the library build pass rate from 40% to 100%.
        w_lo = self.prior._w_min
        w_hi = self.prior._w_max
        W_hover = math.sqrt(self.drone_sim.g / (self.prior._k_w * self.num_motors))
        self._w_hover_norm = float(
            (2.0 * W_hover - (w_hi + w_lo)) / max(w_hi - w_lo, 1e-6)
        )

        # ---- expand observation space ----------------------------------
        base_dim = self.observation_space.shape[0]
        self._base_obs_dim = base_dim
        self._full_obs_dim = base_dim + NUM_TASKS + MORPH_FEAT_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._full_obs_dim,), dtype=np.float32,
        )
        # task_one_hot and morph_features are static per env
        self._task_oh = np.zeros(NUM_TASKS, dtype=np.float32)
        self._task_oh[self.task_id] = 1.0
        self._tail = np.concatenate(
            [self._task_oh, self._morph_features_np]
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # Action: residual → residual + prior, all on device
    # ------------------------------------------------------------------

    def step_async(self, residual_actions: np.ndarray) -> None:
        """Compose total action = effort_to_action(prior_effort + α·residual)."""
        residual = torch.as_tensor(
            residual_actions, device=self.dev, dtype=self.dtype,
        )
        # prior_effort matches the parent's world_states layout (12+N)
        effort = self.prior.prior_effort(self.world_states, self._cmaes_params_t)
        total_action = self.prior.effort_to_action(effort + self.alpha * residual)
        # parent's step_async stores actions_t; we bypass it and write directly
        self.prev_actions_t = self.actions_t.clone()
        self.actions_t = total_action

    # ------------------------------------------------------------------
    # Observation: append task_one_hot + morph_features
    # ------------------------------------------------------------------

    def _expand_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Concatenate static tail to the (E, base_dim) gate-env obs."""
        if base_obs.ndim == 1:
            return np.concatenate([base_obs, self._tail])
        tail = np.broadcast_to(self._tail, (base_obs.shape[0], self._tail.shape[0]))
        return np.concatenate([base_obs, tail], axis=1)

    def reset(self) -> np.ndarray:
        base = super().reset()
        # Replace motor state with hover-equivalent so high-TWR morphs
        # don't explode at t=0 (parent inits motor_w to 0 = mid-throttle).
        self.world_states[:, 12:12 + self.num_motors] = self._w_hover_norm
        # Re-pull obs after our motor-state edit (gate obs reads
        # world_states[:, 12:12+N] for motor speeds).
        self._update_obs()
        base = self._obs_t.cpu().numpy()
        return self._expand_obs(base)

    def step_wait(self):
        base_obs, rewards, dones, infos = super().step_wait()
        if self.task_name == "hover":
            # Parent's gate-distance reward is ~0 when the prior already
            # holds the drone at target — no gradient for the residual.
            # Override with 27_v4's shaped hover reward.
            ws = self.world_states
            diverged = ~torch.isfinite(ws).all(dim=1)
            target = self.gate_pos_t[0]
            rewards_t = _hover_reward(ws, target, diverged)
            rewards = rewards_t.cpu().numpy().astype(np.float32)
        # Trajectory tasks (figure8/slalom/shuttle-run/circle) inherit the
        # parent's gate-distance + velocity + upright + altitude shaping
        # unchanged. The prior provides altitude tracking + light rate
        # damping; attitude leveling is disabled per task so the residual
        # can bank freely.
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = self._expand_obs(
                    info["terminal_observation"]
                )
        return self._expand_obs(base_obs), rewards, dones, infos


def _hover_reward(world_states: torch.Tensor, target: torch.Tensor,
                  diverged: torch.Tensor) -> torch.Tensor:
    """Hover reward matching 27_v4 (max ≈ 0.0125/step → ~+15 / 1200 steps)."""
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
        downward_vel > 0.0,
        torch.exp(-1.5 * downward_vel),
        torch.ones_like(downward_vel),
    )
    upright = (phi.cos() * theta.cos()).clamp(min=0.0)

    rew = ((pos_z * 0.30) + (pos_xy * 0.40) + (rate_term * 0.10)
           + (climb_term * 0.05) + (upright * 0.15)) * 0.0125
    return torch.where(diverged, torch.full_like(rew, -1.0), rew)
