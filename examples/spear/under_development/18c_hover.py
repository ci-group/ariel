"""Train a PPO hover controller for an asymmetrical drone loaded from a DroneBlueprint JSON.

Run:
    uv run examples/spear/18_hover.py --blueprint path/to/blueprint.json
    uv run examples/spear/18_hover.py --blueprint path/to/blueprint.json --device cuda:0

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.drone.drone_configuration import DroneConfiguration
from ariel.simulation.drone.dynamics_params import W_MAX_N, W_MIN_N, derive_reference_params
from ariel.simulation.tasks.torch_drone_gate_env import _build_torch_dynamics

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description="PPO hover training from a DroneBlueprint JSON")
parser.add_argument("--blueprint", required=True, help="Path to DroneBlueprint JSON")
parser.add_argument("--device", default="cpu", help="Torch device: 'cpu' or 'cuda:0'")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/hover_ppo/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS      = 5_000_000
PPO_NUM_ENVS   = 32
PPO_N_STEPS    = 1024  # Increased to capture longer trajectory horizons
PPO_BATCH_SIZE = PPO_N_STEPS * PPO_NUM_ENVS // 8

HOVER_TARGET_NED = np.array([0.0, 0.0, -1.5], dtype=np.float32)
GRAVITY = 9.81


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_params(propellers: list) -> tuple[dict, int]:
    """Extract dynamics params from propeller list without constructing DroneSimulator
    (which runs expensive SymPy lambdification we don't need here)."""
    config = DroneConfiguration(propellers)
    params = derive_reference_params(
        propellers=config.propellers,
        mass=float(config.mass),
        inertia=np.asarray(config.inertia_matrix),
        prop_size=propellers[0].get("propsize", 2),
        gravity=GRAVITY,
    )
    return params, config.num_motors


def _compute_u_hover(params: dict, num_motors: int) -> float:
    """Action that drives all motors to vertical hover thrust at steady state."""
    k_w, k, w_min, w_max = params["k_w"], params["k"], params["w_min"], params["w_max"]
    W_hover = math.sqrt(GRAVITY / (k_w * num_motors))
    z = float(np.clip((W_hover - w_min) / (w_max - w_min), 0.0, 1.0))
    disc = (1.0 - k) ** 2 + 4.0 * k * z * z
    U_hover = (-(1.0 - k) + math.sqrt(max(disc, 0.0))) / (2.0 * k)
    return float(np.clip(2.0 * U_hover - 1.0, -1.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Hover environment
# ─────────────────────────────────────────────────────────────────────────────

class TorchDroneHoverEnv(VecEnv):
    """Vectorised hover task.

    Observation: [pos_error(3), vel(3), fwd_vec(3), up_vec(3), rates(3), motor_states(N)]
                 where pos_error = pos - target.
    Reward:      Bounded exponential rewards for position, velocity, and rates.
                 Heavy penalty for crashing/OOB.
    Done:        out-of-bounds (±5 m), numerical divergence, or max_steps reached.
    """

    def __init__(
        self,
        propellers: list,
        num_envs: int,
        target_pos: np.ndarray,
        device: str = "cpu",
        dt: float = 0.01,
        max_steps: int = 1200,
    ) -> None:
        params, num_motors = _build_params(propellers)
        self.num_motors = num_motors
        self.dt = dt
        self.max_steps = max_steps
        self.dev = torch.device(device)
        self.dtype = torch.float32

        self._dynamics = _build_torch_dynamics(
            params, num_motors, GRAVITY, self.dev, self.dtype,
        )
        self.target = torch.tensor(target_pos, device=self.dev, dtype=self.dtype)
        self._u_hover = _compute_u_hover(params, num_motors)

        n = self.num_motors
        state_dim = 12 + n # Internal state uses Euler angles to match dynamics engine
        obs_dim = 15 + n   # Observation uses Fwd/Up vectors instead of Euler
        
        self.states      = torch.zeros((num_envs, state_dim), device=self.dev, dtype=self.dtype)
        self.step_counts = torch.zeros(num_envs, device=self.dev, dtype=torch.long)
        self._acts       = torch.zeros((num_envs, n), device=self.dev, dtype=self.dtype)

        VecEnv.__init__(
            self, num_envs,
            spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float64),
            spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float64),
        )

    def _reset_envs(self, mask: torch.Tensor) -> None:
        n = int(mask.sum().item())
        if n == 0:
            return
        pos = self.target.unsqueeze(0).expand(n, -1).clone()
        pos += (torch.rand((n, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.4
        vel    = (torch.rand((n, 3), device=self.dev, dtype=self.dtype) - 0.5) * 0.2
        att    = torch.zeros((n, 3), device=self.dev, dtype=self.dtype)
        att[:, :2] = (torch.rand((n, 2), device=self.dev, dtype=self.dtype) - 0.5) * (math.pi / 9)
        att[:, 2]  = (torch.rand(n, device=self.dev, dtype=self.dtype) - 0.5) * math.pi
        rates  = torch.zeros((n, 3), device=self.dev, dtype=self.dtype)
        motors = torch.zeros((n, self.num_motors), device=self.dev, dtype=self.dtype)
        
        self.states[mask]      = torch.cat([pos, vel, att, rates, motors], dim=1)
        self.step_counts[mask] = 0

    def _make_obs(self) -> np.ndarray:
        s = self.states
        pos_err = s[:, 0:3] - self.target.unsqueeze(0)
        vel = s[:, 3:6]
        euler = s[:, 6:9]
        rates = s[:, 9:12]
        motors = s[:, 12:]

        # Convert Euler angles to Forward and Up vectors to avoid discontinuities
        phi = euler[:, 0]
        theta = euler[:, 1]
        psi = euler[:, 2]

        cp, sp = torch.cos(phi), torch.sin(phi)
        ct, st = torch.cos(theta), torch.sin(theta)
        cps, sps = torch.cos(psi), torch.sin(psi)

        # Forward vector (X-axis)
        fwd_x = ct * cps
        fwd_y = ct * sps
        fwd_z = -st
        fwd = torch.stack([fwd_x, fwd_y, fwd_z], dim=1)

        # Down vector (Z-axis in NED), so Up is negative Down
        down_x = cp * st * cps + sp * sps
        down_y = cp * st * sps - sp * cps
        down_z = cp * ct
        up = -torch.stack([down_x, down_y, down_z], dim=1)

        return torch.cat([pos_err, vel, fwd, up, rates, motors], dim=1).cpu().numpy()

    def reset(self) -> np.ndarray:
        self._reset_envs(torch.ones(self.num_envs, device=self.dev, dtype=torch.bool))
        return self._make_obs()

    def step_async(self, actions: np.ndarray) -> None:
        raw_actions = torch.as_tensor(actions, device=self.dev, dtype=self.dtype).clamp(-1.0, 1.0)
        self._acts = (self._u_hover + raw_actions * 0.4).clamp(-1.0, 1.0)

    def step_wait(self):
        state_dot = self._dynamics(self.states.T, self._acts.T).T
        self.states = self.states + self.dt * state_dot
        self.step_counts += 1

# ─────────────────────────────────────────────────────────────────
        # UPDATED REWARD LOGIC FOR ASYMMETRIC DRONES
        # ─────────────────────────────────────────────────────────────────
        pos   = self.states[:, 0:3]
        vel   = self.states[:, 3:6]
        rates = self.states[:, 9:12]  # [roll_rate, pitch_rate, yaw_rate]
        
        # 1. Split position error: Be strict about altitude (Z), lenient about drift (X,Y)
        pos_err = pos - self.target.unsqueeze(0)
        dist_xy = torch.norm(pos_err[:, :2], dim=1)
        dist_z  = torch.abs(pos_err[:, 2])

        pos_reward_xy = torch.exp(-0.5 * dist_xy)
        pos_reward_z  = torch.exp(-2.5 * dist_z)  # Higher decay = tighter altitude control

        # 2. Penalize all angular rates, including yaw spin
        roll_pitch_rates = torch.norm(rates[:, :2], dim=1)
        yaw_rate         = rates[:, 2].abs()
        rate_penalty     = torch.exp(-0.5 * roll_pitch_rates) * torch.exp(-1.0 * yaw_rate)

        # 3. Penalize downward velocity specifically (NED frame: +Z is DOWN)
        downward_vel     = vel[:, 2]
        climb_penalty    = torch.where(downward_vel > 0.0, torch.exp(-1.5 * downward_vel), 1.0)

        # Combine into a well-behaved total reward (max possible close to 0.1 per step)
        rewards = ((pos_reward_z * 0.5) + (pos_reward_xy * 0.2) + (rate_penalty * 0.2) + (climb_penalty * 0.1)) * 0.1

        # Action jitter penalty
        act_norm = torch.norm(self._acts, dim=1)
        rewards -= 0.005 * act_norm

        # Additive spin penalty (NED yaw rate, rad/s) — small so it shapes
        # without dominating the ~0.1/step positive reward early on.
        rewards -= 0.005 * yaw_rate

        oob      = (pos.abs() > 5.0).any(dim=1)
        diverged = ~torch.isfinite(self.states).all(dim=1)
        dones    = oob | diverged | (self.step_counts >= self.max_steps)

        rewards[oob | diverged] = -1.0

        if dones.any():
            self._reset_envs(dones)

        return (
            self._make_obs(),
            rewards.cpu().numpy().astype(np.float32),
            dones.cpu().numpy(),
            [{} for _ in range(self.num_envs)],
        )

    def close(self): pass
    def seed(self, seed=None): return []
    def get_attr(self, attr_name, indices=None): raise AttributeError(attr_name)
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *args, indices=None, **kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def render(self, mode="human"): return {}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

bp_path = Path(args.blueprint)
if not bp_path.exists():
    raise FileNotFoundError(f"Blueprint not found: {bp_path}")

bp = DroneBlueprint.load_json(bp_path)
console.log(f"Loaded blueprint: {bp_path}")
console.print(bp.summary())

propellers = blueprint_to_propellers(bp, convention="ned")
console.log(f"Motors: {len(propellers)}")

_raw_env = TorchDroneHoverEnv(
    propellers=propellers,
    num_envs=PPO_NUM_ENVS,
    target_pos=HOVER_TARGET_NED,
    device=args.device,
    dt=0.01,
)
env = VecNormalize(_raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0, gamma=0.999)

# Expanded Network Architecture to untangle asymmetric dynamics
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    n_steps=PPO_N_STEPS,
    batch_size=PPO_BATCH_SIZE,
    gamma=0.999,
    device=args.device,
    verbose=1,
)

console.rule("[bold blue]Training PPO hover policy")
t0 = time.time()
model.learn(total_timesteps=PPO_STEPS)
console.log(f"Training done in {time.time() - t0:.1f}s")

DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)

policy_path = DATA / f"hover_policy_{curr_time}.zip"
model.save(str(policy_path))
console.log(f"Policy  → {policy_path}")

vecnorm_path = DATA / f"vecnormalize_{curr_time}.pkl"
env.save(str(vecnorm_path))
console.log(f"VecNormalize → {vecnorm_path}")

bp.save_json(DATA / f"blueprint_{curr_time}.json")
console.log(f"Blueprint → {DATA / f'blueprint_{curr_time}.json'}")