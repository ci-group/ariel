"""Train a PPO shuttle-run racing controller for a drone loaded from a DroneBlueprint JSON.

Run:
    uv run examples/spear/21_shuttlerun.py --blueprint path/to/blueprint.json
    uv run examples/spear/21_shuttlerun.py --blueprint path/to/blueprint.json --device cuda:0

Mirrors the structure of 18_hover.py but trains on the `backandforth` gate track
using `TorchDroneGateEnv` with v4-style reward shaping (upright bonus,
velocity-toward-gate bonus, soft altitude floor). Tilt-based episode
termination is disabled so evolved morphologies can recover from spins.

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAME    = "shuttlerun"
GATE_CFG_KEY = "backandforth"

curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description=f"PPO {TASK_NAME} training from a DroneBlueprint JSON")
parser.add_argument("--blueprint", required=True, help="Path to DroneBlueprint JSON")
parser.add_argument("--device", default="cpu", help="Torch device: 'cpu' or 'cuda:0'")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/{TASK_NAME}_ppo/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS      = 5_000_000
PPO_NUM_ENVS   = 64
PPO_N_STEPS    = 1024
PPO_BATCH_SIZE = PPO_N_STEPS * PPO_NUM_ENVS // 8

# Reward shaping (matches 27_train_rl_hex_mtrl_v4 racing tasks).
UPRIGHT_BONUS         = 0.01
EXTRA_YAW_RATE_PEN    = 0.005
VELOCITY_REWARD_COEF  = 0.005
ALTITUDE_FLOOR_Z      = -0.5
ALTITUDE_FLOOR_COEF   = 0.5
TILT_TERMINATE_COS    = 0.0   # disabled — allow recovery from spins
GATES_AHEAD           = 2

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

cfg = GATE_CONFIGS[GATE_CFG_KEY]
gpos = np.asarray(cfg.gate_pos, dtype=np.float64)
gyaw = np.asarray(cfg.gate_yaw, dtype=np.float64)
spos = np.asarray(cfg.starting_pos, dtype=np.float64)
xb   = tuple(np.asarray(cfg.x_bounds, dtype=np.float64).tolist())
yb   = tuple(np.asarray(cfg.y_bounds, dtype=np.float64).tolist())
zb   = tuple(np.asarray(cfg.z_bounds, dtype=np.float64).tolist())
console.log(f"Task: {TASK_NAME}  |  gates: {len(gpos)}  |  start: {spos.tolist()}")

_raw_env = TorchDroneGateEnv(
    num_envs=PPO_NUM_ENVS,
    propellers=propellers,
    gates_pos=gpos,
    gate_yaw=gyaw,
    start_pos=spos,
    x_bounds=xb,
    y_bounds=yb,
    z_bounds=zb,
    gates_ahead=GATES_AHEAD,
    device=args.device,
    dt=0.01,
    seed=args.seed,
    initialize_at_random_gates=True,    # random gate init aids robustness
    upright_bonus=UPRIGHT_BONUS,
    tilt_terminate_cos=TILT_TERMINATE_COS,
    extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
    velocity_reward_coef=VELOCITY_REWARD_COEF,
    altitude_floor_z=ALTITUDE_FLOOR_Z,
    altitude_floor_coef=ALTITUDE_FLOOR_COEF,
)
env = VecNormalize(_raw_env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=0.999)

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

console.rule(f"[bold blue]Training PPO {TASK_NAME} policy")
t0 = time.time()
model.learn(total_timesteps=PPO_STEPS)
console.log(f"Training done in {time.time() - t0:.1f}s")

DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)

policy_path = DATA / f"{TASK_NAME}_policy_{curr_time}.zip"
model.save(str(policy_path))
console.log(f"Policy  → {policy_path}")

vecnorm_path = DATA / f"vecnormalize_{curr_time}.pkl"
env.save(str(vecnorm_path))
console.log(f"VecNormalize → {vecnorm_path}")

bp.save_json(DATA / f"blueprint_{curr_time}.json")
console.log(f"Blueprint → {DATA / f'blueprint_{curr_time}.json'}")
