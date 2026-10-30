"""Train a PPO figure-8 racing controller for a drone loaded from a DroneBlueprint JSON.

Run:
    uv run examples/spear/19_figure8.py --blueprint path/to/blueprint.json
    uv run examples/spear/19_figure8.py --blueprint path/to/blueprint.json --device cuda:0

Mirrors the structure of 18_hover.py but trains on the `figure8` gate track
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

TASK_NAME    = "figure8"
GATE_CFG_KEY = "figure8"

curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(description=f"PPO {TASK_NAME} training from a DroneBlueprint JSON")
parser.add_argument("--blueprint", required=True, help="Path to DroneBlueprint JSON")
parser.add_argument("--device", default="cpu", help="Torch device: 'cpu' or 'cuda:0'")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/{TASK_NAME}_ppo/{curr_time}")
parser.add_argument("--warmstart-dir", default=None,
                    help="Optional dir containing a hover policy (zip) + "
                         "vecnormalize (pkl), e.g. output of 18b_hover_gate.py. "
                         "When set, PPO is initialised from the hover weights, "
                         "VecNormalize stats are inherited, and reward shaping "
                         "switches to a sparse-ish profile (gate-pass + safety "
                         "terms only).")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS      = 20_000_000
PPO_NUM_ENVS   = 128
PPO_N_STEPS    = 2048
PPO_BATCH_SIZE = PPO_N_STEPS * PPO_NUM_ENVS // 8

# Reward shaping. Two profiles:
#  - "dense" (default, no warm-start): matches 27_train_rl_hex_mtrl_v4 racing
#    tasks. Drove the policy into a velocity-overshoot basin in earlier runs.
#  - "sparse" (when --warmstart-dir is set): only the env's intrinsic
#    distance-shaped progress (d_old - d_new) + gate-pass bonus + safety
#    terms (altitude floor, tilt termination). Per-step velocity & upright
#    bonuses removed so the hover-warm-started policy isn't pulled into the
#    "accelerate hard" basin.
if args.warmstart_dir is not None:
    UPRIGHT_BONUS         = 0.0
    EXTRA_YAW_RATE_PEN    = 0.0
    VELOCITY_REWARD_COEF  = 0.0
    ALTITUDE_FLOOR_Z      = -0.5
    ALTITUDE_FLOOR_COEF   = 0.5
    TILT_TERMINATE_COS    = 0.1   # ≈ 85° — kill flailing episodes
else:
    UPRIGHT_BONUS         = 0.01
    EXTRA_YAW_RATE_PEN    = 0.005
    VELOCITY_REWARD_COEF  = 0.005
    ALTITUDE_FLOOR_Z      = -0.5
    ALTITUDE_FLOOR_COEF   = 0.5
    TILT_TERMINATE_COS    = 0.0
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
    initialize_at_random_gates=False,   # canonical start for figure-8 closed loop
    upright_bonus=UPRIGHT_BONUS,
    tilt_terminate_cos=TILT_TERMINATE_COS,
    extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
    velocity_reward_coef=VELOCITY_REWARD_COEF,
    altitude_floor_z=ALTITUDE_FLOOR_Z,
    altitude_floor_coef=ALTITUDE_FLOOR_COEF,
)
# Wrap with VecNormalize. If a warm-start dir is given, inherit the obs-norm
# (and reward-norm) running stats from the hover run so the warm-started
# policy doesn't see a sudden obs distribution shift in the first rollouts.
if args.warmstart_dir is not None:
    ws_dir = Path(args.warmstart_dir)
    policy_zips = sorted(ws_dir.glob("*_policy_*.zip")) or sorted(ws_dir.glob("*.zip"))
    vn_pkls     = sorted(ws_dir.glob("vecnormalize_*.pkl"))
    if not policy_zips:
        raise FileNotFoundError(f"No *_policy_*.zip in warm-start dir: {ws_dir}")
    if not vn_pkls:
        raise FileNotFoundError(f"No vecnormalize_*.pkl in warm-start dir: {ws_dir}")
    policy_zip = policy_zips[-1]
    vn_pkl     = vn_pkls[-1]
    console.log(f"[warm-start] policy = {policy_zip}")
    console.log(f"[warm-start] vecnorm = {vn_pkl}")
    env = VecNormalize.load(str(vn_pkl), _raw_env)
    env.training    = True
    env.norm_reward = True
    env.clip_reward = 10.0
    env.gamma       = 0.999
    model = PPO.load(
        str(policy_zip),
        env=env,
        device=args.device,
        # NOTE: training-time hyperparams override the loaded ones so we can
        # tune them per-task without re-loading.
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        gamma=0.999,
        verbose=1,
    )
else:
    env = VecNormalize(_raw_env, norm_obs=True, norm_reward=True, clip_reward=10.0, gamma=0.999)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[64, 64]),
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
