"""PPO training for a 2-inch X-quad on the figure-8 gate-racing task.

Trains a 64×64 MLP policy with stable-baselines3 PPO on DroneGateEnv
(airevolve's vectorised gym environment). Saves a TensorBoard log, a
window-metrics CSV, and the trained policy checkpoint. Optionally renders
an mp4 rollout of the trained policy.

Run:
    # Quick smoke test (CPU):
    python examples/d_drones/6_train_rl_figure8.py \\
        --total-steps 1e5 --num-envs 50 --device cpu

    # Full training (GPU, ~30 min):
    python examples/d_drones/6_train_rl_figure8.py --total-steps 5e6

    # Headless (no video):
    python examples/d_drones/6_train_rl_figure8.py --total-steps 1e5 --no-video --device cpu

Requires: stable-baselines3, torch  (uv pip install stable-baselines3 torch)
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import torch

from ariel.simulation.tasks.drone_gate_env import (
    DroneGateEnv,
    gate_pos as DEFAULT_GATE_POS,
    gate_yaw as DEFAULT_GATE_YAW,
)
from ariel.simulation.drone.propeller_data import create_standard_propeller_config
import ariel.simulation.drone.controllers.utils as ctrl_utils

# ---------------------------------------------------------------------------
# Window-metrics callback (vendored for self-containedness)
# ---------------------------------------------------------------------------

class WindowMetricsCallback(BaseCallback):
    """Log gate-passes and episode stats once per ``window`` env steps to CSV."""

    def __init__(self, window: int = 100_000, csv_path: str = "window_metrics.csv",
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        self.window = window
        self.csv_path = csv_path
        self.last_log = 0
        self.gates_in_window = 0
        self.ep_rewards: list[float] = []
        self.ep_lens: list[int] = []
        os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestep", "gate_passes", "mean_ep_reward", "mean_ep_length", "n_episodes"]
            )

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            self.gates_in_window += info.get("gate_passes", 0)
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
                self.ep_lens.append(info["episode"]["l"])

        if self.num_timesteps - self.last_log >= self.window:
            mean_r = float(np.mean(self.ep_rewards)) if self.ep_rewards else float("nan")
            mean_l = float(np.mean(self.ep_lens)) if self.ep_lens else float("nan")
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    self.num_timesteps,
                    self.gates_in_window,
                    round(mean_r, 3),
                    round(mean_l, 1),
                    len(self.ep_rewards),
                ])
            self.last_log = self.num_timesteps
            self.gates_in_window = 0
            self.ep_rewards = []
            self.ep_lens = []
        return True

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="PPO on figure-8 drone gate-racing task")
parser.add_argument("--total-steps", type=float, default=5e6,
                    help="Total PPO timesteps (default 5e6)")
parser.add_argument("--num-envs", type=int, default=100,
                    help="Parallel environments (default 100)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--arm-length", type=float, default=0.11,
                    help="Arm length in metres (default 0.11)")
parser.add_argument("--save-dir", default="__data__/rl",
                    help="Output dir for logs, policy, and rollout video (default __data__/rl)")
parser.add_argument("--device", default="cuda:0",
                    help="Torch device (default cuda:0; pass cpu for portability)")
parser.add_argument("--tag", default="",
                    help="Extra tag appended to the run name")
parser.add_argument("--no-video", action="store_true",
                    help="Skip post-training rollout video")
parser.add_argument("--video-seconds", type=float, default=20.0,
                    help="Rollout duration for video in seconds (default 20)")
args = parser.parse_args()

total_steps = int(args.total_steps)
save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

run_name = f"figure8_prop{args.prop_size}"
if args.seed is not None:
    run_name += f"_seed{args.seed}"
if args.tag:
    run_name += f"_{args.tag}"
run_name += f"_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

print(
    f"=== RL figure-8 training: prop={args.prop_size}in arm={args.arm_length}m "
    f"envs={args.num_envs} steps={total_steps:_} seed={args.seed} device={args.device} ===",
    flush=True,
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

propeller_config = create_standard_propeller_config(
    "quad", arm_length=args.arm_length, prop_size=args.prop_size
)

env = DroneGateEnv(
    propellers=propeller_config,
    num_envs=args.num_envs,
    device=args.device,
    dt=0.01,
    seed=args.seed,
)

obs = env.reset()
print(
    f"obs shape: {obs.shape}  num_motors: {env.num_motors}  "
    f"mass: {env.drone_sim.mass:.3f} kg  motor_tau: {env.motor_tau}",
    flush=True,
)

# ---------------------------------------------------------------------------
# PPO model
# ---------------------------------------------------------------------------

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU,
    net_arch=dict(pi=[64, 64], vf=[64, 64]),
    log_std_init=0.0,
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=0,
    tensorboard_log=str(save_dir),
    n_steps=1000,
    batch_size=5000,
    n_epochs=10,
    gamma=0.999,
    learning_rate=3e-4,
    clip_range=0.2,
    ent_coef=0.01,
    device=args.device,
)

csv_path = str(save_dir / f"{run_name}_window_metrics.csv")
cb = WindowMetricsCallback(window=100_000, csv_path=csv_path)

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

t0 = time.time()
model.learn(
    total_timesteps=total_steps,
    tb_log_name=run_name,
    callback=cb,
    progress_bar=True,
)
elapsed = time.time() - t0

policy_path = str(save_dir / f"{run_name}_policy")
model.save(policy_path)
print(
    f"=== done in {elapsed:.1f}s ({total_steps / elapsed:.0f} steps/s). "
    f"Policy saved: {policy_path}.zip ===",
    flush=True,
)
print(f"Window-metrics CSV: {csv_path}", flush=True)

# ---------------------------------------------------------------------------
# Post-training rollout video
# ---------------------------------------------------------------------------

if not args.no_video:
    print(f"\nRendering {args.video_seconds}s rollout …", flush=True)
    dt_vid = 0.01
    n_steps = int(args.video_seconds / dt_vid)

    obs = env.reset()
    pos_all = np.zeros((n_steps, 3))
    quat_all = np.zeros((n_steps, 4))
    t_all = np.linspace(0, args.video_seconds, n_steps)

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        pos_all[step] = env.drone_sim.pos[0].cpu().numpy() if hasattr(env.drone_sim.pos, "cpu") \
            else env.drone_sim.pos[0]
        quat_all[step] = env.drone_sim.quat[0].cpu().numpy() if hasattr(env.drone_sim.quat, "cpu") \
            else env.drone_sim.quat[0]

    sDes_traj_all = np.zeros((n_steps, 16))
    sDes_traj_all[:, :3] = pos_all
    waypoints = DEFAULT_GATE_POS.astype(float)

    video_path = str(save_dir / f"{run_name}_rollout.mp4")
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    ctrl_utils.sameAxisAnimation(
        t_all, waypoints, pos_all, quat_all, sDes_traj_all, dt_vid,
        env.drone_sim.get_params(), 15, 3, 1, "NED",
        gate_pos=DEFAULT_GATE_POS,
        gate_yaw=DEFAULT_GATE_YAW,
        gate_size=1.5,
        save_path=video_path,
    )
    print(f"Rollout video saved: {video_path}", flush=True)
