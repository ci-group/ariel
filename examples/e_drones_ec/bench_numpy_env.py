"""Benchmark: DroneGateEnv — SymPy-lambdified NumPy dynamics (CPU only).

Measures raw environment throughput independently of any RL training loop.
Compare results against bench_torch_env.py to quantify the GPU backend gain.

Usage:
    uv run examples/e_drones_ec/bench_numpy_env.py
    uv run examples/e_drones_ec/bench_numpy_env.py --num-envs 256 --steps 500
"""
import argparse
import time

import numpy as np
from rich.console import Console
from rich.table import Table

from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.drone_simulator import DroneSimulator

console = Console()

parser = argparse.ArgumentParser(description="Benchmark NumPy DroneGateEnv throughput")
parser.add_argument("--num-envs", type=int, default=500,
                    help="Number of parallel environments (default 500)")
parser.add_argument("--steps",    type=int, default=1000,
                    help="Number of env steps to time (default 1000)")
parser.add_argument("--warmup",   type=int, default=50,
                    help="Warm-up steps excluded from timing (default 50)")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

# ── build a standard 6-arm drone ──────────────────────────────────────────
np.random.seed(args.seed)

# 6 arms at 60° spacing, mid-range parameters
N_ARMS  = 6
angles  = np.linspace(0, 2 * np.pi, N_ARMS, endpoint=False)
# Columns: arm_length, arm_az, arm_elev, motor_az, motor_pitch, spin_dir
arms = np.column_stack([
    np.full(N_ARMS, 0.11),                           # arm length (m)
    angles,                                          # arm azimuth
    np.zeros(N_ARMS),                               # arm elevation
    np.zeros(N_ARMS),                               # motor disc azimuth
    np.zeros(N_ARMS),                               # motor disc pitch
    np.tile([0, 1], N_ARMS // 2),                  # alternating spin directions
])

bp          = spherical_angular_to_blueprint(arms, propsize=2)
propellers  = blueprint_to_propellers(bp, convention="ned")

# ── build env ─────────────────────────────────────────────────────────────
console.rule("[bold]NumPy DroneGateEnv benchmark")
console.log(f"num_envs={args.num_envs}  steps={args.steps}  warmup={args.warmup}")

env = DroneGateEnv(
    num_envs=args.num_envs,
    propellers=propellers,
    seed=args.seed,
)
n_motors = env.num_motors
console.log(f"motors={n_motors}  obs_len={env.obs_len}  dt={env.dt}")

obs = env.reset()

# ── warm-up ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
for _ in range(args.warmup):
    actions = rng.uniform(-1.0, 1.0, (args.num_envs, n_motors)).astype(np.float32)
    env.step_async(actions)
    env.step_wait()

# ── timed loop ────────────────────────────────────────────────────────────
# Time three sub-sections separately using manual splits.
t_dynamics  = 0.0
t_obs       = 0.0
t_total     = 0.0

# Monkey-patch to measure sub-timings.
_orig_step_wait = env.step_wait.__func__  # unbound

total_rewards    = np.zeros(args.num_envs)
total_resets     = 0
step_times       = []

t_start_all = time.perf_counter()
for i in range(args.steps):
    actions = rng.uniform(-1.0, 1.0, (args.num_envs, n_motors)).astype(np.float32)
    env.step_async(actions)

    t0 = time.perf_counter()
    obs, rew, dones, infos = env.step_wait()
    t1 = time.perf_counter()

    step_times.append(t1 - t0)
    total_rewards += rew
    total_resets  += int(dones.sum())

t_end_all = time.perf_counter()
wall_time = t_end_all - t_start_all

# ── results ───────────────────────────────────────────────────────────────
total_transitions = args.steps * args.num_envs
sps               = total_transitions / wall_time
step_ms_mean      = np.mean(step_times) * 1e3
step_ms_p95       = np.percentile(step_times, 95) * 1e3
step_ms_p99       = np.percentile(step_times, 99) * 1e3

table = Table(title="NumPy DroneGateEnv — results", show_header=True)
table.add_column("Metric",        style="cyan",  no_wrap=True)
table.add_column("Value",         style="green", justify="right")
table.add_row("Backend",          "NumPy / SymPy-lambdified (CPU)")
table.add_row("num_envs",         str(args.num_envs))
table.add_row("steps timed",      str(args.steps))
table.add_row("total transitions",f"{total_transitions:,}")
table.add_row("wall time (s)",    f"{wall_time:.3f}")
table.add_row("steps / sec",      f"{sps:,.0f}")
table.add_row("transitions / sec",f"{sps:,.0f}")
table.add_row("step_wait mean",   f"{step_ms_mean:.3f} ms")
table.add_row("step_wait p95",    f"{step_ms_p95:.3f} ms")
table.add_row("step_wait p99",    f"{step_ms_p99:.3f} ms")
table.add_row("env resets",       str(total_resets))
console.print(table)
console.log("[bold]Tip:[/bold] run bench_torch_env.py with matching flags to compare.")
