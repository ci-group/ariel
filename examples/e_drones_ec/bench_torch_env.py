"""Benchmark: TorchDroneGateEnv — hand-written PyTorch dynamics (CPU or CUDA).

Measures raw environment throughput independently of any RL training loop.
Compare results against bench_numpy_env.py to quantify the GPU backend gain.

Usage:
    # CPU (apples-to-apples vs NumPy):
    uv run examples/e_drones_ec/bench_torch_env.py --device cpu

    # CUDA:
    uv run examples/e_drones_ec/bench_torch_env.py --device cuda:0

    # Match the RL run config:
    uv run examples/e_drones_ec/bench_torch_env.py --device cuda:0 --num-envs 500 --steps 1000
"""
import argparse
import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint

console = Console()

parser = argparse.ArgumentParser(description="Benchmark TorchDroneGateEnv throughput")
parser.add_argument("--num-envs", type=int, default=500,
                    help="Number of parallel environments (default 500)")
parser.add_argument("--steps",    type=int, default=1000,
                    help="Number of env steps to time (default 1000)")
parser.add_argument("--warmup",   type=int, default=50,
                    help="Warm-up steps excluded from timing (default 50)")
parser.add_argument("--device",   default="cuda:0",
                    help="Torch device: cpu or cuda:0 (default cuda:0)")
parser.add_argument("--seed",     type=int, default=42)
args = parser.parse_args()

# ── build a standard 6-arm drone ──────────────────────────────────────────
np.random.seed(args.seed)
torch.manual_seed(args.seed)

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
console.rule(f"[bold]TorchDroneGateEnv benchmark ({args.device})")
console.log(f"num_envs={args.num_envs}  steps={args.steps}  warmup={args.warmup}")

env = TorchDroneGateEnv(
    num_envs=args.num_envs,
    propellers=propellers,
    device=args.device,
    seed=args.seed,
)
n_motors = env.num_motors
console.log(f"motors={n_motors}  state_len={env.state_len}  dt={env.dt}  device={env.dev}")

obs = env.reset()

# ── warm-up ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(args.seed)
for _ in range(args.warmup):
    actions = rng.uniform(-1.0, 1.0, (args.num_envs, n_motors)).astype(np.float32)
    env.step_async(actions)
    env.step_wait()

# If on CUDA, wait for all warm-up kernels to finish before timing starts.
if args.device != "cpu":
    torch.cuda.synchronize()

# ── timed loop ────────────────────────────────────────────────────────────
total_rewards = np.zeros(args.num_envs)
total_resets  = 0
step_times    = []

# Sub-section timers — split step_wait into dynamics vs. cpu-transfer.
t_dynamics_total  = 0.0
t_transfer_total  = 0.0

t_start_all = time.perf_counter()
for i in range(args.steps):
    actions = rng.uniform(-1.0, 1.0, (args.num_envs, n_motors)).astype(np.float32)
    env.step_async(actions)

    # ---- dynamics on device (ends at the synchronize point) ----
    t0 = time.perf_counter()

    # Run physics + gate logic (all on device).
    act = torch.as_tensor(actions, device=env.dev, dtype=env.dtype)
    if env.action_filter_alpha < 1.0:
        env.filtered_acts = (
            env.action_filter_alpha * act
            + (1.0 - env.action_filter_alpha) * env.filtered_acts
        )
        act_in = env.filtered_acts
    else:
        act_in = act

    fs   = env.world_states
    fsd  = env._dynamics(fs.T, act_in.T).T
    new_states = fs + env.dt * fsd

    diverged = ~(torch.isfinite(new_states) & (new_states.abs() < 1e6)).all(dim=1)
    new_states[diverged] = fs[diverged]
    env.step_counts += 1

    pos_old = fs[:, 0:3]
    pos_new = new_states[:, 0:3]
    tg      = env.target_gates % env.num_gates
    gpos    = env.gate_pos_t[tg]
    gyaw    = env.gate_yaw_t[tg]

    d_old   = (pos_old - gpos).norm(dim=1)
    d_new   = (pos_new - gpos).norm(dim=1)
    rewards = d_old - d_new - 0.001 * new_states[:, 9:12].norm(dim=1)

    nx = gyaw.cos(); ny = gyaw.sin()
    proj_old   = (pos_old[:, 0] - gpos[:, 0]) * nx + (pos_old[:, 1] - gpos[:, 1]) * ny
    proj_new   = (pos_new[:, 0] - gpos[:, 0]) * nx + (pos_new[:, 1] - gpos[:, 1]) * ny
    crossed    = (proj_old < 0) & (proj_new > 0)
    in_gate    = (pos_new - gpos).abs().amax(dim=1) < (env.gate_size / 2)
    gate_passed = crossed & in_gate

    final_gate = gate_passed & (env.target_gates == env.num_gates - 1)
    rewards[final_gate] += 10.0

    oob = (
        (new_states[:, 0] < env.x_bounds[0]) | (new_states[:, 0] > env.x_bounds[1]) |
        (new_states[:, 1] < env.y_bounds[0]) | (new_states[:, 1] > env.y_bounds[1]) |
        (new_states[:, 2] < env.z_bounds[0]) | (new_states[:, 2] > env.z_bounds[1])
    )
    rewards[oob | diverged] = -10.0

    max_steps_reached = env.step_counts >= env.max_steps
    dones = max_steps_reached | oob | diverged

    env.target_gates[gate_passed] = (env.target_gates[gate_passed] + 1) % env.num_gates
    env.num_gates_passed[gate_passed] += 1
    env.world_states = new_states
    env._reset_envs(dones)
    env._update_obs()

    # Synchronize before measuring the transfer phase.
    if args.device != "cpu":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    t_dynamics_total += t1 - t0

    # ---- CPU transfer ----
    obs_np     = env._obs_t.cpu().numpy()
    rewards_np = rewards.cpu().numpy()
    dones_np   = dones.cpu().numpy()
    oob_np     = oob.cpu().numpy()
    gp_np      = env.num_gates_passed.cpu().numpy()
    gate_passed_np  = gate_passed.cpu().numpy()
    max_steps_np    = max_steps_reached.cpu().numpy()

    if args.device != "cpu":
        torch.cuda.synchronize()
    t2 = time.perf_counter()
    t_transfer_total += t2 - t1

    step_times.append(t2 - t0)
    total_rewards += rewards_np
    total_resets  += int(dones_np.sum())

t_end_all = time.perf_counter()
wall_time = t_end_all - t_start_all

# ── results ───────────────────────────────────────────────────────────────
total_transitions = args.steps * args.num_envs
sps               = total_transitions / wall_time
step_ms_mean      = np.mean(step_times) * 1e3
step_ms_p95       = np.percentile(step_times, 95) * 1e3
step_ms_p99       = np.percentile(step_times, 99) * 1e3
dyn_pct           = 100.0 * t_dynamics_total / wall_time
xfr_pct           = 100.0 * t_transfer_total / wall_time

# CUDA memory stats
mem_str = "N/A"
if args.device != "cpu" and torch.cuda.is_available():
    alloc_mb  = torch.cuda.memory_allocated()  / 1024**2
    reserv_mb = torch.cuda.memory_reserved()   / 1024**2
    mem_str   = f"{alloc_mb:.1f} MB alloc / {reserv_mb:.1f} MB reserved"

table = Table(title=f"TorchDroneGateEnv — results ({args.device})", show_header=True)
table.add_column("Metric",         style="cyan",  no_wrap=True)
table.add_column("Value",          style="green", justify="right")
table.add_row("Backend",           f"PyTorch ({args.device})")
table.add_row("num_envs",          str(args.num_envs))
table.add_row("steps timed",       str(args.steps))
table.add_row("total transitions", f"{total_transitions:,}")
table.add_row("wall time (s)",     f"{wall_time:.3f}")
table.add_row("steps / sec",       f"{sps:,.0f}")
table.add_row("transitions / sec", f"{sps:,.0f}")
table.add_row("step_wait mean",    f"{step_ms_mean:.3f} ms")
table.add_row("step_wait p95",     f"{step_ms_p95:.3f} ms")
table.add_row("step_wait p99",     f"{step_ms_p99:.3f} ms")
table.add_row("  └ dynamics",      f"{t_dynamics_total*1e3/args.steps:.3f} ms/step  ({dyn_pct:.1f}%)")
table.add_row("  └ CPU transfer",  f"{t_transfer_total*1e3/args.steps:.3f} ms/step  ({xfr_pct:.1f}%)")
table.add_row("env resets",        str(total_resets))
table.add_row("CUDA memory",       mem_str)
console.print(table)
console.log("[bold]Tip:[/bold] run bench_numpy_env.py with matching --num-envs/--steps to compare.")
