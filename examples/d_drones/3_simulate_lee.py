"""3-D simulation of a 2-inch quad with Lee geometric control and B-spline gate trajectory.

Visualises (or headlessly benchmarks) a tuned Lee controller on one of four
gate circuits. Optionally loads a JSON config produced by example 5 (tuning).

Controller gains: attitude and rate gains are auto-scaled from the drone's inertia
(12 rad/s closed-loop bandwidth) to ensure stability at the 5 ms timestep.
Position and velocity gains default to the curriculum-tuner starting values and can
be overridden with --pos-gain / --vel-gain.

Run:
    # Headless test with default B-spline parameters:
    python examples/d_drones/3_simulate_lee.py --gates figure8 --no-viz

    # Load a tuned config and save animation:
    python examples/d_drones/3_simulate_lee.py \\
        --gates figure8 \\
        --bspline-config __data__/lee_tuning/stage3_best.json --save

    # Override position / velocity gains:
    python examples/d_drones/3_simulate_lee.py --gates circle \\
        --pos-gain 12.0 --vel-gain 8.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add the d_drones directory to sys.path so _ctrl_helpers is importable
sys.path.insert(0, str(Path(__file__).parent))
from _ctrl_helpers import ARM_LENGTH, PROP_SIZE, GateChecker, create_2inch_quad

from ariel.simulation.drone.controllers.lee_control.lee_controller import LeeGeometricControl
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
import ariel.simulation.drone.controllers.utils as ctrl_utils
from ariel.simulation.drone.controllers.utils.wind_model import Wind

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Lee geometric controller B-spline simulation for 2-inch quad",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--gates", required=True,
                    choices=["figure8", "circle", "slalom", "backandforth"],
                    help="Gate configuration")
parser.add_argument("--bspline-config",
                    help="Path to B-spline JSON config from tuning (stage3_best.json)")
parser.add_argument("--time", type=float, default=20.0,
                    help="Simulation duration in seconds (default 20)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Simulation time-step in seconds (default 0.005)")
parser.add_argument("--no-viz", action="store_true",
                    help="Disable 3-D animation (headless)")
parser.add_argument("--save", action="store_true",
                    help="Save animation to file")
# Position/velocity gains — defaults from the curriculum tuner starting values.
# Attitude and rate gains are auto-scaled from drone inertia (always stable at dt=5ms).
parser.add_argument("--pos-gain", type=float, default=14.3,
                    help="Lee position P gain (default 14.3)")
parser.add_argument("--vel-gain", type=float, default=9.0,
                    help="Lee velocity P gain (default 9.0)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

gate_config = GATE_CONFIGS[args.gates]
Tf: float = args.time
Ts: float = args.dt
Ti: float = 0.0

print("\n" + "=" * 70)
print("LEE CONTROLLER B-SPLINE SIMULATION — 2-INCH QUAD")
print("=" * 70)
print(f"Gate configuration : {args.gates}")
print(f"Drone              : 2-inch quad (arm={ARM_LENGTH}m, prop={PROP_SIZE}\")")
print(f"Simulation time    : {Tf}s at dt={Ts}s")
if args.bspline_config:
    print(f"Loading config     : {args.bspline_config}")
else:
    print("B-spline           : default parameters")
print(f"Lee gains          : pos={args.pos_gain}  vel={args.vel_gain}  "
      f"att/rate=auto-scaled from inertia")
print("=" * 70 + "\n")

# ---------------------------------------------------------------------------
# Drone, controller, trajectory
# ---------------------------------------------------------------------------

quad = create_2inch_quad()
wind = Wind("None")

ctrl = LeeGeometricControl(
    quad,
    yawType=1,
    orient="NED",
    auto_scale_gains=True,          # att/rate auto-derived from inertia (stable at dt=5ms)
    pos_P_gain=np.array([args.pos_gain] * 3),
    vel_P_gain=np.array([args.vel_gain] * 3),
    # att_P_gain and rate_P_gain intentionally omitted → auto-scaled
)

bspline_params = None
if args.bspline_config:
    with open(args.bspline_config) as f:
        cfg = json.load(f)
    if "bspline_params" in cfg:
        bspline_params = np.array(cfg["bspline_params"])
        print(f"Loaded {len(bspline_params)} B-spline parameters")

traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]), gate_config=gate_config)

if bspline_params is not None:
    traj.bspline_trajectory.set_parameters(bspline_params)
    print("B-spline trajectory loaded from config")
else:
    print("Using default B-spline trajectory")

# Set initial drone state to match trajectory at t=0
start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
_, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
if np.linalg.norm(vel_050[:2]) > 0.001:
    initial_yaw = np.arctan2(vel_050[1], vel_050[0])
else:
    initial_yaw = gate_config.gate_yaw[0]

initial_euler = np.array([0.0, 0.0, initial_yaw])
quad.drone_sim.set_state(
    position=start_pos,
    velocity=np.zeros(3),
    attitude=initial_euler,
    angular_velocity=np.zeros(3),
)
quad._update_state_variables()

print(f"Drone start position : [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}]")
print(f"Initial yaw          : {np.degrees(initial_yaw):.1f}°")

gate_checker = GateChecker(gate_config.gate_pos, gate_config.gate_yaw, gate_config.gate_size)
print(f"Gate detection       : {gate_checker.num_gates} gates\n")

# Seed first command
sDes = traj.desiredState(Ti, Ts, quad)
ctrl.controller(sDes, quad, traj.ctrlType, Ts)

# ---------------------------------------------------------------------------
# Allocate result buffers
# ---------------------------------------------------------------------------

num_steps = int(Tf / Ts + 1)
t_all = np.zeros(num_steps)
s_all = np.zeros((num_steps, len(quad.state)))
pos_all = np.zeros((num_steps, 3))
vel_all = np.zeros((num_steps, 3))
quat_all = np.zeros((num_steps, 4))
omega_all = np.zeros((num_steps, 3))
euler_all = np.zeros((num_steps, 3))
sDes_traj_all = np.zeros((num_steps, len(traj.sDes)))
sDes_calc_all = np.zeros((num_steps, len(ctrl.sDesCalc)))
w_cmd_all = np.zeros((num_steps, len(ctrl.w_cmd)))
wMotor_all = np.zeros((num_steps, 4))
thr_all = np.zeros((num_steps, 4))
tor_all = np.zeros((num_steps, 4))

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

print(f"Running simulation for {Tf}s …")
t0 = time.time()
t = Ti
i = 1

while round(t, 3) < Tf:
    t_all[i] = t
    s_all[i] = quad.state
    pos_all[i] = quad.pos
    vel_all[i] = quad.vel
    quat_all[i] = quad.quat
    omega_all[i] = quad.omega
    euler_all[i] = quad.euler
    sDes_traj_all[i] = traj.sDes
    sDes_calc_all[i] = ctrl.sDesCalc
    w_cmd_all[i] = ctrl.w_cmd
    wMotor_all[i] = quad.wMotor
    thr_all[i] = quad.thr
    tor_all[i] = quad.tor

    # Advance dynamics
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t_new = Ts * i

    # Update trajectory and controller for next step
    sDes = traj.desiredState(t_new, Ts, quad)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)

    # Gate detection
    passed = gate_checker.check_gate_passing(quad.pos)
    if passed:
        print(f"  *** GATE {gate_checker.gates_passed} passed at t={t:.3f}s "
              f"pos=[{quad.pos[0]:.2f}, {quad.pos[1]:.2f}, {quad.pos[2]:.2f}]")

    t = t_new
    i += 1

elapsed = time.time() - t0
print(f"Simulated {t:.2f}s in {elapsed:.3f}s wall-clock")

# Gate summary
print(f"\n{'=' * 70}")
print(f"GATE SUMMARY: {gate_checker.gates_passed} / {gate_checker.num_gates} passed")
print(f"{'=' * 70}\n")

# ---------------------------------------------------------------------------
# Visualisation / animation
# ---------------------------------------------------------------------------

if not args.no_viz:
    import matplotlib.pyplot as plt

    waypoints = np.array(gate_config.gate_pos, dtype=float)
    save_path = str(Path.cwd() / "__data__" / "lee_simulation.mp4") if args.save else None

    # Keep a reference to the FuncAnimation — matplotlib will garbage-collect
    # it otherwise and the figure renders empty.
    anim = ctrl_utils.sameAxisAnimation(
        t_all, waypoints, pos_all, quat_all, sDes_traj_all, Ts,
        quad.params, 15, 3, int(args.save), "NED",
        gate_pos=np.array(gate_config.gate_pos),
        gate_yaw=np.array(gate_config.gate_yaw),
        gate_size=gate_config.gate_size,
        save_path=save_path,
    )

    if save_path:
        print(f"Animation saved to: {save_path}")
    else:
        plt.show()
