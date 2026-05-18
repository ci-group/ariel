"""3D simulation of a drone with Lee Geometric Controller and B-spline gate trajectory.

Mirrors src/airevolve/examples/simulation/run_3D_simulation_lee_ctrl.py
using ARIEL imports.

Usage:
    # Visualise with figure-8 gates (interactive animation):
    uv run examples/d_drones/2_simulate_lee.py --gates figure8

    # Headless with circle gates for 20 s:
    uv run examples/d_drones/2_simulate_lee.py --gates circle --time 20 --no-viz

    # Load tuned controller from JSON produced by 3_tune_lee.py:
    uv run examples/d_drones/2_simulate_lee.py \\
        --gates figure8 --bspline-config __data__/tuning/tuning_results.json

Requires: matplotlib, numpy (bundled in the project's uv environment)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from ariel.simulation.drone import DroneSimulator, create_standard_propeller_config
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import Trajectory
from ariel.simulation.drone.controllers.trajectory_generation.bspline_gate_trajectory import (
    BSplineGateTrajectory,
)
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
import ariel.simulation.drone.controllers.utils as utils

# Lee controller lives in the controllers sub-package (unchanged from airevolve)
try:
    from airevolve.controllers.lee_control.lee_controller import LeeGeometricControl
    from airevolve.controllers.utils.wind_model import Wind
except ImportError:
    print("WARNING: airevolve.controllers.lee_control not available; "
          "Lee controller simulation will not run.")
    LeeGeometricControl = None
    Wind = None

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Lee Controller B-Spline Simulation")
parser.add_argument("--gates", choices=["figure8", "circle", "slalom", "backandforth"],
                    required=True, help="Gate configuration")
parser.add_argument("--bspline-config", default=None,
                    help="Path to tuning_results.json (optional)")
parser.add_argument("--time", type=float, default=20.0,
                    help="Simulation time in seconds (default 20)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Time step in seconds (default 0.005)")
parser.add_argument("--arm-length", type=float, default=0.07,
                    help="Arm length in metres (default 0.07)")
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--no-viz", action="store_true",
                    help="Disable matplotlib animation")
parser.add_argument("--save", action="store_true",
                    help="Save animation to mp4")
args = parser.parse_args()

if LeeGeometricControl is None:
    sys.exit("Lee controller not available. Aborting.")

# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

ARM_LENGTH = args.arm_length
PROP_SIZE = args.prop_size

propellers = create_standard_propeller_config("quad", arm_length=ARM_LENGTH, prop_size=PROP_SIZE)
quad = DroneInterface(0, propellers=propellers)

gate_config = GATE_CONFIGS[args.gates]

# ---------------------------------------------------------------------------
# B-spline trajectory
# ---------------------------------------------------------------------------

bspline_params = None
if args.bspline_config:
    with open(args.bspline_config, "r") as f:
        config = json.load(f)
    if "bspline_params" in config:
        bspline_params = np.array(config["bspline_params"])
        print(f"Loaded {len(bspline_params)} B-spline parameters from {args.bspline_config}")

traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]), gate_config=gate_config)
if bspline_params is not None:
    traj.bspline_trajectory.set_parameters(bspline_params)

# ---------------------------------------------------------------------------
# Initial drone state aligned with trajectory start
# ---------------------------------------------------------------------------

start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
_, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
initial_yaw = (np.arctan2(vel_050[1], vel_050[0])
               if np.linalg.norm(vel_050[:2]) > 0.001
               else gate_config.gate_yaw[0])
initial_euler = np.array([0.0, 0.0, initial_yaw])
quad.drone_sim.set_state(position=start_pos, velocity=np.zeros(3),
                         attitude=initial_euler, angular_velocity=np.zeros(3))
quad._update_state_variables()

print(f"Drone start: pos={np.round(start_pos, 3)}  yaw={np.degrees(initial_yaw):.1f}°")

# ---------------------------------------------------------------------------
# Lee controller
# ---------------------------------------------------------------------------

ctrl = LeeGeometricControl(quad, yawType=3, orient="NED", auto_scale_gains=True)
wind = Wind("None", 0, "NED")
Ts = args.dt
Tf = args.time
Ti = 0.0

sDes = traj.desiredState(0, Ts, quad)
ctrl.controller(sDes, quad, "xyz_pos", Ts)

# ---------------------------------------------------------------------------
# Simulation loop
# ---------------------------------------------------------------------------

numTimeStep = int(Tf / Ts + 1)
t_all   = np.zeros(numTimeStep)
pos_all = np.zeros((numTimeStep, 3))
quat_all = np.zeros((numTimeStep, 4))
sDes_traj_all = np.zeros((numTimeStep, len(traj.sDes)))

t = Ti
i = 1
start_time = time.time()

while round(t, 3) < Tf:
    t_all[i] = t
    pos_all[i] = quad.pos
    quat_all[i] = quad.quat
    sDes_traj_all[i] = traj.sDes

    t = quad.update(t, Ts, ctrl.w_cmd, wind)
    sDes = traj.desiredState(t, Ts, quad)
    ctrl.controller(sDes, quad, "xyz_pos", Ts)
    i += 1

elapsed = time.time() - start_time
print(f"Simulated {Tf}s in {elapsed:.3f}s ({Tf/elapsed:.0f}x real-time)")

# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

if not args.no_viz:
    params = {"dxm": ARM_LENGTH, "dym": ARM_LENGTH, "dzm": 0.05}
    utils.sameAxisAnimation(
        t_all, gate_config.gate_pos, pos_all, quat_all, sDes_traj_all,
        Ts, params, 15, 3, int(args.save), "NED",
        gate_pos=gate_config.gate_pos,
        gate_yaw=gate_config.gate_yaw,
        gate_size=getattr(gate_config, "gate_size", 1.0),
    )
