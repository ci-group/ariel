"""Instantiate a drone from a DroneBlueprint and record a flight video.

This is the natural sequel to ``11_blueprint_demo.py``. The previous demo
proved two different encodings produce identical phenotype properties; here
we close the loop by **flying** a blueprint-derived drone with the Lee
geometric controller through a gate course, and saving the rendered
animation to MP4.

Pipeline exercised end-to-end:

    spherical_angular genome
        → spherical_angular_to_blueprint     (decoder)
        → DroneBlueprint  (saved as JSON for inspection)
        → blueprint_to_propellers(convention="ned")  (backend)
        → DroneInterface / DroneSimulator   (existing ARIEL physics)
        → Lee geometric controller          (existing ARIEL control)
        → sameAxisAnimation -> MP4          (existing ARIEL rendering)

Nothing downstream of the blueprint had to be modified — that's the point.

Run:
    uv run examples/d_drones/12_visualize_from_blueprint.py
    uv run examples/d_drones/12_visualize_from_blueprint.py --gates figure8 --time 12
    uv run examples/d_drones/12_visualize_from_blueprint.py --no-save  # show interactively
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Blueprint layer (new in this branch)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers

# Existing simulation / control stack (unchanged)
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.drone.controllers.lee_control.lee_controller import (
    LeeGeometricControl,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import (
    Trajectory,
)
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone.controllers.utils.wind_model import Wind
import ariel.simulation.drone.controllers.utils as ctrl_utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fly a drone instantiated from a DroneBlueprint",
)
parser.add_argument("--gates", default="circle",
                    choices=["figure8", "circle", "slalom", "backandforth"],
                    help="Gate configuration (default: circle)")
parser.add_argument("--time", type=float, default=10.0,
                    help="Simulation duration in seconds (default 10)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Simulation time-step in seconds (default 0.005)")
parser.add_argument("--arm-len", type=float, default=0.06,
                    help="Arm length in metres for the demo quad (default 0.06)")
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--pos-gain", type=float, default=14.3)
parser.add_argument("--vel-gain", type=float, default=9.0)
parser.add_argument("--no-save", action="store_true",
                    help="Show interactive window instead of writing an MP4")
parser.add_argument("--out",
                    default="__data__/blueprint_demo/blueprint_flight.mp4",
                    help="Output video path")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Build a spherical-angular genome for a 4-arm X-quad
# ---------------------------------------------------------------------------

mag = float(args.arm_len)
genome = np.array([
    # [magnitude, arm_az, arm_pitch, motor_az, motor_pitch, direction]
    [mag,  np.pi / 4,        0.0, 0.0, 0.0, 0],   # front-right, CCW
    [mag,  3 * np.pi / 4,    0.0, 0.0, 0.0, 1],   # front-left,  CW
    [mag, -3 * np.pi / 4,    0.0, 0.0, 0.0, 0],   # rear-left,   CCW
    [mag, -np.pi / 4,        0.0, 0.0, 0.0, 1],   # rear-right,  CW
])


# ---------------------------------------------------------------------------
# 2. Decode → DroneBlueprint, persist for inspection
# ---------------------------------------------------------------------------

bp = spherical_angular_to_blueprint(genome, propsize=args.prop_size)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
bp_path = out_path.with_suffix(".json")
bp.save_json(bp_path)

print("=" * 70)
print(f"DroneBlueprint built from genome ({genome.shape[0]} arms)")
print(f"  saved to: {bp_path}")
print("=" * 70)
print(bp.summary())


# ---------------------------------------------------------------------------
# 3. Backend: Blueprint → propellers list (NED for the Lee controller stack)
# ---------------------------------------------------------------------------

propellers = blueprint_to_propellers(bp, convention="ned")
print("\nPropellers handed to DroneSimulator (NED convention):")
for i, p in enumerate(propellers):
    print(f"  motor {i}: loc={[round(v, 3) for v in p['loc']]}  "
          f"dir={[round(v, 3) if isinstance(v, float) else v for v in p['dir']]}  "
          f"propsize={p['propsize']}")


# ---------------------------------------------------------------------------
# 4. Instantiate DroneInterface + Lee controller + trajectory
# ---------------------------------------------------------------------------

quad = DroneInterface(0, propellers=propellers)
wind = Wind("None")

ctrl = LeeGeometricControl(
    quad,
    yawType=1,
    orient="NED",
    auto_scale_gains=True,
    pos_P_gain=np.array([args.pos_gain] * 3),
    vel_P_gain=np.array([args.vel_gain] * 3),
)

gate_config = GATE_CONFIGS[args.gates]
traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]), gate_config=gate_config)

# Seed initial state to trajectory start (matches 3_simulate_lee.py)
start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
_, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
if np.linalg.norm(vel_050[:2]) > 0.001:
    initial_yaw = float(np.arctan2(vel_050[1], vel_050[0]))
else:
    initial_yaw = float(gate_config.gate_yaw[0])

quad.drone_sim.set_state(
    position=start_pos,
    velocity=np.zeros(3),
    attitude=np.array([0.0, 0.0, initial_yaw]),
    angular_velocity=np.zeros(3),
)
quad._update_state_variables()

sDes = traj.desiredState(0.0, args.dt, quad)
ctrl.controller(sDes, quad, traj.ctrlType, args.dt)


# ---------------------------------------------------------------------------
# 5. Roll the sim forward, log per-step state
# ---------------------------------------------------------------------------

Ts = float(args.dt)
Tf = float(args.time)
num_steps = int(Tf / Ts + 1)

t_all = np.zeros(num_steps)
pos_all = np.zeros((num_steps, 3))
quat_all = np.zeros((num_steps, 4))
sDes_traj_all = np.zeros((num_steps, len(traj.sDes)))

print(f"\nRunning {Tf}s simulation @ dt={Ts}s …")
t0 = time.time()
t = 0.0
i = 1
while round(t, 4) < Tf and i < num_steps:
    t_all[i] = t
    pos_all[i] = quad.pos
    quat_all[i] = quad.quat
    sDes_traj_all[i] = traj.sDes

    quad.update(t, Ts, ctrl.w_cmd, wind)
    t = Ts * i
    sDes = traj.desiredState(t, Ts, quad)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)
    i += 1
print(f"  simulated {t:.2f}s in {time.time() - t0:.2f}s wall-clock")


# ---------------------------------------------------------------------------
# 6. Render → MP4 (or interactive)
# ---------------------------------------------------------------------------

waypoints = np.array(gate_config.gate_pos, dtype=float)
save = not args.no_save
save_path = str(out_path) if save else None

ctrl_utils.sameAxisAnimation(
    t_all[:i], waypoints, pos_all[:i], quat_all[:i], sDes_traj_all[:i], Ts,
    quad.params, 15, 3, int(save), "NED",
    gate_pos=np.array(gate_config.gate_pos),
    gate_yaw=np.array(gate_config.gate_yaw),
    gate_size=gate_config.gate_size,
    save_path=save_path,
)

if save:
    print(f"\nVideo written to: {out_path}")
    print(f"Blueprint JSON  : {bp_path}")
