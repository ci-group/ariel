"""Load a DroneBlueprint from JSON, spawn it in MuJoCo with gravity off.

No controller — ctrl is all zeros.  The drone just floats at its spawn
position so you can inspect the geometry.

Run:
    uv run examples/spear/16_blueprint_mujoco_no_gravity.py --blueprint <path.json>
    uv run examples/spear/16_blueprint_mujoco_no_gravity.py --blueprint __data__/blueprint_demo/blueprint_flight.json
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
from ariel.simulation.environments import SimpleFlatWorld

parser = argparse.ArgumentParser(
    description="Spawn a blueprint drone in MuJoCo with gravity disabled"
)
parser.add_argument("--blueprint", required=True,
                    help="Path to a DroneBlueprint JSON file")
parser.add_argument("--height", type=float, default=1.5,
                    help="Spawn height in metres (default: 1.5)")
parser.add_argument("--time", type=float, default=30.0,
                    help="Viewer duration in seconds (default: 30)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1. Load blueprint
# ---------------------------------------------------------------------------

bp_path = Path(args.blueprint)
if not bp_path.exists():
    raise FileNotFoundError(f"Blueprint not found: {bp_path}")

bp = DroneBlueprint.load_json(bp_path)
print(f"Loaded blueprint from {bp_path}")
print(bp.summary())

# ---------------------------------------------------------------------------
# 2. Blueprint → MuJoCo spec → world
# ---------------------------------------------------------------------------

drone_spec = blueprint_to_mjspec(bp, body_name="drone")

world = SimpleFlatWorld()
world.spawn(
    drone_spec,
    position=(0.0, 0.0, float(args.height)),
    correct_collision_with_floor=False,
)

model = world.spec.compile()
data = mujoco.MjData(model)

# ---------------------------------------------------------------------------
# 3. Turn off gravity
# ---------------------------------------------------------------------------

model.opt.gravity[:] = 0.0

# ---------------------------------------------------------------------------
# 4. Launch viewer — no actuator commands, drone floats in place
# ---------------------------------------------------------------------------

print(f"\nLaunching MuJoCo viewer (gravity=0, no controls) for {args.time}s …")
print("  Close the window or wait for the timeout to exit.\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    deadline = time.time() + args.time
    while viewer.is_running() and time.time() < deadline:
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        slack = model.opt.timestep - (time.time() - step_start)
        if slack > 0:
            time.sleep(slack)

print("Done.")
