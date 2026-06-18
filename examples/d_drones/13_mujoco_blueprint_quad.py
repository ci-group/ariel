"""Simulate a blueprint-derived quadcopter in MuJoCo and record a video.

Pipeline:

    spherical_angular genome
      → spherical_angular_to_blueprint  (decoder)
      → DroneBlueprint                  (saved as JSON for inspection)
      → blueprint_to_mjspec             (NEW backend)
      → mujoco.MjSpec → MjModel         (compiled into ARIEL's SimpleFlatWorld)
      → constant per-motor thrust       (open-loop hover; no controller)
      → video_renderer → MP4

This is the MuJoCo counterpart to example 12 (which used the Python Lee
controller stack). Same blueprint, different backend — exactly the
portability story the ARIEL blueprint architecture promises.

Run:
    uv run examples/d_drones/13_mujoco_blueprint_quad.py
    uv run examples/d_drones/13_mujoco_blueprint_quad.py --duration 8 --thrust 0.55
    uv run examples/d_drones/13_mujoco_blueprint_quad.py --view    # interactive viewer, no MP4
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import mujoco

os.environ.setdefault("OMP_NUM_THREADS", "1")

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.video_recorder import VideoRecorder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="MuJoCo simulation of a blueprint-derived quadcopter"
)
parser.add_argument("--arm-len", type=float, default=0.12,
                    help="Arm length in metres (default 0.12)")
parser.add_argument("--prop-size", type=int, default=5,
                    help="Propeller size in inches (default 5)")
parser.add_argument("--max-thrust", type=float, default=5.0,
                    help="Per-motor max thrust in Newtons (default 5.0)")
parser.add_argument("--thrust", type=float, default=None,
                    help="Fixed per-motor ctrl ∈ [0,1]. If omitted, "
                         "auto-compute hover thrust = m*g / (n * max_thrust).")
parser.add_argument("--duration", type=float, default=6.0,
                    help="Sim duration in seconds (default 6)")
parser.add_argument("--spawn-z", type=float, default=0.5,
                    help="Initial spawn height in m (default 0.5)")
parser.add_argument("--out", default="__data__/blueprint_demo/mujoco_quad.mp4",
                    help="Output video path (ignored when --view is set)")
parser.add_argument("--view", action="store_true",
                    help="Launch MuJoCo passive viewer instead of writing a video")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Build a spherical-angular genome for a 4-arm X-quad
# ---------------------------------------------------------------------------

mag = float(args.arm_len)
genome = np.array([
    # [magnitude, arm_az, arm_pitch, motor_az, motor_pitch, direction]
    [mag,  np.pi / 4,        0.0, 0.0, 0.0, 0],
    [mag,  3 * np.pi / 4,    0.0, 0.0, 0.0, 1],
    [mag, -3 * np.pi / 4,    0.0, 0.0, 0.0, 0],
    [mag, -np.pi / 4,        0.0, 0.0, 0.0, 1],
])


# ---------------------------------------------------------------------------
# 2. Decode → blueprint → MuJoCo MjSpec
# ---------------------------------------------------------------------------

bp = spherical_angular_to_blueprint(genome, propsize=args.prop_size)

drone_spec = blueprint_to_mjspec(
    bp,
    max_thrust=args.max_thrust,
    body_name="quad",
)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
bp_path = out_path.with_suffix(".json")
bp.save_json(bp_path)
print("=" * 70)
print(f"Blueprint: {bp_path}")
print(bp.summary())


# ---------------------------------------------------------------------------
# 3. Spawn into a flat world and compile
# ---------------------------------------------------------------------------

world = SimpleFlatWorld()
world.spawn(
    drone_spec,
    position=(0.0, 0.0, float(args.spawn_z)),
    correct_collision_with_floor=False,  # we *want* to start in mid-air
)

model = world.spec.compile()
data = mujoco.MjData(model)

print(f"\nCompiled model: nbody={model.nbody}  nu={model.nu}  "
      f"total_mass={sum(model.body_mass):.3f} kg")
print(f"Actuators: {[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]}")


# ---------------------------------------------------------------------------
# 4. Decide on a control level
#    Hover thrust per motor = (total_mass * g) / (n_motors * max_thrust)
# ---------------------------------------------------------------------------

total_mass = float(np.sum(model.body_mass))
gravity_mag = abs(float(model.opt.gravity[2]))
n_motors = model.nu

if args.thrust is None:
    # Slightly above hover so the drone rises visibly during the video.
    hover_ctrl = (total_mass * gravity_mag) / (n_motors * args.max_thrust)
    hover_ctrl = float(np.clip(hover_ctrl * 1.08, 0.0, 1.0))
else:
    hover_ctrl = float(args.thrust)

# Symmetric ctrl — any per-motor imbalance tilts the drone and crashes it
# under open-loop control. For tilted/asymmetric flight, plug in a Lee or
# PID controller (see example 12).
ctrl = np.full(n_motors, hover_ctrl, dtype=np.float64)
data.ctrl[:] = ctrl

print(f"\nTotal mass:      {total_mass:.4f} kg")
print(f"Gravity:         {gravity_mag:.3f} m/s²")
print(f"Hover ctrl/motor (auto): {hover_ctrl:.4f}")
print(f"Final ctrl vector:        {data.ctrl.tolist()}")


# ---------------------------------------------------------------------------
# 5. Roll the sim while writing a video
#    A simple controller would loop here; for the demo we keep ctrl fixed
#    and let video_renderer drive the step loop.
# ---------------------------------------------------------------------------

# video_renderer resets data.time but not data.ctrl; we restore ctrl after
# each step by pre-seeding before render and re-applying inside the loop.
# The renderer in this repo doesn't take a step callback, so the simplest
# robust path is to inline a custom render loop:

mujoco.mj_resetData(model, data)
data.ctrl[:] = ctrl

if args.view:
    # Interactive MuJoCo passive viewer — runs in real time, no MP4 written.
    import time
    import mujoco.viewer

    print("\nLaunching MuJoCo passive viewer (close window or Ctrl+C to exit) …")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = time.time()
        while viewer.is_running() and data.time < args.duration:
            step_start = time.time()
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.sync()
            # Pace to wall-clock so the viewer runs at real time.
            slack = model.opt.timestep - (time.time() - step_start)
            if slack > 0:
                time.sleep(slack)
        print(f"Viewer closed at t={data.time:.2f}s "
              f"(wall {time.time() - sim_start:.2f}s)")
else:
    # Headless render → MP4.
    recorder = VideoRecorder(
        file_name=out_path.stem,
        output_folder=out_path.parent,
        width=640,
        height=480,
        fps=30,
    )
    steps_per_frame = max(
        1, int(round(1.0 / (recorder.fps * model.opt.timestep)))
    )
    with mujoco.Renderer(model,
                         width=recorder.width,
                         height=recorder.height) as renderer:
        while data.time < args.duration:
            for _ in range(steps_per_frame):
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
            renderer.update_scene(data)
            recorder.write(frame=renderer.render())
    recorder.release()
    print(f"\nVideo: {out_path}")

print(f"Final drone height: {data.qpos[2]:.3f} m")
