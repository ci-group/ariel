"""Lee figure-8 flight of a blueprint drone, visualised in MuJoCo.

Pipeline:

    spherical_angular genome
      → spherical_angular_to_blueprint        (decoder)
      → DroneBlueprint                        (saved as JSON)
      → blueprint_to_propellers (NED)
            │
            ▼
        DroneInterface + LeeGeometricControl(orient="NED")
          + Trajectory(figure8 gate config, B-spline)
            │
            ▼
        record (t, pos_NED, quat_NED) over the full flight
            │
            ▼
        convert NED → ENU (Z-flip)
            │
      → blueprint_to_mjspec
            │
            ▼
        MuJoCo: kinematic playback of the recorded flight
        (mj_forward each frame; no physics integration since the
         drone's pose is being driven from the recorded trajectory)
            │
            ▼
        MP4 / passive viewer

Why kinematic playback instead of closed-loop in MuJoCo?
- The validated Lee path in this repo is NED; the ENU branch in
  `_wrench_to_motor_commands` has a sign error in the thrust
  allocation that drives w_cmd to the motor floor. Until that's
  patched upstream, the cleanest demo is to run the validated NED
  controller in the Python simulator and visualise the result in
  MuJoCo. The blueprint still drives both backends end-to-end —
  ``blueprint_to_propellers`` feeds the Python sim, and
  ``blueprint_to_mjspec`` produces the visual model.

Run:
    uv run examples/d_drones/14_mujoco_lee_figure8.py
    uv run examples/d_drones/14_mujoco_lee_figure8.py --view --time 12
    uv run examples/d_drones/14_mujoco_lee_figure8.py --altitude 2 --time 20
"""
from __future__ import annotations

import argparse
import copy
import os
import time as _time
from pathlib import Path

import numpy as np
import mujoco

os.environ.setdefault("OMP_NUM_THREADS", "1")

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.backends import (
    blueprint_to_mjspec, blueprint_to_propellers,
)
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.drone.drone_interface import DroneInterface
from ariel.simulation.drone.controllers.lee_control.lee_controller import (
    LeeGeometricControl,
)
from ariel.simulation.drone.controllers.trajectory_generation.trajectory import (
    Trajectory,
)
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone.controllers.utils.wind_model import Wind
from ariel.utils.video_recorder import VideoRecorder


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Fly a blueprint-derived drone through a figure-8 in MuJoCo"
)
parser.add_argument("--arm-len", type=float, default=0.06,
                    help="Arm length in metres (default 0.06)")
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--altitude", type=float, default=1.5,
                    help="Figure-8 flight altitude in metres (default 1.5)")
parser.add_argument("--time", type=float, default=15.0,
                    help="Simulation duration in seconds (default 15)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Controller dt (default 0.005, matches example 3)")
parser.add_argument("--pos-gain", type=float, default=14.3)
parser.add_argument("--vel-gain", type=float, default=9.0)
parser.add_argument("--view", action="store_true",
                    help="Launch passive viewer instead of writing video")
parser.add_argument("--out",
                    default="__data__/blueprint_demo/mujoco_figure8.mp4")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Genome → DroneBlueprint
# ---------------------------------------------------------------------------

mag = float(args.arm_len)
genome = np.array([
    [mag,  np.pi / 4,        0.0, 0.0, 0.0, 0],
    [mag,  3 * np.pi / 4,    0.0, 0.0, 0.0, 1],
    [mag, -3 * np.pi / 4,    0.0, 0.0, 0.0, 0],
    [mag, -np.pi / 4,        0.0, 0.0, 0.0, 1],
])
bp = spherical_angular_to_blueprint(genome, propsize=args.prop_size)

out_path = Path(args.out)
out_path.parent.mkdir(parents=True, exist_ok=True)
bp.save_json(out_path.with_suffix(".json"))


# ---------------------------------------------------------------------------
# 2. Blueprint → DroneInterface (NED) + Lee + figure-8 trajectory
# ---------------------------------------------------------------------------

propellers = blueprint_to_propellers(bp, convention="ned")
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

# Figure-8 gates at the chosen altitude. In NED, "up" is negative Z, so we
# flip the sign of the altitude when placing the gates.
gate_config = copy.deepcopy(GATE_CONFIGS["figure8"])
gate_config.gate_pos = gate_config.gate_pos.copy()
gate_config.gate_pos[:, 2] = -float(args.altitude)
gate_config.starting_pos = gate_config.starting_pos.copy()
gate_config.starting_pos[2] = -float(args.altitude)

traj = Trajectory(quad, "xyz_pos", np.array([15, 3, 1]),
                  gate_config=gate_config)


# ---------------------------------------------------------------------------
# 3. Seed initial state at the trajectory start
# ---------------------------------------------------------------------------

start_pos, _, _ = traj.bspline_trajectory.evaluate(0.0)
_, vel_050, _ = traj.bspline_trajectory.evaluate(0.05)
initial_yaw = float(np.arctan2(vel_050[1], vel_050[0])) \
    if np.linalg.norm(vel_050[:2]) > 1e-3 else float(gate_config.gate_yaw[0])

quad.drone_sim.set_state(
    position=np.array(start_pos),
    velocity=np.zeros(3),
    attitude=np.array([0.0, 0.0, initial_yaw]),
    angular_velocity=np.zeros(3),
)
quad._update_state_variables()
sDes = traj.desiredState(0.0, args.dt, quad)
ctrl.controller(sDes, quad, traj.ctrlType, args.dt)


# ---------------------------------------------------------------------------
# 4. Run the Python simulation in NED — record (pos, quat) per step
# ---------------------------------------------------------------------------

Ts = float(args.dt)
Tf = float(args.time)
num_steps = int(Tf / Ts) + 1

t_log = np.zeros(num_steps)
pos_log = np.zeros((num_steps, 3))
quat_log = np.zeros((num_steps, 4))  # quad.quat is (w, x, y, z) here

print(f"\nRunning Python NED sim for {Tf}s @ dt={Ts}s …")
t0 = _time.time()
i = 0
t = 0.0
while i < num_steps and t < Tf:
    t_log[i] = t
    pos_log[i] = quad.pos
    quat_log[i] = quad.quat

    quad.update(t, Ts, ctrl.w_cmd, wind)
    t = Ts * (i + 1)
    sDes = traj.desiredState(t, Ts, quad)
    ctrl.controller(sDes, quad, traj.ctrlType, Ts)
    i += 1

t_log = t_log[:i]; pos_log = pos_log[:i]; quat_log = quat_log[:i]
print(f"  recorded {i} steps in {_time.time() - t0:.2f}s wall-clock")
print(f"  altitude range (NED z): "
      f"min={pos_log[:, 2].min():.2f}  max={pos_log[:, 2].max():.2f}")


# ---------------------------------------------------------------------------
# 5. NED → ENU conversion for MuJoCo playback (Z-flip on pos; rotate attitude
#    180° about the body's X axis so body +Z points up in ENU instead of
#    down in NED).
# ---------------------------------------------------------------------------

pos_enu = pos_log.copy()
pos_enu[:, 2] = -pos_enu[:, 2]

# NED→ENU attitude: world basis change M = diag(1, 1, -1). The body's
# rotation matrix in ENU is M @ R_ned @ M (M is involutory). At the
# quaternion level this corresponds to negating the y component of the
# quaternion (q_x stays, q_y flips sign, q_z stays, q_w stays) — derived by
# direct matrix-to-quaternion algebra for this specific M.
quat_enu = quat_log.copy()
quat_enu[:, 2] = -quat_enu[:, 2]  # flip the y (= qy) component


# ---------------------------------------------------------------------------
# 6. Build the MuJoCo model from the blueprint and prepare playback
# ---------------------------------------------------------------------------

# Mass-match MuJoCo body to DroneInterface so the model "weighs" what Lee
# planned for (useful even in kinematic playback for any visual indicators
# that depend on inertia).
target_mass = float(quad.params["mB"])
n_arms = sum(1 for _ in bp.children(bp.root_id))  # type: ignore[arg-type]
motor_mass_each = float(quad.drone_sim.config.propellers[0]["mass"])
arm_mass_each = 0.034 * mag  # BEAM_DENSITY × length, matches DroneConfiguration
core_mass = max(1e-4, target_mass - n_arms * (motor_mass_each + arm_mass_each))

drone_spec = blueprint_to_mjspec(
    bp,
    motor_mass=motor_mass_each,
    arm_mass=arm_mass_each,
    core_mass_override=core_mass,
    body_name="quad",
)
world = SimpleFlatWorld()
world.spawn(
    drone_spec,
    position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
    correct_collision_with_floor=False,
)
model = world.spec.compile()
data = mujoco.MjData(model)
model.opt.timestep = Ts


def set_pose(idx: int) -> None:
    """Write the idx-th recorded (pos, quat) into MuJoCo's freejoint qpos."""
    p = pos_enu[idx]
    q = quat_enu[idx]
    data.qpos[0] = float(p[0])
    data.qpos[1] = float(p[1])
    data.qpos[2] = float(p[2])
    data.qpos[3] = float(q[0])   # w
    data.qpos[4] = float(q[1])   # x
    data.qpos[5] = float(q[2])   # y  (already sign-flipped)
    data.qpos[6] = float(q[3])   # z
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# 7. Playback: viewer or MP4
# ---------------------------------------------------------------------------

if args.view:
    import mujoco.viewer

    print("\nLaunching MuJoCo passive viewer …")
    set_pose(0)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        sim_start = _time.time()
        idx = 0
        while viewer.is_running() and idx < len(t_log):
            step_start = _time.time()
            set_pose(idx)
            viewer.sync()
            slack = Ts - (_time.time() - step_start)
            if slack > 0:
                _time.sleep(slack)
            idx += 1
        print(f"Viewer done at t={t_log[min(idx, len(t_log)-1)]:.2f}s "
              f"(wall {_time.time() - sim_start:.2f}s)")
else:
    recorder = VideoRecorder(
        file_name=out_path.stem,
        output_folder=out_path.parent,
        width=720,
        height=540,
        fps=30,
    )
    steps_per_frame = max(1, int(round(1.0 / (recorder.fps * Ts))))
    print(f"\nRendering {len(t_log)} sim steps "
          f"({steps_per_frame} steps/frame) → MP4 …")
    t0 = _time.time()
    with mujoco.Renderer(model, width=recorder.width,
                         height=recorder.height) as renderer:
        for idx in range(0, len(t_log), steps_per_frame):
            set_pose(idx)
            renderer.update_scene(data)
            recorder.write(frame=renderer.render())
    recorder.release()
    print(f"  rendered in {_time.time() - t0:.2f}s wall-clock")
    print(f"\nVideo: {out_path}")

print(f"Final position (ENU): {pos_enu[-1].tolist()}")
