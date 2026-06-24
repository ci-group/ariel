"""Replay a trained gate-task PPO policy in MuJoCo for a given DroneBlueprint.

Generalises 17_hover_mujoco_viz.py to any gate-tracking task trained by
19_figure8.py / 20_slalom.py / 21_shuttlerun.py / 22_circle.py.

Run:
    uv run examples/spear/23_replay_task_mujoco.py \\
        --blueprint path/to/blueprint.json \\
        --policy    path/to/<task>_policy.zip \\
        --task      figure8 \\
        --vecnormalize path/to/vecnormalize.pkl \\
        --view

    # Render MP4 instead of opening the viewer (drop --view):
    uv run examples/spear/23_replay_task_mujoco.py \\
        --blueprint bp.json --policy figure8_policy.zip --task figure8 \\
        --vecnormalize vecnormalize.pkl --out __data__/figure8_viz.mp4

Tasks: hover | figure8 | slalom | shuttlerun | circle

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
import argparse
import math
import time as _time
from pathlib import Path

import mujoco
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec, blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv
from ariel.utils.video_recorder import VideoRecorder

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

TASK_TO_GATE_KEY = {
    "figure8":    "figure8",
    "slalom":     "slalom",
    "shuttlerun": "backandforth",
    "circle":     "circle",
}

parser = argparse.ArgumentParser(description="Replay a gate-task PPO policy in MuJoCo")
parser.add_argument("--blueprint", required=True, help="Path to DroneBlueprint JSON")
parser.add_argument("--policy",    required=True, help="Path to SB3 PPO ZIP")
parser.add_argument("--task",      required=True, choices=list(TASK_TO_GATE_KEY.keys()),
                    help="Task name (must match what the policy was trained on)")
parser.add_argument("--vecnormalize", default=None,
                    help="Optional path to the matching VecNormalize .pkl "
                         "(strongly recommended — obs-norm stats are needed for "
                         "the policy to behave the same as during training)")
parser.add_argument("--rollout-time", type=float, default=20.0,
                    help="Rollout duration in seconds (default 20)")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--device", default="cpu")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out",
                    default=None,
                    help="Output MP4 path (default: __data__/<task>_mujoco/<task>_viz.mp4)")
parser.add_argument("--view", action="store_true",
                    help="Open interactive MuJoCo viewer instead of writing MP4")
args = parser.parse_args()

if args.out is None:
    args.out = f"__data__/{args.task}_mujoco/{args.task}_viz.mp4"

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ─────────────────────────────────────────────────────────────────────────────
# Load blueprint
# ─────────────────────────────────────────────────────────────────────────────

bp = DroneBlueprint.load_json(Path(args.blueprint))
print(bp.summary())
propellers = blueprint_to_propellers(bp, convention="ned")

# ─────────────────────────────────────────────────────────────────────────────
# Build single-env gate task (mirrors training-time config)
# ─────────────────────────────────────────────────────────────────────────────

cfg = GATE_CONFIGS[TASK_TO_GATE_KEY[args.task]]
gpos = np.asarray(cfg.gate_pos,    dtype=np.float64)
gyaw = np.asarray(cfg.gate_yaw,    dtype=np.float64)
spos = np.asarray(cfg.starting_pos, dtype=np.float64)
xb   = tuple(np.asarray(cfg.x_bounds, dtype=np.float64).tolist())
yb   = tuple(np.asarray(cfg.y_bounds, dtype=np.float64).tolist())
zb   = tuple(np.asarray(cfg.z_bounds, dtype=np.float64).tolist())
print(f"Task: {args.task}  |  gates: {len(gpos)}  |  start (NED): {spos.tolist()}")

N = int(args.rollout_time / args.dt) + 1

raw_env = TorchDroneGateEnv(
    num_envs=1,
    propellers=propellers,
    gates_pos=gpos,
    gate_yaw=gyaw,
    start_pos=spos,
    x_bounds=xb,
    y_bounds=yb,
    z_bounds=zb,
    gates_ahead=2,
    device=args.device,
    dt=args.dt,
    seed=args.seed,
    max_steps=N + 1,
    initialize_at_random_gates=False,
    # Reward shaping is irrelevant for replay; leave defaults.
)

# Wrap with VecNormalize to apply the trained obs-normalization stats.
if args.vecnormalize is not None:
    vn_path = Path(args.vecnormalize)
    if not vn_path.exists():
        raise FileNotFoundError(f"VecNormalize file not found: {vn_path}")
    env = VecNormalize.load(str(vn_path), raw_env)
    env.training = False
    env.norm_reward = False
    print(f"VecNormalize loaded: {vn_path}")
else:
    env = raw_env
    print("WARNING: no --vecnormalize given — policy will see raw (unnormalized) obs. "
          "If training used VecNormalize(norm_obs=True), behavior will be wrong.")

model = PPO.load(args.policy, env=None, device=args.device)
policy_obs_dim = model.observation_space.shape[0]
env_obs_dim    = env.observation_space.shape[0]
if policy_obs_dim != env_obs_dim:
    raise ValueError(
        f"Policy obs dim ({policy_obs_dim}) != env obs dim ({env_obs_dim}). "
        "Blueprint or task probably doesn't match the trained policy."
    )
print(f"Policy loaded: {args.policy}")

# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

pos_ned   = np.zeros((N, 3), dtype=np.float32)
euler_log = np.zeros((N, 3), dtype=np.float32)
rates_log = np.zeros((N, 3), dtype=np.float32)
action_log = np.zeros((N, raw_env.num_motors), dtype=np.float32)
target_gate_log = np.zeros(N, dtype=np.int64)

obs = env.reset()
gates_passed_total = 0
for i in range(N):
    pos_ned[i]    = raw_env.world_states[0, 0:3].cpu().numpy()
    euler_log[i]  = raw_env.world_states[0, 6:9].cpu().numpy()
    rates_log[i]  = raw_env.world_states[0, 9:12].cpu().numpy()
    target_gate_log[i] = int(raw_env.target_gates[0].item())
    action, _ = model.predict(obs, deterministic=True)
    action_log[i] = action[0]
    obs, _, _, _ = env.step(action)
    gates_passed_total = int(raw_env.num_gates_passed[0].item())

deg = np.degrees
print(
    f"\nRollout done: {N} steps ({N * args.dt:.1f}s)\n"
    f"  Gates passed: {gates_passed_total} (final target idx={int(target_gate_log[-1])})\n"
    f"  XYZ-NED range:  x=[{pos_ned[:,0].min():.2f},{pos_ned[:,0].max():.2f}]  "
    f"y=[{pos_ned[:,1].min():.2f},{pos_ned[:,1].max():.2f}]  "
    f"z=[{pos_ned[:,2].min():.2f},{pos_ned[:,2].max():.2f}]\n"
    f"  Roll  (deg): [{deg(euler_log[:,0]).min():.1f}, {deg(euler_log[:,0]).max():.1f}]\n"
    f"  Pitch (deg): [{deg(euler_log[:,1]).min():.1f}, {deg(euler_log[:,1]).max():.1f}]\n"
    f"  Yaw   (deg): [{deg(euler_log[:,2]).min():.1f}, {deg(euler_log[:,2]).max():.1f}]\n"
    f"  Action saturation (|a|>0.9): {(np.abs(action_log)>0.9).mean()*100:.1f}%"
)

# ─────────────────────────────────────────────────────────────────────────────
# NED → ENU (only Z is flipped here, matching 17_hover_mujoco_viz / 14)
# ─────────────────────────────────────────────────────────────────────────────

pos_enu = pos_ned.copy()
pos_enu[:, 2] = -pos_enu[:, 2]

phi, theta, psi = euler_log[:, 0], euler_log[:, 1], euler_log[:, 2]
cy, sy = np.cos(psi / 2), np.sin(psi / 2)
cp, sp = np.cos(theta / 2), np.sin(theta / 2)
cr, sr = np.cos(phi / 2), np.sin(phi / 2)
quat_enu = np.stack([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    -(cr * sp * cy + sr * cp * sy),
    cr * cp * sy - sr * sp * cy,
], axis=-1)

# Gate positions/yaws in ENU
gates_enu = gpos.copy()
gates_enu[:, 2] = -gates_enu[:, 2]
gates_yaw_enu = -gyaw  # yaw flips sign when Z flips


def _yaw_quat(yaw: float):
    """Quaternion (w, x, y, z) for rotation about ENU z by `yaw` radians."""
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


# ─────────────────────────────────────────────────────────────────────────────
# Build MuJoCo scene: drone + gate markers
# ─────────────────────────────────────────────────────────────────────────────

arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06

drone_spec = blueprint_to_mjspec(
    bp,
    motor_mass=0.01,
    arm_mass=0.034 * mean_arm_len,
    body_name=f"{args.task}_drone",
)
world_mj = SimpleFlatWorld()
world_mj.spawn(
    drone_spec,
    position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
    correct_collision_with_floor=False,
)

# Gate markers: a thin vertical "panel" (box) centred on the gate, oriented
# so its long axis lies in the gate plane (perpendicular to gate yaw),
# plus a small sphere at the gate centre for clarity.
GATE_HALF_WIDTH  = 0.6   # m, half-extent across the gate opening
GATE_HALF_HEIGHT = 0.4   # m, half-extent vertically
GATE_HALF_THICK  = 0.02  # m, thickness of the panel
SPHERE_R         = 0.08

for gi, (gp, gy) in enumerate(zip(gates_enu, gates_yaw_enu)):
    body = world_mj.spec.worldbody.add_body(
        name=f"gate_{gi}",
        pos=[float(gp[0]), float(gp[1]), float(gp[2])],
        quat=_yaw_quat(float(gy)),
    )
    # Highlight current/first target gate in green; others in yellow.
    rgba_panel  = (0.95, 0.85, 0.15, 0.45)
    rgba_marker = (0.1, 0.9, 0.1, 0.9) if gi == 0 else (0.95, 0.85, 0.15, 0.9)
    body.add_geom(
        name=f"gate_panel_{gi}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        # In gate-local frame: x = travel direction, y = lateral, z = up.
        size=[GATE_HALF_THICK, GATE_HALF_WIDTH, GATE_HALF_HEIGHT],
        rgba=rgba_panel,
        contype=0, conaffinity=0,
    )
    body.add_geom(
        name=f"gate_marker_{gi}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[SPHERE_R, 0.0, 0.0],
        rgba=rgba_marker,
        contype=0, conaffinity=0,
    )

model_mj = world_mj.spec.compile()
data_mj  = mujoco.MjData(model_mj)
model_mj.opt.timestep = args.dt


def _set_pose(idx: int) -> None:
    p, q = pos_enu[idx], quat_enu[idx]
    data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
    data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
    data_mj.qvel[:] = 0.0
    mujoco.mj_forward(model_mj, data_mj)


# ─────────────────────────────────────────────────────────────────────────────
# View or render
# ─────────────────────────────────────────────────────────────────────────────

if args.view:
    import mujoco.viewer
    print("Launching MuJoCo passive viewer …")
    _set_pose(0)
    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        idx = 0
        while viewer.is_running() and idx < len(pos_enu):
            t0 = _time.time()
            _set_pose(idx)
            viewer.sync()
            slack = args.dt - (_time.time() - t0)
            if slack > 0:
                _time.sleep(slack)
            idx += 1
else:
    mp4 = Path(args.out)
    mp4.parent.mkdir(parents=True, exist_ok=True)
    recorder = VideoRecorder(
        file_name=mp4.stem, output_folder=mp4.parent,
        width=720, height=540, fps=30,
    )
    steps_per_frame = max(1, int(round(1.0 / (recorder.fps * args.dt))))
    t_render = _time.time()
    with mujoco.Renderer(model_mj, width=recorder.width, height=recorder.height) as renderer:
        for idx in range(0, len(pos_enu), steps_per_frame):
            _set_pose(idx)
            renderer.update_scene(data_mj)
            recorder.write(frame=renderer.render())
    recorder.release()
    print(f"Rendered in {_time.time() - t_render:.2f}s")
    print(f"Video: {mp4}")

print("Done.")
