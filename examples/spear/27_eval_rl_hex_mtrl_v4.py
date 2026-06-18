"""Visualize the v4 MTRL policy sequentially on the four test tasks in MuJoCo.

Matches 27_train_rl_hex_mtrl_v4.py: figure8 at density 1 with a canonical
start; slalom / shuttle-run with random-gate init; tilt termination at
cos=0.10; velocity-toward-gate reward and soft altitude floor enabled.

Run:
    uv run examples/spear/27_eval_rl_hex_mtrl_v4.py \\
        --policy __data__/spear_rl_hex_mtrl_v4/RUN/policy.zip

NOTE: do NOT add `from __future__ import annotations` to this file.
"""

import argparse
import importlib.util
import math
import sys
import time as _time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_mjspec, blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv
from ariel.utils.video_recorder import VideoRecorder


# Import the v3 policy + helpers
_TRAIN_PATH = Path(__file__).with_name("27_train_rl_hex_mtrl_v4.py")
_spec = importlib.util.spec_from_file_location("mtrl_train_v4", _TRAIN_PATH)
_mt = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["mtrl_train_v4"] = _mt
_spec.loader.exec_module(_mt)  # type: ignore[union-attr]
MTRLActorCriticPolicy = _mt.MTRLActorCriticPolicy
TASK_NAMES = _mt.TASK_NAMES
NUM_TASKS = _mt.NUM_TASKS
SHARED_OBS_DIM = _mt.SHARED_OBS_DIM
BASE_OBS_DIM = _mt.BASE_OBS_DIM
FULL_OBS_DIM = _mt.FULL_OBS_DIM
build_hex_genome = _mt.build_hex_genome
_task_config = _mt._task_config
UPRIGHT_BONUS = _mt.UPRIGHT_BONUS
TILT_TERMINATE_COS = _mt.TILT_TERMINATE_COS
EXTRA_YAW_RATE_PEN = _mt.EXTRA_YAW_RATE_PEN
VELOCITY_REWARD_COEF = _mt.VELOCITY_REWARD_COEF
ALTITUDE_FLOOR_Z = _mt.ALTITUDE_FLOOR_Z
ALTITUDE_FLOOR_COEF = _mt.ALTITUDE_FLOOR_COEF


# ─────────────────────────────────────────────────────────────────────────────
# Single-task wrapper that appends a fixed task one-hot
# ─────────────────────────────────────────────────────────────────────────────

class _TaskOneHotWrapper(VecEnv):
    def __init__(self, raw_env: VecEnv, task_id: int, num_tasks: int):
        self._env = raw_env
        self._task_oh = np.zeros((raw_env.num_envs, num_tasks), dtype=np.float32)
        self._task_oh[:, task_id] = 1.0
        full_dim = raw_env.observation_space.shape[0] + num_tasks
        obs_space = spaces.Box(-np.inf, np.inf, (full_dim,), dtype=np.float32)
        VecEnv.__init__(self, raw_env.num_envs, obs_space, raw_env.action_space)

    def reset(self):
        return np.concatenate([self._env.reset().astype(np.float32), self._task_oh], axis=1)

    def step_async(self, actions):
        self._env.step_async(actions)

    def step_wait(self):
        o, r, d, info = self._env.step_wait()
        o = np.concatenate([o.astype(np.float32), self._task_oh], axis=1)
        for i, item in enumerate(info):
            if "terminal_observation" in item:
                item["terminal_observation"] = np.concatenate(
                    [item["terminal_observation"].astype(np.float32), self._task_oh[i]]
                )
        return o, r, d, info

    def close(self): self._env.close()
    def seed(self, seed=None): return []
    def get_attr(self, attr_name, indices=None): raise AttributeError(attr_name)
    def set_attr(self, attr_name, value, indices=None): pass
    def env_method(self, method_name, *args, indices=None, **kwargs): pass
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def render(self, mode="human"): return {}


# ─────────────────────────────────────────────────────────────────────────────
# Rollout
# ─────────────────────────────────────────────────────────────────────────────

def _rollout(task_name, task_id, propellers, model, vn_path, device, dt, n_steps,
             seed, gate_density):
    gpos, gyaw, spos = _task_config(task_name, density=gate_density)
    is_hover = (task_name == "hover")
    # Tight bounds — divergence ends the episode instead of drifting to ±200m.
    # Slalom's natural x extent is large, so widen its x bound only.
    xb = (-20.0, 20.0)
    if task_name == "slalom":
        xb = (-5.0, float(gpos[:, 0].max()) + 5.0)
    # v4: figure8 fixed-start to match training; slalom/shuttle-run random.
    random_init = (not is_hover) and (task_name != "figure8")
    raw_env = TorchDroneGateEnv(
        num_envs=1,
        propellers=propellers,
        gates_pos=gpos,
        gate_yaw=gyaw,
        start_pos=spos,
        x_bounds=xb,
        y_bounds=(-20.0, 20.0),
        z_bounds=(-10.0, 0.5),
        gates_ahead=2,
        device=device,
        dt=dt,
        seed=seed,
        max_steps=n_steps,
        initialize_at_random_gates=random_init,
        upright_bonus=0.0 if is_hover else UPRIGHT_BONUS,
        tilt_terminate_cos=TILT_TERMINATE_COS,
        extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
        velocity_reward_coef=0.0 if is_hover else VELOCITY_REWARD_COEF,
        altitude_floor_z=ALTITUDE_FLOOR_Z,
        altitude_floor_coef=0.0 if is_hover else ALTITUDE_FLOOR_COEF,
    )
    wrapped = _TaskOneHotWrapper(raw_env, task_id, NUM_TASKS)

    if vn_path is not None:
        vn = VecNormalize.load(vn_path, wrapped)
        vn.training = False
        vn.norm_reward = False
        env = vn
    else:
        env = wrapped

    pos_ned = np.zeros((n_steps, 3), dtype=np.float32)
    euler = np.zeros((n_steps, 3), dtype=np.float32)
    gates_passed = 0
    done_at = n_steps
    term_cause = "timeout"
    # Track *every* termination across the rollout (envs reset and continue
    # via the wrapper), so the user sees if e.g. the policy crashes 6 times.
    term_counts = {"collided": 0, "tilted": 0, "out_of_bounds": 0,
                   "diverged": 0, "timeout": 0}
    obs = env.reset()
    for i in range(n_steps):
        pos_ned[i] = raw_env.world_states[0, 0:3].cpu().numpy()
        euler[i] = raw_env.world_states[0, 6:9].cpu().numpy()
        action, _ = model.predict(obs, deterministic=False)
        obs, _, dones, infos = env.step(action)
        gp = infos[0].get("num_gates_passed", None)
        if gp is not None:
            gates_passed = max(gates_passed, int(np.asarray(gp)[0]))
        if dones[0]:
            info = infos[0]
            cause = None
            for flag in ("collided", "tilted", "out_of_bounds", "diverged"):
                if info.get(flag):
                    cause = flag
                    break
            if cause is None:
                cause = "timeout"
            term_counts[cause] += 1
            if done_at == n_steps:
                done_at = i + 1
                term_cause = cause
    return pos_ned[:done_at], euler[:done_at], gates_passed, raw_env, term_cause, term_counts


# ─────────────────────────────────────────────────────────────────────────────
# MuJoCo scene
# ─────────────────────────────────────────────────────────────────────────────

def _build_mjmodel(bp, gpos, gyaw, dt, start_xyz_enu):
    arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
    mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06
    drone_spec = blueprint_to_mjspec(
        bp, motor_mass=0.01, arm_mass=0.034 * mean_arm_len, body_name="hex",
    )
    # Size the floor from the task's gate extents so it actually covers
    # the corridor. Slalom alone needs ~400m of floor.
    gpos_np = np.asarray(gpos, dtype=np.float64)
    pad = 3.0
    span_x = max(abs(float(gpos_np[:, 0].min())),
                 abs(float(gpos_np[:, 0].max())),
                 abs(float(start_xyz_enu[0])))
    span_y = max(abs(float(gpos_np[:, 1].min())),
                 abs(float(gpos_np[:, 1].max())),
                 abs(float(start_xyz_enu[1])))
    floor_w = 2.0 * (span_x + pad)
    floor_h = 2.0 * (span_y + pad)
    world = SimpleFlatWorld(floor_size=(floor_w, floor_h, 1))
    world.spawn(
        drone_spec,
        position=(float(start_xyz_enu[0]), float(start_xyz_enu[1]), float(start_xyz_enu[2])),
        correct_collision_with_floor=False,
    )

    gate_pos = np.asarray(gpos, dtype=np.float64)
    gate_yaw = np.asarray(gyaw, dtype=np.float64)
    half = 1.5 / 2.0
    depth = 0.02
    max_markers = min(len(gate_pos), 60)
    for i in range(max_markers):
        gx, gy = float(gate_pos[i, 0]), float(gate_pos[i, 1])
        gz = float(-gate_pos[i, 2])
        yaw = float(gate_yaw[i])
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        b = world.spec.worldbody.add_body(name=f"gate_{i}", pos=[gx, gy, gz], quat=[qw, 0.0, 0.0, qz])
        b.add_geom(
            name=f"gate_plane_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[depth, half, half], rgba=(1.0, 0.55, 0.0, 0.3),
            contype=0, conaffinity=0,
        )
        b.add_geom(
            name=f"gate_center_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.06, 0.0, 0.0], rgba=(1.0, 0.1, 0.1, 1.0),
            contype=0, conaffinity=0,
        )

    model_mj = world.spec.compile()
    data_mj = mujoco.MjData(model_mj)
    model_mj.opt.timestep = dt
    return model_mj, data_mj


def _ned_to_enu(pos_ned, euler):
    pos_enu = pos_ned.copy()
    pos_enu[:, 2] = -pos_enu[:, 2]
    phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    cp, sp = np.cos(theta / 2), np.sin(theta / 2)
    cr, sr = np.cos(phi / 2), np.sin(phi / 2)
    quat_enu = np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        -(cr * sp * cy + sr * cp * sy),
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)
    return pos_enu, quat_enu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--vecnormalize", default=None)
    parser.add_argument("--task", choices=list(TASK_NAMES) + ["all"], default="all")
    parser.add_argument("--rollout-time", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gate-density", type=int, default=3,
                        help="Must match the value used at training time.")
    parser.add_argument("--blueprint-json", default=None,
                        help="Optional DroneBlueprint .json (as written by ariel's EA "
                             "via best_bp.save_json). Defaults to the standard hex.")
    parser.add_argument("--no-view", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    vn_path = args.vecnormalize
    if vn_path is None:
        candidate = Path(args.policy).with_name("vecnormalize.pkl")
        vn_path = str(candidate) if candidate.exists() else None
    if vn_path:
        print(f"using vecnormalize stats: {vn_path}")
    else:
        print("WARNING: no vecnormalize.pkl found — policy will see raw obs (likely poor).")

    if args.blueprint_json is not None:
        bp = DroneBlueprint.load_json(args.blueprint_json)
        print(f"morphology: loaded blueprint from {args.blueprint_json}")
    else:
        genome = build_hex_genome()
        bp = spherical_angular_to_blueprint(genome)
    propellers = blueprint_to_propellers(bp, convention="ned")

    model = PPO.load(
        args.policy, env=None, device=args.device,
        custom_objects={"policy_class": MTRLActorCriticPolicy},
    )
    print(f"policy loaded: {args.policy}")
    print(f"obs_dim={model.observation_space.shape[0]}  act_dim={model.action_space.shape[0]}")

    tasks_to_run = list(TASK_NAMES) if args.task == "all" else [args.task]
    n_steps = int(args.rollout_time / args.dt) + 1

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.policy).parent / "viz"
    if args.no_view:
        out_dir.mkdir(parents=True, exist_ok=True)

    for task_name in tasks_to_run:
        task_id = TASK_NAMES.index(task_name)
        print(f"\n=== TASK: {task_name} (id={task_id}) ===")
        pos_ned, euler, gates_passed, raw_env, term_cause, term_counts = _rollout(
            task_name, task_id, propellers, model, vn_path,
            args.device, args.dt, n_steps, args.seed, args.gate_density,
        )
        n_logged = len(pos_ned)
        counts_str = " ".join(f"{k}={v}" for k, v in term_counts.items() if v > 0) or "none"
        print(
            f"  steps_used={n_logged}/{n_steps}  total gates passed={gates_passed}  "
            f"first_term={term_cause}\n"
            f"  termination breakdown over full rollout: {counts_str}\n"
            f"  pos x: [{pos_ned[:,0].min():+.2f}, {pos_ned[:,0].max():+.2f}]\n"
            f"  pos y: [{pos_ned[:,1].min():+.2f}, {pos_ned[:,1].max():+.2f}]\n"
            f"  pos z (NED, +down): [{pos_ned[:,2].min():+.2f}, {pos_ned[:,2].max():+.2f}]"
        )

        pos_enu, quat_enu = _ned_to_enu(pos_ned, euler)
        gpos, gyaw, _spos = _task_config(task_name, density=args.gate_density)
        model_mj, data_mj = _build_mjmodel(bp, gpos, gyaw, args.dt, pos_enu[0])

        def _set_pose(idx: int) -> None:
            p, q = pos_enu[idx], quat_enu[idx]
            data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
            data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
            data_mj.qvel[:] = 0.0
            mujoco.mj_forward(model_mj, data_mj)

        if not args.no_view:
            print(f"  launching viewer for {task_name} — close window to advance.")
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
            out = out_dir / f"{task_name}_{_time.strftime('%Y%m%d_%H%M%S')}.mp4"
            recorder = VideoRecorder(
                file_name=out.stem, output_folder=out.parent,
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
            print(f"  rendered in {_time.time() - t_render:.2f}s → {out}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
