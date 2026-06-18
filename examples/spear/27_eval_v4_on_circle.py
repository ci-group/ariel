"""Out-of-distribution evaluation: run the v4 MTRL policy on the `circle`
task, which the policy was NEVER trained on.

The v4 policy was trained on a 4-task one-hot {figure8, slalom, shuttle-run,
hover}. CircleGates is a NEW task, so we have to pick which of the trained
task identities to present to the policy. Figure-8 is the closest analogue
(both are tight, closed, low-altitude loops), so it's the default. Use
`--task-id all` to compare across all four task-encoder selections.

Run:
    uv run examples/spear/27_eval_v4_on_circle.py \\
        --policy __data__/spear_rl_hex_mtrl_v4/RUN/policy.zip

    # try every task-encoder one after another:
    uv run examples/spear/27_eval_v4_on_circle.py \\
        --policy ... --task-id all

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
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.environments import SimpleFlatWorld
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv


# Import v4 policy class + shaping constants
_TRAIN_PATH = Path(__file__).with_name("27_train_rl_hex_mtrl_v4.py")
_spec = importlib.util.spec_from_file_location("mtrl_train_v4", _TRAIN_PATH)
_mt = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["mtrl_train_v4"] = _mt
_spec.loader.exec_module(_mt)  # type: ignore[union-attr]
MTRLActorCriticPolicy = _mt.MTRLActorCriticPolicy
TASK_NAMES = _mt.TASK_NAMES
NUM_TASKS = _mt.NUM_TASKS
build_hex_genome = _mt.build_hex_genome
UPRIGHT_BONUS = _mt.UPRIGHT_BONUS
TILT_TERMINATE_COS = _mt.TILT_TERMINATE_COS
EXTRA_YAW_RATE_PEN = _mt.EXTRA_YAW_RATE_PEN
VELOCITY_REWARD_COEF = _mt.VELOCITY_REWARD_COEF
ALTITUDE_FLOOR_Z = _mt.ALTITUDE_FLOOR_Z
ALTITUDE_FLOOR_COEF = _mt.ALTITUDE_FLOOR_COEF


class _TaskOneHotWrapper(VecEnv):
    """Same as in 27_eval; appends a fixed task one-hot to obs."""

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


def _build_circle_env(propellers, task_one_hot_id: int, device: str, dt: float,
                      n_steps: int, seed: int):
    """Single-env CircleGates rollout env, configured to match v4 training shape."""
    cfg = GATE_CONFIGS["circle"]
    gpos = np.asarray(cfg.gate_pos, dtype=np.float64)
    gyaw = np.asarray(cfg.gate_yaw, dtype=np.float64)
    spos = np.asarray(cfg.starting_pos, dtype=np.float64)

    raw_env = TorchDroneGateEnv(
        num_envs=1,
        propellers=propellers,
        gates_pos=gpos,
        gate_yaw=gyaw,
        start_pos=spos,
        x_bounds=(-20.0, 20.0),
        y_bounds=(-20.0, 20.0),
        z_bounds=(-10.0, 0.5),
        gates_ahead=2,
        device=device,
        dt=dt,
        seed=seed,
        max_steps=n_steps,
        initialize_at_random_gates=False,
        # Match v4 training-time shaping so termination behaves identically
        upright_bonus=UPRIGHT_BONUS,
        tilt_terminate_cos=TILT_TERMINATE_COS,
        extra_yaw_rate_pen=EXTRA_YAW_RATE_PEN,
        velocity_reward_coef=VELOCITY_REWARD_COEF,
        altitude_floor_z=ALTITUDE_FLOOR_Z,
        altitude_floor_coef=ALTITUDE_FLOOR_COEF,
    )
    wrapped = _TaskOneHotWrapper(raw_env, task_one_hot_id, NUM_TASKS)
    return raw_env, wrapped, gpos, gyaw, spos


def _rollout(env, raw_env, model, n_steps: int):
    pos_ned = np.zeros((n_steps, 3), dtype=np.float32)
    euler = np.zeros((n_steps, 3), dtype=np.float32)
    gates_passed = 0
    done_at = n_steps
    term_cause = "timeout"
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
            cause = "timeout"
            for flag in ("collided", "tilted", "out_of_bounds", "diverged"):
                if infos[0].get(flag):
                    cause = flag
                    break
            term_counts[cause] += 1
            if done_at == n_steps:
                done_at = i + 1
                term_cause = cause
    return pos_ned[:done_at], euler[:done_at], gates_passed, term_cause, term_counts


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


def _build_scene(bp, gpos, gyaw, dt, pos_enu):
    arm_lengths = [bp.payload(a).length for a in bp.children(bp.root_id)]  # type: ignore[arg-type]
    mean_arm_len = float(np.mean(arm_lengths)) if arm_lengths else 0.06
    drone_spec = blueprint_to_mjspec(
        bp, motor_mass=0.01, arm_mass=0.034 * mean_arm_len, body_name="hex",
    )

    # Floor sized to cover the circle (radius ~1.5m) plus drone path margin.
    pad = 3.0
    all_x = np.concatenate([gpos[:, 0], pos_enu[:, 0]])
    all_y = np.concatenate([gpos[:, 1], pos_enu[:, 1]])
    floor_w = 2.0 * (max(abs(float(all_x.min())), abs(float(all_x.max()))) + pad)
    floor_h = 2.0 * (max(abs(float(all_y.min())), abs(float(all_y.max()))) + pad)
    world = SimpleFlatWorld(floor_size=(floor_w, floor_h, 1))
    world.spawn(drone_spec,
                position=(float(pos_enu[0, 0]), float(pos_enu[0, 1]), float(pos_enu[0, 2])),
                correct_collision_with_floor=False)

    half = 1.5 / 2.0
    depth = 0.02
    for i in range(len(gpos)):
        gx, gy = float(gpos[i, 0]), float(gpos[i, 1])
        gz_enu = float(-gpos[i, 2])
        yaw = float(gyaw[i])
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        b = world.spec.worldbody.add_body(
            name=f"gate_{i}", pos=[gx, gy, gz_enu], quat=[qw, 0.0, 0.0, qz],
        )
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


def _viewer_loop(model_mj, data_mj, pos_enu, quat_enu, dt: float):
    def _set_pose(idx: int) -> None:
        p, q = pos_enu[idx], quat_enu[idx]
        data_mj.qpos[0:3] = [float(p[0]), float(p[1]), float(p[2])]
        data_mj.qpos[3:7] = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        data_mj.qvel[:] = 0.0
        mujoco.mj_forward(model_mj, data_mj)

    _set_pose(0)
    n_frames = len(pos_enu)
    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        while viewer.is_running():
            for idx in range(n_frames):
                if not viewer.is_running():
                    break
                t0 = _time.time()
                _set_pose(idx)
                viewer.sync()
                slack = dt - (_time.time() - t0)
                if slack > 0:
                    _time.sleep(slack)
            # 1-second hold between loops
            t_hold = _time.time()
            while viewer.is_running() and _time.time() - t_hold < 1.0:
                viewer.sync()
                _time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--vecnormalize", default=None,
                        help="Path to vecnormalize.pkl. Defaults to <policy_dir>/vecnormalize.pkl.")
    parser.add_argument(
        "--task-id", default="figure8",
        help="Which trained task encoder to present to the policy. "
             "Accepted: figure8 / slalom / shuttle-run / hover, an integer 0-3, "
             "or 'all' to roll out each in sequence. Default: figure8.",
    )
    parser.add_argument("--rollout-time", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-view", action="store_true")
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
        print("WARNING: no vecnormalize.pkl found — policy will see raw obs.")

    # Resolve --task-id into a list of (name, id) pairs
    if args.task_id == "all":
        task_pairs = list(enumerate(TASK_NAMES))
        task_pairs = [(i, n) for i, n in task_pairs]
    else:
        try:
            tid = int(args.task_id)
            tname = TASK_NAMES[tid]
        except ValueError:
            if args.task_id not in TASK_NAMES:
                raise SystemExit(
                    f"--task-id must be one of {list(TASK_NAMES)}, an int 0-3, or 'all'."
                )
            tid = TASK_NAMES.index(args.task_id)
            tname = args.task_id
        task_pairs = [(tid, tname)]

    genome = build_hex_genome()
    bp = spherical_angular_to_blueprint(genome)
    propellers = blueprint_to_propellers(bp, convention="ned")

    model = PPO.load(
        args.policy, env=None, device=args.device,
        custom_objects={"policy_class": MTRLActorCriticPolicy},
    )
    print(f"policy loaded: {args.policy}")
    print(f"obs_dim={model.observation_space.shape[0]}  "
          f"act_dim={model.action_space.shape[0]}")

    n_steps = int(args.rollout_time / args.dt) + 1

    for tid, tname in task_pairs:
        print(f"\n=== CIRCLE task, presenting one-hot for '{tname}' (id={tid}) ===")
        raw_env, wrapped, gpos, gyaw, spos = _build_circle_env(
            propellers=propellers, task_one_hot_id=tid,
            device=args.device, dt=args.dt, n_steps=n_steps, seed=args.seed,
        )
        if vn_path is not None:
            vn = VecNormalize.load(vn_path, wrapped)
            vn.training = False
            vn.norm_reward = False
            env = vn
        else:
            env = wrapped

        pos_ned, euler, gates_passed, term_cause, term_counts = _rollout(
            env, raw_env, model, n_steps,
        )
        n_logged = len(pos_ned)
        counts_str = " ".join(f"{k}={v}" for k, v in term_counts.items() if v > 0) or "none"
        elapsed_s = n_logged * args.dt
        gps = gates_passed / elapsed_s if elapsed_s > 0 else 0.0
        print(
            f"  steps_used={n_logged}/{n_steps} ({elapsed_s:.1f}s)  "
            f"gates_passed={gates_passed}  ({gps:.2f}/s)  first_term={term_cause}\n"
            f"  termination breakdown: {counts_str}\n"
            f"  pos x: [{pos_ned[:,0].min():+.2f}, {pos_ned[:,0].max():+.2f}]\n"
            f"  pos y: [{pos_ned[:,1].min():+.2f}, {pos_ned[:,1].max():+.2f}]\n"
            f"  pos z (NED, +down): [{pos_ned[:,2].min():+.2f}, {pos_ned[:,2].max():+.2f}]"
        )

        if args.no_view:
            continue

        pos_enu, quat_enu = _ned_to_enu(pos_ned, euler)
        model_mj, data_mj = _build_scene(bp, gpos, gyaw, args.dt, pos_enu)
        print(f"  launching viewer ({len(pos_enu)} frames, looping until you close it).")
        _viewer_loop(model_mj, data_mj, pos_enu, quat_enu, args.dt)

    print("\ndone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
