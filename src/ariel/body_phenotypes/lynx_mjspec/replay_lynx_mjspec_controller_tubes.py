import argparse
import time
from typing import cast

import mujoco
import mujoco.viewer
import numpy as np

from ariel.body_phenotypes.lynx_mjspec.lynx_arm import LynxArm
from ariel.body_phenotypes.lynx_mjspec.table import TableWorld

NUM_JOINTS = 6
NUM_TUBES = 5

TUBE_MIN = 0.1
TUBE_MAX = 1.0

DEFAULT_TARGET = [0.20, 0.00, 1.20]
DEFAULT_SIM_STEPS = 3500
DEFAULT_CTRL_FREQ = 20
DEFAULT_TOUCH_THRESHOLD = 0.01

TUBES_PATH = "best_lynx_mjspec_tube_lengths.npy"
BRAIN_PATH = "best_lynx_mjspec_brain_weights.npy"


class FastNumpyNetwork:
    """Fast 3-layer NumPy MLP compatible with evolution script weights."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, weights: np.ndarray) -> None:
        w1_end = hidden_size * input_size
        b1_end = w1_end + hidden_size

        w2_end = b1_end + (hidden_size * hidden_size)
        b2_end = w2_end + hidden_size

        w3_end = b2_end + (output_size * hidden_size)
        b3_end = w3_end + output_size

        if len(weights) != b3_end:
            raise ValueError(f"Invalid weight size {len(weights)}, expected {b3_end}")

        self.w1 = weights[0:w1_end].reshape(hidden_size, input_size)
        self.b1 = weights[w1_end:b1_end]

        self.w2 = weights[b1_end:w2_end].reshape(hidden_size, hidden_size)
        self.b2 = weights[w2_end:b2_end]

        self.w3 = weights[b2_end:w3_end].reshape(output_size, hidden_size)
        self.b3 = weights[w3_end:b3_end]

    @staticmethod
    def _elu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0.0, x, np.exp(x) - 1.0)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.dot(self.w1, x) + self.b1
        x = self._elu(x)

        x = np.dot(self.w2, x) + self.b2
        x = self._elu(x)

        x = np.dot(self.w3, x) + self.b3
        return np.tanh(x)


def infer_hidden_size(num_weights: int, input_size: int, output_size: int) -> int | None:
    # total = h^2 + h*(input + output + 2) + output
    a = 1
    b = input_size + output_size + 2
    c = output_size - num_weights

    disc = b * b - 4 * a * c
    if disc < 0:
        return None

    root = (-b + np.sqrt(disc)) / (2 * a)
    h = int(round(float(root)))
    if h <= 0:
        return None

    expected = (h * input_size + h) + (h * h + h) + (output_size * h + output_size)
    if expected != num_weights:
        return None
    return h


def get_actuated_joint_ids(model: mujoco.MjModel, count: int = NUM_JOINTS) -> list[int]:
    ids: list[int] = []
    for i in range(min(count, model.nu)):
        jid = int(model.actuator_trnid[i, 0])
        if jid >= 0:
            ids.append(jid)
    return ids


def resolve_site_id(model: mujoco.MjModel, base_name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, base_name)
    if sid != -1:
        return sid
    for i in range(model.nsite):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if name.endswith(base_name):
            return i
    return -1


def get_joint_state(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    q = np.zeros(NUM_JOINTS, dtype=np.float64)
    qd = np.zeros(NUM_JOINTS, dtype=np.float64)
    for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
        qaddr = int(model.jnt_qposadr[jid])
        daddr = int(model.jnt_dofadr[jid])
        q[i] = data.qpos[qaddr]
        qd[i] = data.qvel[daddr]
    return q, qd


def build_model(tube_lengths: np.ndarray) -> tuple[mujoco.MjModel, mujoco.MjData, int, int, list[int]]:
    config = {
        "num_joints": 6,
        "genotype_tube": [1, 1, 1, 1, 1],
        "genotype_joints": 6,
        "tube_lengths": np.clip(tube_lengths, TUBE_MIN, TUBE_MAX).tolist(),
        "rotation_angles": [0.0, -1.57, 0.0, 0.0, 0.0, 0.0],
        "task": "reach",
    }

    arm = LynxArm(config=config)
    world = TableWorld()
    world.spawn(arm.spec)

    model = cast(mujoco.MjModel, world.spec.compile())
    data = mujoco.MjData(model)

    tcp_sid = resolve_site_id(model, "tcp")
    tgt_sid = resolve_site_id(model, "target")
    if tcp_sid == -1 or tgt_sid == -1:
        raise RuntimeError("Could not resolve TCP/target site IDs")

    joint_ids = get_actuated_joint_ids(model, count=NUM_JOINTS)
    return model, data, tcp_sid, tgt_sid, joint_ids


def run_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    net: FastNumpyNetwork,
    tcp_sid: int,
    tgt_sid: int,
    joint_ids: list[int],
    target: np.ndarray,
    sim_steps: int,
    ctrl_freq: int,
) -> tuple[float, float, float, float | None]:
    model.site_pos[tgt_sid] = target
    mujoco.mj_forward(model, data)

    initial_distance = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
    min_distance = initial_distance
    final_distance = initial_distance
    first_touch_time = None

    for step in range(sim_steps):
        if step % ctrl_freq == 0:
            tcp_pos = data.site_xpos[tcp_sid]
            target_pos = data.site_xpos[tgt_sid]
            rel_target = target_pos - tcp_pos

            q, qd = get_joint_state(model, data, joint_ids)
            phase = np.array([
                2.0 * np.sin(data.time * 2.0 * np.pi),
                2.0 * np.cos(data.time * 2.0 * np.pi),
            ])
            obs = np.concatenate([q, qd, rel_target, phase]).astype(np.float64)

            action = net.forward(obs) * 0.35
            desired = q + action

            for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
                if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                    lo, hi = model.jnt_range[jid]
                    desired[i] = np.clip(desired[i], lo, hi)

            data.ctrl[:NUM_JOINTS] = desired

        mujoco.mj_step(model, data)
        d = float(np.linalg.norm(data.site_xpos[tcp_sid] - data.site_xpos[tgt_sid]))
        final_distance = d
        if d < min_distance:
            min_distance = d
        if first_touch_time is None and d <= DEFAULT_TOUCH_THRESHOLD:
            first_touch_time = float(data.time)

    return initial_distance, min_distance, final_distance, first_touch_time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay evolved Lynx MjSpec controller+tube solution")
    parser.add_argument("--tubes-path", type=str, default=TUBES_PATH)
    parser.add_argument("--brain-path", type=str, default=BRAIN_PATH)
    parser.add_argument("--hidden-size", type=int, default=None, help="Override hidden size if auto-infer fails")
    parser.add_argument("--sim-steps", type=int, default=DEFAULT_SIM_STEPS)
    parser.add_argument("--ctrl-freq", type=int, default=DEFAULT_CTRL_FREQ)
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET[0])
    parser.add_argument("--target-y", type=float, default=DEFAULT_TARGET[1])
    parser.add_argument("--target-z", type=float, default=DEFAULT_TARGET[2])
    parser.add_argument("--eval-only", action="store_true", help="Run headless metrics and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tubes = np.load(args.tubes_path)
    brain = np.load(args.brain_path)

    if len(tubes) != NUM_TUBES:
        raise ValueError(f"Invalid tube length vector size: got {len(tubes)}, expected {NUM_TUBES}")

    input_size = NUM_JOINTS + NUM_JOINTS + 3 + 2
    output_size = NUM_JOINTS

    inferred_hidden = infer_hidden_size(len(brain), input_size, output_size)
    hidden_size = args.hidden_size if args.hidden_size is not None else inferred_hidden
    if hidden_size is None:
        raise ValueError(
            "Could not infer hidden size from brain weights. Provide --hidden-size explicitly."
        )

    target = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)

    model, data, tcp_sid, tgt_sid, joint_ids = build_model(tubes)
    net = FastNumpyNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size, weights=brain)

    if args.eval_only:
        init_d, min_d, final_d, touch_t = run_rollout(
            model=model,
            data=data,
            net=net,
            tcp_sid=tcp_sid,
            tgt_sid=tgt_sid,
            joint_ids=joint_ids,
            target=target,
            sim_steps=int(args.sim_steps),
            ctrl_freq=int(args.ctrl_freq),
        )
        print(f"initial_distance={init_d:.6f}")
        print(f"min_distance={min_d:.6f}")
        print(f"final_distance={final_d:.6f}")
        print(f"touch_time={touch_t if touch_t is not None else 'none'}")
        print(f"tube_lengths={np.round(tubes, 4).tolist()}")
        return

    model.site_pos[tgt_sid] = target
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if step % int(args.ctrl_freq) == 0:
                tcp_pos = data.site_xpos[tcp_sid]
                target_pos = data.site_xpos[tgt_sid]
                rel_target = target_pos - tcp_pos

                q, qd = get_joint_state(model, data, joint_ids)
                phase = np.array([
                    2.0 * np.sin(data.time * 2.0 * np.pi),
                    2.0 * np.cos(data.time * 2.0 * np.pi),
                ])
                obs = np.concatenate([q, qd, rel_target, phase]).astype(np.float64)

                action = net.forward(obs) * 0.35
                desired = q + action

                for i, jid in enumerate(joint_ids[:NUM_JOINTS]):
                    if getattr(model, "jnt_limited", None) is not None and model.jnt_limited[jid]:
                        lo, hi = model.jnt_range[jid]
                        desired[i] = np.clip(desired[i], lo, hi)

                data.ctrl[:NUM_JOINTS] = desired

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1 / 60.0)
            step += 1


if __name__ == "__main__":
    main()
