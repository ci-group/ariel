"""Shared utilities for the gecko experiments — independent of morphology topology."""

from __future__ import annotations

import mujoco as mj
import numpy as np


def scale_actions(raw: np.ndarray) -> np.ndarray:
    """Scale tanh output [-1, 1] to ARIEL joint range [-π/2, π/2]."""
    return raw * (np.pi / 2)


def get_standard_obs(
    model: mj.MjModel,
    data: mj.MjData,
    actuator_to_module: list[int],
    joint_name_to_module: dict[str, int],
) -> np.ndarray:
    """Full-state observation for StandardMLP.

    Returns shape (2*nu + 9,): [joint_pos(nu), joint_vel(nu), euler(3), linvel(3), angvel(3)].
    """
    joint_pos_map: dict[int, float] = {}
    joint_vel_map: dict[int, float] = {}
    for i in range(model.njnt):
        jname = model.joint(i).name
        matched = next(
            (k for k in joint_name_to_module if jname == k or jname.endswith("_" + k)),
            None,
        )
        if matched is not None:
            mod_idx = joint_name_to_module[matched]
            addr = model.joint(i).qposadr[0]
            vadr = model.joint(i).dofadr[0]
            joint_pos_map[mod_idx] = float(data.qpos[addr])
            joint_vel_map[mod_idx] = float(data.qvel[vadr])

    jpos = np.array([joint_pos_map.get(m, 0.0) for m in actuator_to_module], dtype=np.float32)
    jvel = np.array([joint_vel_map.get(m, 0.0) for m in actuator_to_module], dtype=np.float32)

    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    roll  = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx**2 + qy**2))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))

    euler  = np.array([roll, pitch, yaw], dtype=np.float32)
    linvel = np.array(data.qvel[0:3], dtype=np.float32)
    angvel = np.array(data.qvel[3:6], dtype=np.float32)

    return np.concatenate([jpos, jvel, euler, linvel, angvel])
