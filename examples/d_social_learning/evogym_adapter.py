"""EvoGym-specific observation extraction and action scaling for DistributedMLP.

# Environment: evogym-venv (Python 3.10) — EvoGym requires Python 3.10.
#              Do NOT run with the main uv/ariel venv.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ariel.simulation.controllers.distributed_mlp import EMPTY_NODE, NodeObservation

_VOXEL_ONEHOT_SIZE = 5
_BODY_ROWS = 5
_BODY_COLS = 5


def get_actuator_order(body: np.ndarray) -> list[int]:
    """Return flat indices of actuator voxels (types 3 or 4) in row-major order."""
    rows, cols = body.shape
    return [
        r * cols + c
        for r in range(rows)
        for c in range(cols)
        if body[r, c] in (3.0, 4.0)
    ]


def body_to_adjacency(body: np.ndarray) -> dict[int, list[int | None]]:
    """Moore d=1 neighbourhood for each actuator voxel.

    Returns ``{actuator_flat_idx: [nb_flat_idx_or_None, ...]}``.  The 8-slot
    list runs through the 3×3 window in row-major order (centre skipped).
    ``None`` means outside the body or empty voxel.
    """
    rows, cols = body.shape
    actuators = get_actuator_order(body)
    result: dict[int, list[int | None]] = {}

    for flat in actuators:
        r, c = divmod(flat, cols)
        slots: list[int | None] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and body[nr, nc] != 0:
                    slots.append(nr * cols + nc)
                else:
                    slots.append(None)
        result[flat] = slots

    return result


def _build_voxel_sensor_map(body: np.ndarray) -> defaultdict[int, list[int]]:
    """Map voxel flat index → list of corner-point sensor indices."""
    rows, cols = body.shape
    sensor_grid: dict[tuple[int, int], int] = {}
    idx = 0
    for x in range(rows):
        for y in range(cols):
            if body[x, y] > 0:
                for dx in (0, 1):
                    for dy in (0, 1):
                        key = (x + dx, y + dy)
                        if key not in sensor_grid:
                            sensor_grid[key] = idx
                            idx += 1

    voxel_map: defaultdict[int, list[int]] = defaultdict(list)
    for x in range(rows):
        for y in range(cols):
            if body[x, y] == 0:
                continue
            flat = x * cols + y
            for dx in (0, 1):
                for dy in (0, 1):
                    voxel_map[flat].append(sensor_grid[(x + dx, y + dy)])

    return voxel_map


def _voxel_onehot(vtype: float) -> np.ndarray:
    oh = np.zeros(_VOXEL_ONEHOT_SIZE, dtype=np.float32)
    idx = int(vtype)
    if 0 <= idx < _VOXEL_ONEHOT_SIZE:
        oh[idx] = 1.0
    return oh


def _rect_area(corners: list[tuple[float, float]]) -> float:
    a, b, c, _ = corners
    w = math.hypot(b[0] - a[0], b[1] - a[1])
    h = math.hypot(c[0] - a[0], c[1] - a[1])
    return w * h


def get_node_inputs(
    sim,
    body: np.ndarray,
    adjacency: dict[int, list[int | None]],
    timestep: int,
) -> tuple[list[tuple[NodeObservation, list[NodeObservation]]], float]:
    """Build ``(node_inputs, time_signal)`` for ``DistributedMLP.forward_all``."""
    current_time = sim.get_time()
    positions = sim.object_pos_at_time(current_time, "robot")
    velocities = sim.object_vel_at_time(current_time, "robot")

    voxel_map = _build_voxel_sensor_map(body)
    actuators = get_actuator_order(body)

    def make_obs(flat_idx: int) -> NodeObservation:
        vtype = body.flat[flat_idx]
        if vtype == 0 or flat_idx not in voxel_map:
            return EMPTY_NODE
        sensor_indices = voxel_map[flat_idx]
        corners = [(positions[0][i], positions[1][i]) for i in sensor_indices]
        vxs = [velocities[0][i] for i in sensor_indices]
        vys = [velocities[1][i] for i in sensor_indices]
        vx = np.clip(float(sum(vxs) / len(vxs)), -20.0, 20.0) / 20.0
        vy = np.clip(float(sum(vys) / len(vys)), -20.0, 20.0) / 20.0
        area = _rect_area(corners) if len(corners) == 4 else 0.0
        state = float(np.clip(area, 0.0, 3.0) / 1.5 - 1.0)
        oh = _voxel_onehot(vtype)
        return NodeObservation(vx, vy, state, oh)

    # EvoGym default timestep is 0.01s; 100 steps = 1.0s cycle
    time_signal = np.sin(2 * np.pi * timestep / 100)

    node_inputs: list[tuple[NodeObservation, list[NodeObservation]]] = []
    for flat in actuators:
        self_obs = make_obs(flat)
        nb_slots = adjacency.get(flat, [None] * 8)
        nb_obs = [make_obs(nb) if nb is not None else EMPTY_NODE for nb in nb_slots]
        node_inputs.append((self_obs, nb_obs))

    return node_inputs, time_signal


def scale_actions(raw: np.ndarray) -> np.ndarray:
    """Scale tanh output [-1, 1] to EvoGym action range [0.6, 1.6]."""
    return 0.6 + (raw + 1.0) * 0.5


def get_standard_obs(
    sim,
    body: np.ndarray,
    timestep: int,
) -> np.ndarray:
    """Full-state observation for ``StandardMLP`` on EvoGym.

    Returns shape ``(2*nu + 9,)``:
    ``[volume_normalised(nu), mean_vx(nu), mean_x, mean_y, mean_vx, mean_vy,
    0, 0, 0, 0, time_signal]``.
    """
    actuators = get_actuator_order(body)
    nu = len(actuators)
    voxel_map = _build_voxel_sensor_map(body)

    current_time = sim.get_time()
    positions = sim.object_pos_at_time(current_time, "robot")
    velocities = sim.object_vel_at_time(current_time, "robot")

    def voxel_state(flat_idx: int) -> tuple[float, float, float, float, float]:
        if body.flat[flat_idx] == 0 or flat_idx not in voxel_map:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        sensor_indices = voxel_map[flat_idx]
        corners = [(positions[0][i], positions[1][i]) for i in sensor_indices]
        vxs = [velocities[0][i] for i in sensor_indices]
        vys = [velocities[1][i] for i in sensor_indices]
        vx = float(sum(vxs) / len(vxs))
        vy = float(sum(vys) / len(vys))
        area = _rect_area(corners) if len(corners) == 4 else 0.0
        cx = float(sum(c[0] for c in corners) / len(corners))
        cy = float(sum(c[1] for c in corners) / len(corners))
        return (vx, vy, area, cx, cy)

    jpos = np.zeros(nu, dtype=np.float32)
    jvel = np.zeros(nu, dtype=np.float32)
    for i, flat in enumerate(actuators):
        vx, vy, area, _, _ = voxel_state(flat)
        jpos[i] = float(np.clip(area / 1.5 - 1.0, -1.0, 1.0))
        jvel[i] = vx

    all_vx = float(np.mean(velocities[0]))
    all_vy = float(np.mean(velocities[1]))
    mean_x = float(np.mean(positions[0]))
    mean_y = float(np.mean(positions[1]))
    time_signal = (timestep % 100) / 99.0
    body_state = np.array(
        [mean_x, mean_y, all_vx, all_vy, 0.0, 0.0, 0.0, 0.0, time_signal],
        dtype=np.float32,
    )

    return np.concatenate([jpos, jvel, body_state])
