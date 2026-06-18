"""Procedural obstacle generation for ARIEL MuJoCo worlds.

Generates randomized obstacle layouts (forest / urban / indoor / random /
gates / custom) and attaches them to an existing ``mj.MjSpec`` via the
``spec.worldbody.add_body(...).add_geom(...)`` builder API used elsewhere in
ARIEL (e.g. the gate markers added in ``25_eval_rl_hex_mtrl.py``).

Generators are ported from ``multi_drone_mujoco/wrappers/obstacles.py`` in
the sibling ``MuJoCo-drones-gym`` repo. The XML-string emitter from the
source (``obstacles_to_xml``) was dropped because ARIEL builds worlds
programmatically with ``MjSpec`` rather than parsing XML.

Coordinate convention
---------------------
Generators emit positions in MuJoCo-native ENU with **Z up** (matching the
source). When attaching to an ARIEL world (e.g. ``SimpleFlatWorld``) no
conversion is needed — feed positions in directly. If a caller is working
in NED (e.g. ``TorchDroneGateEnv`` state), they must convert before
passing positions to ``Obstacle`` (negate Z).

Typical usage
-------------
>>> from ariel.simulation.environments import SimpleFlatWorld
>>> from ariel.simulation.environments.obstacles import (
...     ObstacleConfig, ObstacleType, generate_obstacles, attach_obstacles,
... )
>>> world = SimpleFlatWorld()
>>> cfg = ObstacleConfig(obstacle_type=ObstacleType.FOREST, num_obstacles=20, seed=0)
>>> attach_obstacles(world.spec, generate_obstacles(cfg))
>>> model = world.spec.compile()
"""

from dataclasses import dataclass, field
from enum import Enum

import mujoco as mj
import numpy as np


class ObstacleType(Enum):
    """Pre-defined obstacle environment types."""

    NONE = "none"
    FOREST = "forest"
    URBAN = "urban"
    INDOOR = "indoor"
    RANDOM = "random"
    GATES = "gates"
    CUSTOM = "custom"


@dataclass
class Obstacle:
    """Single obstacle definition.

    ``size`` semantics follow MuJoCo geom conventions:
        - box:      half-extents (sx, sy, sz)
        - cylinder: (radius, half_height)
        - capsule:  (radius, half_height)
        - sphere:   (radius,)
    """

    geom_type: str
    position: np.ndarray
    size: np.ndarray
    rgba: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class ObstacleConfig:
    """Configuration for procedural obstacle generation.

    Parameters
    ----------
    obstacle_type : ObstacleType
        Pre-defined environment layout.
    num_obstacles : int
        Number of obstacles to generate. For ``INDOOR`` this includes the
        4 walls + ceiling; interior count is ``num_obstacles - 5``.
    arena_size : tuple
        ``(x_half, y_half, z_max)`` arena bounds in metres.
    min_spacing : float
        Minimum xy distance between obstacle centres.
    seed : int or None
        Random seed for reproducibility.
    safe_zone_radius : float
        No obstacles within this xy radius of any ``safe_zone_centers`` point.
    safe_zone_centers : ndarray (N, 3) or None
        Points to keep clear (typically spawn locations / target gates).
    custom_obstacles : list[Obstacle]
        Returned verbatim when ``obstacle_type == CUSTOM``.
    """

    obstacle_type: ObstacleType = ObstacleType.NONE
    num_obstacles: int = 20
    arena_size: tuple[float, float, float] = (3.0, 3.0, 2.5)
    min_spacing: float = 0.3
    seed: int | None = None
    safe_zone_radius: float = 0.5
    safe_zone_centers: np.ndarray | None = None
    custom_obstacles: list[Obstacle] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Generators
# ─────────────────────────────────────────────────────────────────────────────

def generate_obstacles(config: ObstacleConfig) -> list[Obstacle]:
    """Generate obstacle list from config."""
    if config.obstacle_type == ObstacleType.NONE:
        return []
    if config.obstacle_type == ObstacleType.CUSTOM:
        return list(config.custom_obstacles)

    rng = np.random.default_rng(config.seed)
    generators = {
        ObstacleType.FOREST: _generate_forest,
        ObstacleType.URBAN: _generate_urban,
        ObstacleType.INDOOR: _generate_indoor,
        ObstacleType.RANDOM: _generate_random,
        ObstacleType.GATES: _generate_gates,
    }
    return generators[config.obstacle_type](config, rng)


def _try_place(position: np.ndarray, placed: list[np.ndarray],
               config: ObstacleConfig) -> bool:
    """Reject positions inside a safe zone or too close to existing obstacles."""
    if config.safe_zone_centers is not None:
        for center in config.safe_zone_centers:
            if np.linalg.norm(position[:2] - center[:2]) < config.safe_zone_radius:
                return False
    for p in placed:
        if np.linalg.norm(position[:2] - p[:2]) < config.min_spacing:
            return False
    return True


def _generate_forest(config: ObstacleConfig, rng: np.random.Generator) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    placed: list[np.ndarray] = []
    xh, yh, zh = config.arena_size
    attempts = 0
    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        radius = rng.uniform(0.03, 0.12)
        height = rng.uniform(0.5, zh)
        pos = np.array([x, y, height / 2])
        if not _try_place(pos, placed, config):
            continue
        g = rng.uniform(0.2, 0.5)
        rgba = np.array([rng.uniform(0.3, 0.6), g, rng.uniform(0.1, 0.3), 1.0])
        obstacles.append(Obstacle(
            geom_type="cylinder",
            position=pos,
            size=np.array([radius, height / 2]),
            rgba=rgba,
        ))
        placed.append(pos)
    return obstacles


def _generate_urban(config: ObstacleConfig, rng: np.random.Generator) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    placed: list[np.ndarray] = []
    xh, yh, zh = config.arena_size
    attempts = 0
    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        sx = rng.uniform(0.1, 0.5)
        sy = rng.uniform(0.1, 0.5)
        sz = rng.uniform(0.3, zh)
        pos = np.array([x, y, sz / 2])
        if not _try_place(pos, placed, config):
            continue
        gray = rng.uniform(0.3, 0.7)
        rgba = np.array([gray, gray, gray * rng.uniform(0.9, 1.1), 1.0])
        obstacles.append(Obstacle(
            geom_type="box",
            position=pos,
            size=np.array([sx, sy, sz / 2]),
            rgba=rgba,
        ))
        placed.append(pos)
    return obstacles


def _generate_indoor(config: ObstacleConfig, rng: np.random.Generator) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    xh, yh, zh = config.arena_size

    wall_thickness = 0.05
    wall_height = zh
    walls = [
        Obstacle("box", np.array([xh, 0, wall_height / 2]),
                 np.array([wall_thickness, yh, wall_height / 2]),
                 np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([-xh, 0, wall_height / 2]),
                 np.array([wall_thickness, yh, wall_height / 2]),
                 np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([0, yh, wall_height / 2]),
                 np.array([xh, wall_thickness, wall_height / 2]),
                 np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([0, -yh, wall_height / 2]),
                 np.array([xh, wall_thickness, wall_height / 2]),
                 np.array([0.8, 0.8, 0.75, 1.0])),
        Obstacle("box", np.array([0, 0, zh]),
                 np.array([xh, yh, wall_thickness]),
                 np.array([0.9, 0.9, 0.85, 1.0])),
    ]
    obstacles.extend(walls)

    placed = [w.position for w in walls]
    n_interior = config.num_obstacles - len(walls)
    attempts = 0
    while len(obstacles) - len(walls) < n_interior and attempts < max(1, n_interior) * 20:
        attempts += 1
        x = rng.uniform(-xh * 0.8, xh * 0.8)
        y = rng.uniform(-yh * 0.8, yh * 0.8)
        kind = rng.choice(["table", "column", "shelf"])

        if kind == "table":
            sx, sy, sz = rng.uniform(0.2, 0.5), rng.uniform(0.2, 0.5), rng.uniform(0.3, 0.8)
            pos = np.array([x, y, sz / 2])
            geom = "box"
            size = np.array([sx, sy, sz / 2])
            rgba = np.array([0.6, 0.4, 0.2, 1.0])
        elif kind == "column":
            r = rng.uniform(0.05, 0.15)
            pos = np.array([x, y, zh / 2])
            geom = "cylinder"
            size = np.array([r, zh / 2])
            rgba = np.array([0.7, 0.7, 0.7, 1.0])
        else:
            sx = rng.uniform(0.1, 0.3)
            pos = np.array([x, y, rng.uniform(0.5, zh * 0.8)])
            geom = "box"
            size = np.array([sx, 0.05, 0.15])
            rgba = np.array([0.5, 0.3, 0.1, 1.0])

        if _try_place(pos, placed, config):
            obstacles.append(Obstacle(geom, pos, size, rgba))
            placed.append(pos)

    return obstacles


def _generate_random(config: ObstacleConfig, rng: np.random.Generator) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    placed: list[np.ndarray] = []
    xh, yh, zh = config.arena_size
    attempts = 0
    while len(obstacles) < config.num_obstacles and attempts < config.num_obstacles * 20:
        attempts += 1
        x = rng.uniform(-xh, xh)
        y = rng.uniform(-yh, yh)
        z = rng.uniform(0.1, zh)
        pos = np.array([x, y, z])
        if not _try_place(pos, placed, config):
            continue

        geom = rng.choice(["box", "cylinder", "sphere"])
        if geom == "box":
            size = rng.uniform(0.05, 0.25, size=3)
        elif geom == "cylinder":
            size = np.array([rng.uniform(0.03, 0.15), rng.uniform(0.1, 0.4)])
        else:
            size = np.array([rng.uniform(0.05, 0.2)])

        rgba = np.concatenate([rng.uniform(0.2, 0.9, size=3), [1.0]])
        euler = rng.uniform(-0.5, 0.5, size=3)
        obstacles.append(Obstacle(geom, pos, size, rgba, euler))
        placed.append(pos)
    return obstacles


def _generate_gates(config: ObstacleConfig, rng: np.random.Generator) -> list[Obstacle]:
    obstacles: list[Obstacle] = []
    n_gates = min(config.num_obstacles, 10)
    xh, yh, _ = config.arena_size

    for i in range(n_gates):
        angle = 2 * np.pi * i / n_gates
        radius = min(xh, yh) * 0.6
        cx = radius * np.cos(angle)
        cy = radius * np.sin(angle)
        gate_h = rng.uniform(0.8, 1.5)
        gate_w = rng.uniform(0.4, 0.8)
        thickness = 0.03

        yaw = angle + np.pi / 2
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        # Left pillar
        lx = cx + gate_w / 2 * cos_y
        ly = cy + gate_w / 2 * sin_y
        obstacles.append(Obstacle(
            "box", np.array([lx, ly, gate_h / 2]),
            np.array([thickness, thickness, gate_h / 2]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))
        # Right pillar
        rx = cx - gate_w / 2 * cos_y
        ry = cy - gate_w / 2 * sin_y
        obstacles.append(Obstacle(
            "box", np.array([rx, ry, gate_h / 2]),
            np.array([thickness, thickness, gate_h / 2]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))
        # Top bar
        obstacles.append(Obstacle(
            "box", np.array([cx, cy, gate_h]),
            np.array([thickness, gate_w / 2, thickness]),
            np.array([1.0, 0.3, 0.1, 1.0]),
            np.array([0, 0, yaw]),
        ))

    return obstacles


# ─────────────────────────────────────────────────────────────────────────────
# MjSpec attachment
# ─────────────────────────────────────────────────────────────────────────────

_GEOM_TYPE_MAP = {
    "box":      mj.mjtGeom.mjGEOM_BOX,
    "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
    "capsule":  mj.mjtGeom.mjGEOM_CAPSULE,
    "sphere":   mj.mjtGeom.mjGEOM_SPHERE,
}


def _euler_xyz_to_quat(euler: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler (roll-pitch-yaw) → [w, x, y, z] quaternion."""
    r, p, y = float(euler[0]), float(euler[1]), float(euler[2])
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def attach_obstacles(
    spec: mj.MjSpec,
    obstacles: list[Obstacle],
    *,
    name_prefix: str = "obstacle",
    collidable: bool = True,
) -> int:
    """Attach a list of obstacles to ``spec.worldbody`` in place.

    Each obstacle becomes one body containing one geom, mirroring the gate
    marker pattern in ``25_eval_rl_hex_mtrl.py``.

    Parameters
    ----------
    spec : mj.MjSpec
        World spec to mutate. Must not be already compiled.
    obstacles : list[Obstacle]
    name_prefix : str
        Bodies and geoms are named ``{prefix}_body_{i}`` / ``{prefix}_geom_{i}``.
    collidable : bool
        ``True`` → ``contype=1, conaffinity=1`` (default). ``False`` → both
        zero, useful for visual-only markers.

    Returns
    -------
    int
        Number of obstacles attached.
    """
    contype = 1 if collidable else 0
    conaffinity = 1 if collidable else 0

    for i, obs in enumerate(obstacles):
        geom_type = _GEOM_TYPE_MAP.get(obs.geom_type)
        if geom_type is None:
            raise ValueError(
                f"Unsupported geom_type {obs.geom_type!r}. "
                f"Expected one of {sorted(_GEOM_TYPE_MAP.keys())}."
            )

        pos = np.asarray(obs.position, dtype=np.float64).tolist()
        quat = _euler_xyz_to_quat(np.asarray(obs.euler, dtype=np.float64)).tolist()

        # MuJoCo size arrays must be length 3 even when only the first 1-2
        # entries are read for the chosen geom type.
        raw_size = np.asarray(obs.size, dtype=np.float64).reshape(-1)
        size = np.zeros(3, dtype=np.float64)
        size[:len(raw_size)] = raw_size

        rgba = np.asarray(obs.rgba, dtype=np.float64).reshape(-1)
        if rgba.size != 4:
            raise ValueError(f"Obstacle.rgba must be length 4, got {rgba.size}")

        body = spec.worldbody.add_body(
            name=f"{name_prefix}_body_{i}",
            pos=pos,
            quat=quat,
        )
        body.add_geom(
            name=f"{name_prefix}_geom_{i}",
            type=geom_type,
            size=size.tolist(),
            rgba=rgba.tolist(),
            contype=contype,
            conaffinity=conaffinity,
        )

    return len(obstacles)
