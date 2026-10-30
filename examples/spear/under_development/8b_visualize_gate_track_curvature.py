"""Visualise a quintic gate track in MuJoCo using the curvature-based generator.

Identical to 8_visualize_gate_track.py but uses path_curvature_based.py
(PyTorch, equidistant resampling + loop rejection) instead of planner_generator.py.

Two modes
---------
1. Load saved gate files from a previous run (--gate-pos / --gate-yaw):

    uv run examples/spear/8b_visualize_gate_track_curvature.py \\
        --gate-pos __data__/drone_evo_rl_quintic/RUN_ID/gate_pos_RUN_ID.npy \\
        --gate-yaw __data__/drone_evo_rl_quintic/RUN_ID/gate_yaw_RUN_ID.npy

   Or point at the run directory and let it auto-detect:

    uv run examples/spear/8b_visualize_gate_track_curvature.py \\
        --run-dir __data__/drone_evo_rl_quintic/RUN_ID

2. Generate a fresh track from the quintic coefficients (default):

    uv run examples/spear/8b_visualize_gate_track_curvature.py
    uv run examples/spear/8b_visualize_gate_track_curvature.py --seed 42
    uv run examples/spear/8b_visualize_gate_track_curvature.py \\
        --path-steps 8 --path-scale 3.0 --dense-steps 500 --seed 7
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from path_curvature_based import generate_paths_from_coefficients  # noqa: E402

parser = argparse.ArgumentParser(description="Visualise a quintic gate track in MuJoCo (curvature-based generator)")
# ── saved-run inputs (takes priority over generation args) ────────────────────
parser.add_argument("--run-dir",  default=None,
                    help="Run directory written by 7_drone_evo_rl_quintic.py; "
                         "auto-detects gate_pos_*.npy and gate_yaw_*.npy inside.")
parser.add_argument("--gate-pos", default=None,
                    help="Explicit path to gate_pos_*.npy (N×3, NED coords).")
parser.add_argument("--gate-yaw", default=None,
                    help="Explicit path to gate_yaw_*.npy (N,).")
# ── generation args (used when no saved files are provided) ───────────────────
parser.add_argument("--seed",        type=int,   default=None)
parser.add_argument("--path-steps",  type=int,   default=20,
                    help="Number of gates sampled from the dense path (default 20)")
parser.add_argument("--dense-steps", type=int,   default=1000,
                    help="Points in the dense path before subsampling (default 1000)")
parser.add_argument("--path-scale",  type=float, default=5.0,
                    help="Scale applied to the clipped [-1,1] path coords (default 5.0 → ±5 m)")
parser.add_argument("--z-height",    type=float, default=-1.5,
                    help="Gate altitude in NED z (default -1.5 → 1.5 m AGL)")
parser.add_argument("--gate-size",   type=float, default=1.5,
                    help="Gate opening side length in metres (default 1.5)")
parser.add_argument("--coeffs",
                    default=str(_REPO_ROOT / "goal_generator_ltu"
                                / "polynomial_goal_generator" / "quintic_coeffs.npy"))
args = parser.parse_args()

# ── resolve gate data ─────────────────────────────────────────────────────────
def _latest(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[-1] if matches else None

_gpos_path = Path(args.gate_pos) if args.gate_pos else (
    _latest(Path(args.run_dir), "gate_pos_*.npy") if args.run_dir else None
)
_gyaw_path = Path(args.gate_yaw) if args.gate_yaw else (
    _latest(Path(args.run_dir), "gate_yaw_*.npy") if args.run_dir else None
)

if _gpos_path is not None and _gyaw_path is not None:
    # ── mode 1: load from saved run files ─────────────────────────────────────
    gate_pos  = np.load(_gpos_path).astype(np.float32)   # (N, 3) NED
    gate_yaws = np.load(_gyaw_path).astype(np.float32)   # (N,)
    print(f"Loaded {len(gate_pos)} gates from saved run:")
    print(f"  gate_pos : {_gpos_path}")
    print(f"  gate_yaw : {_gyaw_path}")
else:
    # ── mode 2: generate fresh from quintic coefficients ──────────────────────
    rng  = np.random.default_rng(args.seed)
    seed = int(rng.integers(0, 2**31))
    print(f"Generating fresh track  seed={seed}")

    coefficients = torch.from_numpy(np.load(args.coeffs)).to(torch.float64)
    paths, yaws = generate_paths_from_coefficients(
        coefficients=coefficients,
        num_generate=1,
        steps=args.dense_steps,
        seed=seed,
        clip_range=(-1.0, 1.0),
    )
    xy_dense  = paths[0].numpy() * args.path_scale   # (dense_steps, 2)
    yaw_dense = yaws[0].numpy()                       # (dense_steps,) — tangent angles

    indices   = np.linspace(0, args.dense_steps - 1, args.path_steps, dtype=int)
    xy_gates  = xy_dense[indices]
    gate_yaws = yaw_dense[indices].astype(np.float32)
    gate_pos  = np.column_stack([
        xy_gates[:, 0],
        xy_gates[:, 1],
        np.full(args.path_steps, args.z_height),
    ]).astype(np.float32)

start_pos = (gate_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)

n_gates = len(gate_pos)
print(f"\nGates ({n_gates})  [NED coords, yaw = direction of travel]:")
for i, (pos, yaw) in enumerate(zip(gate_pos, gate_yaws)):
    print(f"  [{i}]  x={pos[0]:+.2f}  y={pos[1]:+.2f}  z={pos[2]:+.2f}"
          f"  yaw={math.degrees(yaw):+.1f}°")

dists = np.linalg.norm(np.diff(gate_pos[:, :2], axis=0), axis=1)
print(f"\nGate-to-gate distances: {dists.round(2)} m  (mean {dists.mean():.2f} m)")

# ── build MuJoCo scene ────────────────────────────────────────────────────────
import mujoco
import mujoco.viewer
from ariel.simulation.environments import SimpleFlatWorld

world = SimpleFlatWorld()
half  = args.gate_size / 2.0
depth = 0.02

for i, (pos, yaw) in enumerate(zip(gate_pos, gate_yaws)):
    gx = float(pos[0])
    gy = float(pos[1])
    gz = float(-pos[2])   # NED z → ENU z (up)

    # Yaw quaternion: rotation by `yaw` around world Z  →  (w, x, y, z)
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)

    body = world.spec.worldbody.add_body(
        name=f"gate_{i}",
        pos=[gx, gy, gz],
        quat=[qw, 0.0, 0.0, qz],
    )

    # size = [depth/2, half, half]:  X thin (depth),  Y wide (gate width),  Z tall (gate height)
    body.add_geom(
        name=f"gate_plane_{i}",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[depth, half, half],
        rgba=(1.0, 0.55, 0.0, 0.3),
        contype=0,
        conaffinity=0,
    )
    # Red centre sphere.
    body.add_geom(
        name=f"gate_marker_{i}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.07, 0.0, 0.0],
        rgba=(1.0, 0.1, 0.1, 1.0),
        contype=0,
        conaffinity=0,
    )

    # Green cylinder placed 0.5 m BEHIND the gate pointing in the travel direction.
    ax = float(-math.cos(yaw))
    ay = float(-math.sin(yaw))
    arr_body = world.spec.worldbody.add_body(
        name=f"gate_arrow_{i}",
        pos=[gx + ax * 0.55, gy + ay * 0.55, gz],
        quat=[qw, 0.0, 0.0, qz],
    )
    arr_body.add_geom(
        name=f"gate_arrow_geom_{i}",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.035, 0.22, 0.0],
        euler=[0.0, math.pi / 2, 0.0],
        rgba=(0.2, 0.85, 0.2, 0.9),
        contype=0,
        conaffinity=0,
    )

    # White sphere label above gate.
    lbl = world.spec.worldbody.add_body(
        name=f"gate_label_{i}", pos=[gx, gy, gz + half + 0.18])
    lbl.add_geom(
        name=f"gate_label_geom_{i}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.06, 0.0, 0.0],
        rgba=(1.0, 1.0, 1.0, 1.0),
        contype=0,
        conaffinity=0,
    )

# Blue start-position marker.
sb = world.spec.worldbody.add_body(
    name="start_pos",
    pos=[float(start_pos[0]), float(start_pos[1]), float(-start_pos[2])],
)
sb.add_geom(
    name="start_marker",
    type=mujoco.mjtGeom.mjGEOM_SPHERE,
    size=[0.13, 0.0, 0.0],
    rgba=(0.1, 0.4, 1.0, 1.0),
    contype=0,
    conaffinity=0,
)

# Grey midpoint dots along path.
for i in range(n_gates):
    j   = (i + 1) % n_gates
    mid = (gate_pos[i] + gate_pos[j]) / 2
    mb  = world.spec.worldbody.add_body(
        name=f"path_dot_{i}",
        pos=[float(mid[0]), float(mid[1]), float(-mid[2])],
    )
    mb.add_geom(
        name=f"path_dot_geom_{i}",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.03, 0.0, 0.0],
        rgba=(0.6, 0.6, 0.6, 0.4),
        contype=0,
        conaffinity=0,
    )

model = world.spec.compile()
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print("\nLaunching MuJoCo viewer …")
print("  Orange slabs  = gate planes  (drone flies through, facing you head-on when aligned)")
print("  Red spheres   = gate centres")
print("  Green arrows  = placed behind the gate, pointing in the approach direction")
print("  Blue sphere   = start position")
print("  White spheres = gate numbers (count from blue start)")
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth   = 135
    viewer.cam.elevation = -35
    viewer.cam.distance  = 14.0
    viewer.cam.lookat    = [
        float(gate_pos[:, 0].mean()),
        float(gate_pos[:, 1].mean()),
        float(-gate_pos[:, 2].mean()),
    ]
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
