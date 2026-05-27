"""Export a DroneBlueprint to USD and visualise it with trimesh.

Pipeline:

    spherical_angular genome
      → spherical_angular_to_blueprint  (decoder)
      → DroneBlueprint                  (saved as JSON)
      → blueprint_to_usd                (USD ASCII backend)
      → .usda file                      (loadable by Isaac Lab)
      → trimesh 3-D scene               (interactive viewer + PNG)

The same blueprint that drives MuJoCo (example 13) is exported here as a
plain-text .usda file — no pxr / Omniverse install required. The trimesh
viewer shows the geometry the USD describes so you can inspect the structure
before loading it in Isaac Lab.

Run:
    uv run examples/d_drones/17_visualize_usd_blueprint.py
    uv run examples/d_drones/17_visualize_usd_blueprint.py --arm-len 0.15 --prop-size 6
    uv run examples/d_drones/17_visualize_usd_blueprint.py --no-show   # headless, save PNG only
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
import trimesh.transformations as tf
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ariel.body_phenotypes.drone.blueprint import (
    DroneBlueprint, ArmNode, MotorNode, RotorNode, CorePlateNode,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.backends import blueprint_to_usd, _rpy_to_R

# ---------------------------------------------------------------------------
# CLI

parser = argparse.ArgumentParser(description="Export drone blueprint to USD and visualise")
parser.add_argument("--arm-len", type=float, default=0.12,
                    help="Arm length in metres (default 0.12)")
parser.add_argument("--prop-size", type=int, default=5,
                    help="Propeller size in inches (default 5)")
parser.add_argument("--arms", type=int, default=4,
                    help="Number of arms (default 4)")
parser.add_argument("--arm-radius", type=float, default=0.005)
parser.add_argument("--motor-radius", type=float, default=0.015)
parser.add_argument("--motor-thickness", type=float, default=0.008)
parser.add_argument("--no-show", action="store_true",
                    help="Save figures without opening interactive windows")
parser.add_argument("--out", default="__data__/usd_demo",
                    help="Output directory (default __data__/usd_demo)")
args = parser.parse_args()

out_dir = Path(args.out)
out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Build blueprint — evenly-spaced X-configuration

n = args.arms
genome = np.array([
    [args.arm_len, math.radians(45 + 90 * i), 0.0, 0.0, 0.0, i % 2]
    for i in range(n)
])
bp = spherical_angular_to_blueprint(genome)
bp.save_json(out_dir / "blueprint.json")

print("=" * 70)
print(bp.summary())

# ---------------------------------------------------------------------------
# USD export

usda_path = blueprint_to_usd(
    bp,
    str(out_dir / "drone.usda"),
    arm_radius=args.arm_radius,
    motor_radius=args.motor_radius,
    motor_thickness=args.motor_thickness,
)
print(f"\nUSDA written → {usda_path}")
print(f"  ({Path(usda_path).stat().st_size} bytes)")

# ---------------------------------------------------------------------------
# Build trimesh scene (same geometry as the USD)

def _T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

ARM_RADIUS   = args.arm_radius
MOTOR_RADIUS = args.motor_radius
MOTOR_H      = 2.0 * args.motor_thickness

scene = trimesh.Scene()

core = bp.payload(bp.root_id)
core_mesh = trimesh.creation.cylinder(radius=core.radius, height=core.thickness, sections=48)
core_mesh.visual.face_colors = [51, 102, 204, 255]
scene.add_geometry(core_mesh, node_name="core",
                   geom_name="core_geom", transform=np.eye(4))

motor_index = 0
for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
    arm = bp.payload(arm_id)
    if not isinstance(arm, ArmNode):
        continue

    R_arm = _rpy_to_R(*arm.pose.rpy)
    arm_origin = np.array(arm.pose.xyz)

    # Capsule/cylinder along arm's local +X, centred at midpoint in world frame
    # trimesh cylinder is along Z; rotate Z→X with Ry(+90°)
    R_Yp90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    arm_mid_world = arm_origin + R_arm @ np.array([arm.length / 2.0, 0.0, 0.0])
    T_arm = _T(R_arm @ R_Yp90, arm_mid_world)
    arm_mesh = trimesh.creation.cylinder(radius=ARM_RADIUS, height=arm.length,
                                         sections=16, transform=T_arm)
    arm_mesh.visual.face_colors = [76, 76, 76, 255]
    scene.add_geometry(arm_mesh, node_name=f"arm_{arm_id}",
                       geom_name=f"arm_{arm_id}_geom")

    for motor_id in bp.children(arm_id):
        motor = bp.payload(motor_id)
        if not isinstance(motor, MotorNode):
            continue

        motor_world = arm_origin + R_arm @ np.array(motor.pose.xyz)
        R_motor = R_arm @ _rpy_to_R(*motor.pose.rpy)
        T_motor = _T(R_motor, motor_world)

        motor_mesh = trimesh.creation.cylinder(radius=MOTOR_RADIUS, height=MOTOR_H,
                                               sections=24, transform=T_motor)
        rgba = [255, 51, 51, 255] if motor.spin == "cw" else [51, 204, 51, 255]
        motor_mesh.visual.face_colors = rgba
        scene.add_geometry(motor_mesh, node_name=f"motor_{motor_index}",
                           geom_name=f"motor_{motor_index}_geom")

        for rotor_id in bp.children(motor_id):
            rotor = bp.payload(rotor_id)
            if not isinstance(rotor, RotorNode):
                continue

            rotor_z = args.motor_thickness + 0.001
            rotor_world = motor_world + R_motor @ np.array([0.0, 0.0, rotor_z])
            T_rotor = _T(R_motor, rotor_world)
            rotor_mesh = trimesh.creation.cylinder(radius=rotor.radius, height=0.002,
                                                   sections=48, transform=T_rotor)
            rotor_mesh.visual.face_colors = [200, 200, 200, 130]
            scene.add_geometry(rotor_mesh, node_name=f"rotor_{motor_index}",
                               geom_name=f"rotor_{motor_index}_geom")

        motor_index += 1

# ---------------------------------------------------------------------------
# Save matplotlib 3D figure (always, even in headless mode)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
ax.set_title(f"DroneBlueprint — {n}-arm drone  (arm={args.arm_len:.3f} m, "
             f"prop={args.prop_size}\")")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

for geom in scene.geometry.values():
    verts = geom.vertices
    faces = geom.faces
    fc = geom.visual.face_colors
    if fc is not None and len(fc):
        color = tuple(c / 255.0 for c in fc[0])
    else:
        color = (0.5, 0.5, 0.5, 1.0)
    poly = Poly3DCollection(verts[faces], alpha=float(color[3]),
                            facecolor=color[:3], edgecolor="none")
    ax.add_collection3d(poly)

# Fit axes to combined bounds
bounds = scene.bounds
if bounds is not None:
    pad = 0.05
    ax.set_xlim(bounds[0, 0] - pad, bounds[1, 0] + pad)
    ax.set_ylim(bounds[0, 1] - pad, bounds[1, 1] + pad)
    ax.set_zlim(bounds[0, 2] - pad, bounds[1, 2] + pad)

ax.view_init(elev=30, azim=45)
plt.tight_layout()
png_path = out_dir / "drone_usd_preview.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight")
print(f"PNG saved        → {png_path}")

if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Interactive trimesh viewer (skipped in headless mode)

if not args.no_show:
    print("\nOpening trimesh viewer — rotate with left-click, zoom with scroll …")
    scene.show(caption="DroneBlueprint USD preview")

print("\nDone.")
print(f"  Load in Isaac Lab:  UsdFileCfg(usd_path='{usda_path}')")
