#!/usr/bin/env python3
"""Convert an airevolve Spherical Angular genome into a URDF drone.

Genome format (airevolve, SphericalAngularDroneGenomeHandler): a (narms, 6) numpy array,
one row per rotor:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
Unused rows (narms < max_narms) are NaN-padded.

Per-arm kinematic chain (arm pointing along its local +X):
    base_link -[shoulder, Z, at plate edge]-> inner_arm -[elbow, Y, at ELBOW_FRACTION]->
    outer_arm -[motor tilt, X]-> motor -[rotor spin, Z]-> rotor
"""
import argparse
import math
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Hand-picked 4-rotor genome, used when no genome file is given.
SAMPLE_GENOME = np.array([
    # magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction
    [0.25, 0.0,          0.0, 0.0, 0.0, 0],
    [0.15, math.pi / 2,  0.0, 0.0, 0.0, 1],
    [0.15, math.pi,      0.0, 0.0, 0.0, 0],
    [0.15, -math.pi / 2, 0.0, 0.0, 0.0, 1],
])

CORE_MASS, CORE_RADIUS, CORE_THICKNESS = 0.4, 0.05, 0.01
ARM_RADIUS, ARM_MASS_PER_M = 0.008, 0.18
MOTOR_RADIUS, MOTOR_HEIGHT, MOTOR_MASS = 0.014, 0.02, 0.06
ROTOR_RADIUS, ROTOR_THICKNESS = 0.0635, 0.002
ROTOR_HUB_RADIUS, ROTOR_HUB_HEIGHT, ROTOR_HUB_MASS = 0.01, 0.006, 0.01
ROTOR_MOUNT_Z = 0.016
ELBOW_FRACTION = 0.5

# (axis, limit, effort, velocity, damping); limit=None means continuous.
SHOULDER = ((0, 0, 1), 0.5, 8.0, 3.0, 0.08)
ELBOW = ((0, 1, 0), 1.0, 6.0, 3.0, 0.08)
MOTOR_TILT = ((1, 0, 0), 0.30, 6.0, 3.0, 0.05)
ROTOR_SPIN = ((0, 0, 1), None, 2.5, 400.0, 0.001)

MATERIALS = """\
  <material name="mat_core"><color rgba="0.15 0.15 0.18 1.0"/></material>
  <material name="mat_arm"><color rgba="0.22 0.22 0.24 1.0"/></material>
  <material name="mat_motor"><color rgba="0.70 0.70 0.72 1.0"/></material>
  <material name="mat_rotor_ccw"><color rgba="0.10 0.10 0.10 0.85"/></material>
  <material name="mat_rotor_cw"><color rgba="0.05 0.25 0.55 0.85"/></material>"""


def vec(v) -> str:
    return " ".join(f"{float(x):.6f}" for x in v)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def thrust_direction(motor_rotation: float, motor_pitch: float) -> np.ndarray:
    """Unit thrust-axis direction in base_link frame from the genome's motor angles."""
    return np.array([
        math.cos(motor_rotation) * math.sin(motor_pitch),
        math.sin(motor_rotation) * math.sin(motor_pitch),
        math.cos(motor_pitch),
    ])


def link(name, radius, length, material, mass, xyz=(0, 0, 0), rpy=(0, 0, 0), inertia_radius=None, inertia_length=None):
    """Link with a cylinder visual/collision and a Z-axis solid-cylinder inertia at the link origin."""
    r, h = inertia_radius or radius, inertia_length or length
    ixx = mass * (3 * r * r + h * h) / 12.0
    izz = 0.5 * mass * r * r
    origin = f'<origin xyz="{vec(xyz)}" rpy="{vec(rpy)}"/>'
    geometry = f'<geometry><cylinder radius="{radius:.6f}" length="{length:.6f}"/></geometry>'
    return f"""\
  <link name="{name}">
    <visual>{origin}{geometry}<material name="{material}"/></visual>
    <collision>{origin}{geometry}</collision>
    <inertial><origin xyz="0 0 0" rpy="0 0 0"/><mass value="{mass:.6g}"/><inertia ixx="{ixx:.6g}" ixy="0" ixz="0" iyy="{ixx:.6g}" iyz="0" izz="{izz:.6g}"/></inertial>
  </link>"""


def joint(name, parent, child, xyz, rpy, spec):
    axis, limit, effort, velocity, damping = spec
    jtype = "continuous" if limit is None else "revolute"
    bounds = "" if limit is None else f'lower="{-limit:.6f}" upper="{limit:.6f}" '
    return f"""\
  <joint name="{name}" type="{jtype}">
    <parent link="{parent}"/>
    <child link="{child}"/>
    <origin xyz="{vec(xyz)}" rpy="{vec(rpy)}"/>
    <axis xyz="{vec(axis)}"/>
    <dynamics damping="{damping:.4f}" friction="0.0000"/>
    <limit {bounds}effort="{effort:.4f}" velocity="{velocity:.4f}"/>
  </joint>"""


def arm(name: str, row: np.ndarray) -> list[str]:
    magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction = row
    span = magnitude - CORE_RADIUS
    if span <= 0.0:
        raise ValueError(f"{name}: magnitude {magnitude} must exceed core radius {CORE_RADIUS}")
    inner_len = ELBOW_FRACTION * span
    outer_len = span - inner_len

    # Arm frame: yaw arm_rotation, elevation arm_pitch (0 = horizontal, +pi/2 = straight up).
    # Deliberately NOT airevolve's arm_conversions.spherical_to_cartesian (polar-from-Z convention),
    # which is inconsistent with how arm_pitch is used everywhere else in airevolve.
    r_arm = rot_z(arm_rotation) @ rot_y(-arm_pitch)
    shoulder_xyz = r_arm[:, 0] * CORE_RADIUS

    # The shoulder (Z), elbow (Y) and motor tilt (X) hinges rest at zero and can't reach an
    # arbitrary thrust direction, so the residual is folded into the rotor joint's static rpy
    # (spin phase about the resolved Z axis is physically irrelevant).
    lx, ly, lz = r_arm.T @ thrust_direction(motor_rotation, motor_pitch)
    rotor_rpy = (0.0, math.acos(max(-1.0, min(1.0, lz))), math.atan2(ly, lx))
    spin = "cw" if direction >= 0.5 else "ccw"

    arm_rpy = (0.0, math.pi / 2, 0.0)
    return [
        link(f"{name}_inner_arm_link", ARM_RADIUS, inner_len, "mat_arm", inner_len * ARM_MASS_PER_M, (inner_len / 2, 0, 0), arm_rpy),
        joint(f"base_to_{name}_shoulder_joint", "base_link", f"{name}_inner_arm_link", shoulder_xyz, (0.0, -arm_pitch, arm_rotation), SHOULDER),
        link(f"{name}_outer_arm_link", ARM_RADIUS, outer_len, "mat_arm", outer_len * ARM_MASS_PER_M, (outer_len / 2, 0, 0), arm_rpy),
        joint(f"{name}_elbow_joint", f"{name}_inner_arm_link", f"{name}_outer_arm_link", (inner_len, 0, 0), (0, 0, 0), ELBOW),
        link(f"{name}_motor_link", MOTOR_RADIUS, MOTOR_HEIGHT, "mat_motor", MOTOR_MASS),
        joint(f"{name}_arm_to_motor_joint", f"{name}_outer_arm_link", f"{name}_motor_link", (outer_len, 0, 0), (0, 0, 0), MOTOR_TILT),
        link(f"{name}_rotor_link", ROTOR_RADIUS, ROTOR_THICKNESS, f"mat_rotor_{spin}", ROTOR_HUB_MASS,
             inertia_radius=ROTOR_HUB_RADIUS, inertia_length=ROTOR_HUB_HEIGHT),
        joint(f"{name}_motor_to_rotor_joint", f"{name}_motor_link", f"{name}_rotor_link", (0, 0, ROTOR_MOUNT_Z), rotor_rpy, ROTOR_SPIN),
    ]


def genome_to_urdf(genome: np.ndarray, name: str) -> str:
    rows = genome[~np.isnan(genome[:, 0])]
    if len(rows) == 0:
        raise ValueError("Genome has no valid (non-NaN) rotor rows")
    parts = [
        '<?xml version="1.0"?>',
        f'<robot name="{name}">',
        MATERIALS,
        link("base_link", CORE_RADIUS, CORE_THICKNESS, "mat_core", CORE_MASS),
    ]
    for i, row in enumerate(rows):
        parts += arm(f"arm_{i + 1}", row)
    parts.append("</robot>\n")
    return "\n".join(parts)


def write_urdf(genome: np.ndarray, name: str, out_dir: Path = OUTPUT_DIR / "urdf") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.urdf"
    path.write_text(genome_to_urdf(genome, name), encoding="utf-8")
    return path


def load_genome(path: Path | None) -> tuple[np.ndarray, str]:
    """Genome from a .npy file, or the built-in sample; returns (genome, default name)."""
    return (np.load(path), path.stem) if path else (SAMPLE_GENOME, "sample")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--genome", type=Path, help="Genome .npy file (default: built-in sample).")
    parser.add_argument("--name", type=str, help="Robot name (default: genome file stem).")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "urdf")
    args = parser.parse_args()
    genome, name = load_genome(args.genome)
    print(f"Wrote URDF: {write_urdf(genome, args.name or name, args.out_dir)}")
