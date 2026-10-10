"""Genome → DroneBlueprint decoders.

Each decoder takes a genome (or genome handler) and emits a DroneBlueprint.
This is the architectural seam the consortium plugs into: a new partner
adds a decoder; the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import math
import numpy as np

from .blueprint import (
    DroneBlueprint,
    CorePlateNode,
    ArmNode,
    MotorNode,
    RotorNode,
    Pose,
)


# ---------- decoder 1: spherical-angular ----------

def spherical_angular_to_blueprint(
    genome: np.ndarray,
    *,
    core_mass: float = 0.4,
    core_radius: float = 0.05,
    core_thickness: float = 0.01,
    propsize: int = 5,
    rotor_radius: float = 0.0635,
) -> DroneBlueprint:
    """Decode airevolve's spherical-angular genome → DroneBlueprint.

    Genome layout (per row): [magnitude, arm_az, arm_pitch, motor_az, motor_pitch, direction]
    - magnitude  : distance from core to motor (sim units; treated as metres here)
    - arm_az     : azimuth of arm in XY plane (rad)
    - arm_pitch  : elevation above XY plane (rad), 0 = horizontal
    - motor_az   : motor thrust-vector azimuth in world XY (rad)
    - motor_pitch: motor thrust-vector pitch (rad), 0 = thrust along +Z
    - direction  : 0=CCW, 1=CW
    NaN rows are inactive arm slots.
    """
    bp = DroneBlueprint()
    core_id = bp.add(CorePlateNode(
        mass=core_mass, radius=core_radius, thickness=core_thickness
    ))

    valid = ~np.isnan(genome[:, 0])
    for row in genome[valid]:
        mag, arm_az, arm_pitch, motor_az, motor_pitch, direction = (float(x) for x in row)

        # Arm pose: position the arm's attachment frame at the core's surface
        # along (arm_az, arm_pitch); the arm itself extends along its local +X.
        arm_pose = Pose(
            xyz=(0.0, 0.0, 0.0),                 # mounted at core centre frame
            rpy=(0.0, -arm_pitch, arm_az),       # yaw to azimuth, pitch to elevation
        )
        arm_id = bp.add(ArmNode(length=mag, pose=arm_pose), parent=core_id)

        # Motor pose: at arm tip (local +X by `length`), with thrust direction
        # encoded as rpy that — when composed with the arm's frame — yields the
        # genome's world-frame (motor_az, motor_pitch). For the demo we store
        # the relative offset literally; the backend collapses the chain.
        motor_local_rpy = (0.0, motor_pitch - arm_pitch, motor_az - arm_az)
        motor_pose = Pose(xyz=(mag, 0.0, 0.0), rpy=motor_local_rpy)
        spin = "cw" if direction >= 0.5 else "ccw"
        motor_id = bp.add(
            MotorNode(pose=motor_pose, spin=spin, propsize=propsize),
            parent=arm_id,
        )

        bp.add(RotorNode(radius=rotor_radius), parent=motor_id)

    return bp


# ---------- decoder 2: cartesian-Euler (sketched, shows portability) ----------

def cartesian_euler_to_blueprint(
    genome: np.ndarray,
    *,
    core_mass: float = 0.4,
    propsize: int = 5,
) -> DroneBlueprint:
    """Decode a Cartesian-Euler genome → DroneBlueprint.

    Genome layout (per row): [x, y, z, roll, pitch, yaw, direction]
    Each row places a motor directly at a Cartesian offset from the core,
    with an Euler-angle thrust orientation. Demonstrates that an entirely
    different encoding produces the same downstream blueprint shape.
    """
    bp = DroneBlueprint()
    core_id = bp.add(CorePlateNode(mass=core_mass))

    valid = ~np.isnan(genome[:, 0])
    for row in genome[valid]:
        x, y, z, roll, pitch, yaw, direction = (float(v) for v in row)
        arm_length = math.sqrt(x * x + y * y + z * z)
        arm_az = math.atan2(y, x)
        arm_el = math.atan2(z, math.sqrt(x * x + y * y))

        arm_pose = Pose(xyz=(0.0, 0.0, 0.0), rpy=(0.0, -arm_el, arm_az))
        arm_id = bp.add(ArmNode(length=arm_length, pose=arm_pose), parent=core_id)

        motor_pose = Pose(xyz=(arm_length, 0.0, 0.0), rpy=(roll, pitch, yaw))
        spin = "cw" if direction >= 0.5 else "ccw"
        motor_id = bp.add(
            MotorNode(pose=motor_pose, spin=spin, propsize=propsize),
            parent=arm_id,
        )
        bp.add(RotorNode(), parent=motor_id)

    return bp
