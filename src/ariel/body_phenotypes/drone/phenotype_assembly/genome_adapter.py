"""
Genome → CAD Parameter Adapter

Converts a ``SphericalAngularDroneGenomeHandler`` genome into
``DroneCADParameters`` which can then be fed directly to the assembler.

Genome column layout
--------------------
Index  Name             Range         Units
  0    magnitude        [0.5, 2.0]    (simulation scale; multiply by length_scale → mm)
  1    arm_rotation     [0, 2π]       radians  – azimuthal angle θ in XY plane
  2    arm_pitch        [−π/2, π/2]   radians  – elevation angle above XY plane
                                                 (0 = horizontal, π/2 = straight up)
  3    motor_rotation   [0, 2π]       radians  – motor disc azimuth
  4    motor_pitch      [−π/2, π/2]   radians  – motor disc elevation angle
  5    direction        {0, 1}        –        – propeller spin (0=CCW, 1=CW)

Mapping to ``ArmCADParameters``
--------------------------------
attachment_angle = degrees(arm_rotation)   # azimuth directly
arm_elevation    = degrees(arm_pitch)      # elevation angle maps directly
                                           # 0   → arm lies in horizontal XY plane
                                           # π/2 → arm points straight up
                                           # −π/2→ arm points straight down
motor_tilt       = acos(n_local_z)         # local-frame tilt recovered by
motor_azimuth    = atan2(n_local_y,        # inverting the assembler's Ry/Rz
                         n_local_x)        # so motor_rotation/motor_pitch define
                                           # the disc normal in **world space**
arm_length       = magnitude * scale − attach_radius
                                           # subtract the attachment radial offset so
                                           # the motor tip lands at the same (x,y,z)
                                           # as the simulation motor position

snap_mounts
-----------
When ``snap_mounts=True`` the arm mount sphere is snapped to the nearest of
``num_mount_positions`` evenly-spaced discrete positions on the plate rim.
The motor tip stays at the same world-space position; the arm length,
elevation, and azimuth are recomputed from the vector between the snapped
mount centre and the motor tip.
"""

import math
import numpy as np
from typing import List, Optional, Tuple

from .models import ArmCADParameters, DroneCADParameters, AssemblyConfig


def _snap_angle(angle_rad: float, num_positions: int) -> float:
    """Round ``angle_rad`` to the nearest of ``num_positions`` evenly-spaced
    angles around the circle.  Returns the snapped angle in radians in [0, 2π)."""
    step = 2.0 * math.pi / num_positions
    return round(angle_rad / step) * step % (2.0 * math.pi)


def _assign_mount_slots(
    cont_angles_rad: List[float],
    motor_tips: List[Tuple[float, float, float]],
    num_positions: int,
    attach_radius: float,
    mount_z: float,
) -> List[float]:
    """Assign each arm to a unique discrete slot on the plate rim, minimising
    the total 3-D distance from each assigned slot centre to the arm's motor tip.

    When the number of arms is ≤ ``num_positions`` the assignment is globally
    optimal (Hungarian algorithm via ``scipy.optimize.linear_sum_assignment``).
    When there are more arms than slots a greedy nearest-neighbour with
    collision avoidance is used as a fallback.

    Args:
        cont_angles_rad: Continuous (genome) azimuth for each arm, in radians.
        motor_tips: World-space (x, y, z) of each arm's motor tip.
        num_positions: Number of equally-spaced discrete slots around the rim.
        attach_radius: Radial distance from drone centre to the slot centre.
        mount_z: Z coordinate of every slot centre (plate mid-plane).

    Returns:
        List of assigned slot angles, one per arm, in radians, in [0, 2π).
    """
    num_arms = len(cont_angles_rad)
    step = 2.0 * math.pi / num_positions
    slot_angles = [i * step for i in range(num_positions)]

    # Slot centre positions on the plate rim
    slot_positions = [
        (attach_radius * math.cos(a), attach_radius * math.sin(a), mount_z)
        for a in slot_angles
    ]

    if num_arms <= num_positions:
        # Build cost matrix: cost[arm_idx, slot_idx] = distance from slot to motor tip
        cost = np.zeros((num_arms, num_positions))
        for ai, (mx, my, mz) in enumerate(motor_tips):
            for si, (sx, sy, sz) in enumerate(slot_positions):
                cost[ai, si] = math.sqrt(
                    (mx - sx) ** 2 + (my - sy) ** 2 + (mz - sz) ** 2
                )

        try:
            from scipy.optimize import linear_sum_assignment
            _row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            # scipy not available — fall through to greedy
            col_ind = None

        if col_ind is not None:
            assigned = [slot_angles[col_ind[ai]] for ai in range(num_arms)]
            return assigned

    # Greedy fallback: process arms in order of how far their nearest free slot
    # is from their continuous angle (most-constrained first).
    available = list(range(num_positions))
    # Sort arms by distance to nearest available slot (ascending), so arms
    # that are close to their ideal slot get first pick.
    order = sorted(
        range(num_arms),
        key=lambda ai: min(
            abs(cont_angles_rad[ai] - slot_angles[si]) % (2.0 * math.pi)
            for si in available
        ),
    )
    assigned_slots = [None] * num_arms
    for ai in order:
        mx, my, mz = motor_tips[ai]
        best_si = min(
            available,
            key=lambda si: math.sqrt(
                (mx - slot_positions[si][0]) ** 2
                + (my - slot_positions[si][1]) ** 2
                + (mz - slot_positions[si][2]) ** 2
            ),
        )
        assigned_slots[ai] = slot_angles[best_si]
        available.remove(best_si)
        if not available:
            # Ran out of unique slots — reopen all (more arms than slots)
            available = list(range(num_positions))

    return [a % (2.0 * math.pi) for a in assigned_slots]


def genome_to_cad_parameters(
    genome_handler,
    magnitude_to_length_scale: float = 100.0,
    distribute_arms_evenly: bool = False,
    assembly_config: Optional[AssemblyConfig] = None,
    snap_mounts: bool = False,
    num_mount_positions: int = 8,
) -> DroneCADParameters:
    """
    Convert a SphericalAngularDroneGenomeHandler to DroneCADParameters.

    The genome stores arm positions as spherical coordinates where ``arm_pitch``
    is an **elevation** angle (degrees above the XY plane), matching the
    convention used by ``utils.convert_to_cartesian`` in the visualizer.

    Because the physical arm attaches at the plate rim (not the drone centre),
    the arm tube length is reduced by the attachment radial offset so that the
    motor tip ends up at the same absolute position as the simulation motor.

    Args:
        genome_handler: A ``SphericalAngularDroneGenomeHandler`` instance with
            a valid genome loaded.
        magnitude_to_length_scale: Multiplier converting genome magnitude to
            the simulation-space motor distance from centre in mm.
            Default 100 mm / unit.
        distribute_arms_evenly: When *True*, ignore the genome's
            ``arm_rotation`` and instead space arms evenly around the plate
            (0°, 360/n°, …).  Useful for visualisation of symmetric designs
            but loses fidelity to the evolved genome.  Default *False*.
        assembly_config: Physical assembly dimensions used to compute the
            attachment radial offset.  Defaults to ``AssemblyConfig()``.
        snap_mounts: When *True*, snap each arm mount to the nearest of
            ``num_mount_positions`` discrete positions on the plate rim.
            The motor tip position is preserved; arm length, elevation, and
            azimuth are recomputed from the vector between the snapped mount
            and the original motor tip.  Default *False*.
        num_mount_positions: Number of discrete mount positions around the
            plate rim.  Only used when ``snap_mounts=True``.  Default 8
            (positions at 0°, 45°, 90°, …, 315°).

    Returns:
        ``DroneCADParameters`` containing one ``ArmCADParameters`` per active arm.
    """
    if assembly_config is None:
        assembly_config = AssemblyConfig()

    # Radial distance from the drone centre to where the arm tube starts
    # (the sphere centre on the plate rim).
    attach_radius = assembly_config.arm_attach_radial_distance
    # Z height of the sphere centre (plate mid-plane)
    mount_z = assembly_config.plate_thickness / 2.0
    # Sphere offset along the arm axis (from tube end to disc centre)
    sphere_offset = assembly_config.sphere_offset

    valid_arms = genome_handler.get_valid_arms()
    num_arms = len(valid_arms)

    if num_arms == 0:
        return DroneCADParameters(arms=[])

    # ── Pass 1: collect continuous angles and motor tip positions ────────────
    cont_angles_rad: List[float] = []
    motor_tips: List[Tuple[float, float, float]] = []

    for i, arm in enumerate(valid_arms):
        magnitude, arm_rotation, arm_pitch, _motor_rotation, _motor_pitch, _direction = arm

        if distribute_arms_evenly:
            cont_angle_rad = math.radians((360.0 / num_arms) * i)
        else:
            cont_angle_rad = float(arm_rotation) % (2.0 * math.pi)

        motor_distance = float(magnitude) * magnitude_to_length_scale
        pitch_rad = float(arm_pitch)
        motor_x = motor_distance * math.cos(pitch_rad) * math.cos(cont_angle_rad)
        motor_y = motor_distance * math.cos(pitch_rad) * math.sin(cont_angle_rad)
        motor_z = motor_distance * math.sin(pitch_rad)

        cont_angles_rad.append(cont_angle_rad)
        motor_tips.append((motor_x, motor_y, motor_z))

    # ── Snap mount angles: globally optimal unique-slot assignment ───────────
    if snap_mounts:
        snapped_rads = _assign_mount_slots(
            cont_angles_rad, motor_tips, num_mount_positions, attach_radius, mount_z
        )
    else:
        snapped_rads = list(cont_angles_rad)

    # ── Pass 2: build ArmCADParameters from the assigned mount angles ────────
    arms_params: List[ArmCADParameters] = []

    for i, arm in enumerate(valid_arms):
        _magnitude, _arm_rotation, _arm_pitch, motor_rotation, motor_pitch, direction = arm

        motor_x, motor_y, motor_z = motor_tips[i]
        snapped_rad = snapped_rads[i]

        mount_angle = float(np.degrees(snapped_rad)) % 360.0

        # ── Mount position on plate rim ──────────────────────────────────────
        mount_x = attach_radius * math.cos(snapped_rad)
        mount_y = attach_radius * math.sin(snapped_rad)

        # ── Vector from mount centre to motor tip ────────────────────────────
        vx = motor_x - mount_x
        vy = motor_y - mount_y
        vz = motor_z - mount_z

        # Arm tube azimuth: direction the arm points from the mount
        attachment_angle = float(np.degrees(math.atan2(vy, vx))) % 360.0

        # Arm elevation: angle above horizontal
        vxy = math.sqrt(vx * vx + vy * vy)
        arm_elevation = float(np.degrees(math.atan2(vz, vxy)))

        # Arm length: distance from mount sphere surface to motor tip.
        # Subtract sphere_offset because the motor disc sits sphere_offset mm
        # beyond the end of the arm tube (see create_motor_arm).
        arm_vector_length = math.sqrt(vx * vx + vy * vy + vz * vz)
        arm_length = max(arm_vector_length - sphere_offset, 1.0)

        # ── Motor orientation ─────────────────────────────────────────────────
        # The genome's motor_rotation (yaw) and motor_pitch define the motor
        # disc orientation in **world space**:
        #
        #   motor_pitch = 0    → disc faces straight up (+Z in world space)
        #   motor_pitch = π/2  → disc faces radially outward (toward motor tip)
        #   motor_rotation     → azimuthal rotation of the disc normal in world XY
        #
        # motor_arm.py builds the disc in the arm's local frame (tube = +Z), then
        # the assembler applies:
        #
        #   Step 1: Ry(β)  where β = 90° − arm_elevation  (tilt arm to elevation)
        #   Step 2: Rz(α)  where α = attachment_angle      (spin arm to azimuth)
        #
        # To produce the correct world-space normal from genome parameters, we
        # must express the desired world normal in the arm's local frame by
        # applying the **inverse** of the assembler's rotations:
        #
        #   n_local = Ry(−β) @ Rz(−α) @ n_world
        #
        # where n_world = [sin(p)·cos(y), sin(p)·sin(y), cos(p)]
        #       p = motor_pitch (radians), y = motor_rotation (radians)
        #
        # Then (motor_tilt, motor_azimuth) are recovered by decomposing n_local:
        #   motor_tilt    = acos(n_local_z)              in [0°, 180°]
        #   motor_azimuth = atan2(n_local_y, n_local_x)  in [0°, 360°)

        p_rad = float(motor_pitch)
        y_rad = float(motor_rotation)

        # Desired world-space disc normal
        nwx = math.sin(p_rad) * math.cos(y_rad)
        nwy = math.sin(p_rad) * math.sin(y_rad)
        nwz = math.cos(p_rad)

        # Inverse of assembler's Rz(α): apply Rz(−α)
        alpha = math.radians(attachment_angle)
        ca, sa = math.cos(alpha), math.sin(alpha)
        nx_rz = nwx * ca + nwy * sa
        ny_rz = -nwx * sa + nwy * ca
        nz_rz = nwz

        # Inverse of assembler's Ry(β): apply Ry(−β)
        beta = math.radians(90.0 - arm_elevation)
        cb, sb = math.cos(beta), math.sin(beta)
        nlx = nx_rz * cb - nz_rz * sb
        nly = ny_rz
        nlz = nx_rz * sb + nz_rz * cb

        # Decompose local normal into (motor_tilt, motor_azimuth)
        motor_tilt = math.degrees(math.acos(max(-1.0, min(1.0, nlz))))
        motor_azimuth = math.degrees(math.atan2(nly, nlx)) % 360.0

        # ── Motor flip detection ──────────────────────────────────────────────
        # Check whether the world-space disc normal points back into the arm
        # (i.e. the motor faces toward the core plate).  We already have the
        # world normal n_world = (nwx, nwy, nwz) from the computation above.
        #
        # If dot(arm_outward, n_world) < 0 the motor faces back toward the
        # plate — the disc would intersect the arm tube.  Flip by rotating
        # 180° around the local X-axis:
        #   motor_tilt    →  180° − motor_tilt
        #   motor_azimuth →  motor_azimuth + 180°
        # The spin direction is also inverted so thrust direction is preserved.
        _FLIP_DEG = 5.0          # degrees past 90° before flip triggers
        _FLIP_COS = math.cos(math.radians(90.0 + _FLIP_DEG))  # ≈ −0.087

        arm_len = math.sqrt(vx * vx + vy * vy + vz * vz)
        if arm_len > 0:
            ux, uy, uz = vx / arm_len, vy / arm_len, vz / arm_len
        else:
            ux, uy, uz = 1.0, 0.0, 0.0

        dot = ux * nwx + uy * nwy + uz * nwz
        if dot < _FLIP_COS:
            motor_tilt = 180.0 - motor_tilt
            motor_azimuth = (motor_azimuth + 180.0) % 360.0
            direction = 1 - int(direction)

        arms_params.append(ArmCADParameters(
            attachment_angle=attachment_angle,
            mount_angle=mount_angle,
            arm_elevation=arm_elevation,
            arm_length=arm_length,
            motor_tilt=motor_tilt,
            motor_azimuth=motor_azimuth,
            direction=int(direction),
        ))

    return DroneCADParameters(arms=arms_params)
