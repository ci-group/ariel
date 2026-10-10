"""
Drone Assembler

Places arm-mount and motor-arm parts onto the core plate at the correct
position and orientation for each arm.

Coordinate conventions
----------------------
Core plate
  Sits at the origin.  Bottom face at Z=0, top face at Z=plate_thickness.
  The plate centre is at (0, 0).  The outer rim is at radius = plate_radius.

Arm mount (sphere + clamp)
  Created in local space with the clamp slot pointing in the +X direction.
  The slot centre (= plate rim contact) is at local X = slot_x =
  sphere_radius - clamp_inset from the sphere centre.

  To place it correctly we:
    1. Rotate 180° around Z so the slot faces the *inward* direction (−X →
       toward plate centre after the next rotation).
    2. Rotate by ``attachment_angle`` around Z.
    3. Translate sphere centre to (sphere_centre_r × cos, sphere_centre_r × sin, z).

  ``sphere_centre_r`` = plate_radius - clamp_inset  (sphere inset from rim).

Motor arm tube
  Created in local space pointing along +Z.  The tube base is at Z=0, the
  motor disc centre is at Z = arm_length + sphere_offset.

  To place it correctly we:
    1. Rotate −(90° − arm_elevation)° around Y to tilt the tube from vertical
       to the correct elevation angle.  After this the tube points radially
       outward in the XZ plane.
    2. Rotate by ``attachment_angle`` around Z to spin to the correct azimuth.
    3. Translate the tube base to the sphere surface point on the plate rim.
"""

import math
import cadquery as cq
from typing import Dict, List, Tuple

from .models import ArmCADParameters, AssemblyConfig
from .parts.arm_mount import create_arm_mount
from .parts.motor_arm import create_motor_arm


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────

def arm_attach_position(
    attachment_angle_deg: float,
    cfg: AssemblyConfig,
) -> Tuple[float, float, float]:
    """
    Return the (x, y, z) of the arm mount sphere centre for a given attachment angle.

    The sphere sits just outside the plate rim, overlapping it by ``clamp_inset``
    so the clamp jaws can grip the outer ring.  The sphere centre is therefore at:

        r = plate_radius + (sphere_radius - clamp_inset)

    With defaults (plate_radius=30, sphere_radius=12, clamp_inset=10.5):
        r = 30 + 1.5 = 31.5 mm

    Z is set to plate_thickness / 2 so the sphere straddles the plate mid-plane,
    letting the slot cutter (in arm_mount.py) remove material for the rim to slide in.
    """
    r = cfg.plate_radius + (cfg.sphere_radius - cfg.clamp_inset)
    angle_rad = math.radians(attachment_angle_deg)
    return (
        r * math.cos(angle_rad),
        r * math.sin(angle_rad),
        cfg.plate_thickness / 2.0,
    )


def place_arm_on_plate(
    arm_params: ArmCADParameters,
    cfg: AssemblyConfig,
) -> Dict[str, cq.Workplane]:
    """
    Build all parts for one arm and position them relative to the core plate.

    Args:
        arm_params: Parameters for this arm from ``DroneCADParameters.arms``.
        cfg: Physical assembly configuration.

    Returns:
        dict with keys ``'sphere'``, ``'motor_arm'``.  All solids are already
        in the global drone frame (origin = plate centre bottom).
    """
    # mount_angle: where the sphere sits on the plate rim (discrete when snapping)
    # attachment_angle: azimuth the arm tube points toward in world space
    # These are equal when snap_mounts=False; they differ when the mount has
    # been snapped to a discrete position.
    mount_angle = arm_params.mount_angle
    arm_azimuth = arm_params.attachment_angle

    # ── 1. Arm mount (sphere) ─────────────────────────────────────────────────
    mount_parts = create_arm_mount(
        sphere_radius=cfg.sphere_radius,
        clamp_inset=cfg.clamp_inset,
        arm_plate_diameter=cfg.plate_diameter,
        arm_plate_thickness=cfg.plate_thickness,
        arm_screw_hole_inset=cfg.arm_screw_hole_inset,
    )

    sphere_centre_x, sphere_centre_y, sphere_centre_z = arm_attach_position(mount_angle, cfg)

    placed_mount: Dict[str, cq.Workplane] = {}
    for part_name, part_wp in mount_parts.items():
        solid = part_wp.val()
        # Slot faces +X in local space; rotate 180° so slot now faces −X
        # (toward the plate centre), then rotate to the mount azimuth on the rim.
        solid = solid.rotate((0, 0, 0), (0, 0, 1), 180.0)
        solid = solid.rotate((0, 0, 0), (0, 0, 1), mount_angle)
        solid = solid.translate((sphere_centre_x, sphere_centre_y, sphere_centre_z))
        placed_mount[part_name] = cq.Workplane("XY").add(solid)

    # ── 2. Motor arm tube ────────────────────────────────────────────────────
    motor_arm_wp = create_motor_arm(
        arm_length=arm_params.arm_length,
        motor_tilt=arm_params.motor_tilt,
        motor_azimuth=arm_params.motor_azimuth,
        cylinder_inner_radius=cfg.cylinder_inner_radius,
        wall_thickness=cfg.wall_thickness,
        cylinder_extension=cfg.cylinder_extension,
        sphere_offset=cfg.sphere_offset,
        disc_diameter=cfg.disc_diameter,
        disc_thickness=cfg.disc_thickness,
        motor_screw_count=cfg.motor_screw_count,
        motor_screw_start_angle=cfg.motor_screw_start_angle,
        motor_screw_depth=cfg.motor_screw_depth,
        center_hole_diameter=cfg.center_hole_diameter,
    )

    arm_solid = motor_arm_wp.val()

    # The tube is created along +Z.  We want it to point from the mount sphere
    # toward the motor tip at ``arm_elevation`` degrees above horizontal.
    #
    # Step 1: tilt from vertical (+Z) to the desired elevation.
    #   elevation = 0  → horizontal → tilt 90° around Y maps +Z→+X (outward)
    #   elevation = 90 → straight up → no tilt needed (stays at +Z)
    # Rotation angle = 90° − elevation  (positive Y rotation: +Z toward +X)
    tilt_angle = 90.0 - arm_params.arm_elevation  # degrees
    arm_solid = arm_solid.rotate((0, 0, 0), (0, 1, 0), tilt_angle)

    # Step 2: spin to the arm's world-space azimuth (direction toward motor tip).
    # This may differ from mount_angle when snap_mounts=True.
    arm_solid = arm_solid.rotate((0, 0, 0), (0, 0, 1), arm_azimuth)

    # Step 3: translate so that the tube base sits at the sphere surface on
    # the plate rim side.  The sphere centre is at sphere_centre_{x,y,z}.
    # The arm exits the sphere in the *outward* radial direction.  After the
    # rotations above the tube base (Z=0 in local space) maps to a point that
    # is ``sphere_radius`` inward from the sphere centre along the arm axis,
    # so we need to account for that offset.
    #
    # Rather than computing the surface intersection analytically we place the
    # tube base at the sphere centre and let the sphere geometry handle the
    # overlap visually.  For printed parts the sphere clamp and tube are two
    # separate pieces, so exact surface contact is a fitment concern, not a
    # geometry error.
    arm_solid = arm_solid.translate((sphere_centre_x, sphere_centre_y, sphere_centre_z))

    placed_mount["motor_arm"] = cq.Workplane("XY").add(arm_solid)

    return placed_mount


def assemble_drone(
    arms_params: List[ArmCADParameters],
    cfg: AssemblyConfig,
    core_plate_solid,
) -> Tuple[cq.Compound, Dict]:
    """
    Assemble all drone parts into a single compound and a named-parts dict.

    Args:
        arms_params: List of ``ArmCADParameters``, one per arm.
        cfg: Assembly configuration.
        core_plate_solid: The core plate CadQuery solid (from ``create_core_plate``).

    Returns:
        Tuple of:
          - ``cq.Compound``: All parts merged for STL export.
          - ``dict``: ``{'core_plate': solid, 'arms': [{part_name: solid, …}, …]}``
            for STEP assembly export with named / coloured parts.
    """
    all_solids = [core_plate_solid.val()]
    named = {"core_plate": core_plate_solid, "arms": []}

    for arm_params in arms_params:
        placed = place_arm_on_plate(arm_params, cfg)
        arm_entry = {}

        for part_name, part_wp in placed.items():
            solid = part_wp.val()
            all_solids.append(solid)
            arm_entry[part_name] = part_wp

        named["arms"].append(arm_entry)

    compound = cq.Compound.makeCompound(all_solids)
    return compound, named
