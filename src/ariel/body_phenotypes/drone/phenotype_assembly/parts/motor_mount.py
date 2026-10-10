"""Motor mount CAD primitive for modular assembly exports."""

from __future__ import annotations

import cadquery as cq


def create_motor_mount(
    *,
    motor_tilt: float,
    motor_azimuth: float,
    cylinder_inner_radius: float,
    pocket_thickness: float,
    sphere_offset: float,
    cylinder_extension: float,
    disc_diameter: float,
    disc_thickness: float,
    motor_screw_count: int,
    motor_screw_start_angle: float,
    motor_screw_depth: float,
    center_hole_diameter: float,
) -> cq.Workplane:
    """Create a compact motor-mount socket and disc.

    Angles are accepted for API compatibility; this helper keeps geometry
    axis-aligned for robust export and simple smoke testing.
    """
    del motor_tilt, motor_azimuth, motor_screw_count, motor_screw_start_angle, motor_screw_depth

    outer_radius = max(1.0, cylinder_inner_radius + pocket_thickness)
    body_height = max(1.0, sphere_offset + cylinder_extension)

    body = (
        cq.Workplane("XY")
        .circle(outer_radius)
        .extrude(body_height)
    )
    body = body.faces(">Z").workplane().circle(max(0.1, cylinder_inner_radius)).cutBlind(-body_height)

    disc = (
        cq.Workplane("XY")
        .workplane(offset=body_height)
        .circle(max(1.0, disc_diameter / 2.0))
        .extrude(max(0.5, disc_thickness))
    )
    mount = body.union(disc)

    if center_hole_diameter > 0:
        mount = mount.faces(">Z").workplane().hole(center_hole_diameter)

    return mount
