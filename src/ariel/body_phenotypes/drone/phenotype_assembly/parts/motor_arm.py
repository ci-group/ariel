"""Motor arm CAD primitive."""

from __future__ import annotations

import cadquery as cq
import math


def create_motor_arm(
    *,
    arm_length: float,
    motor_tilt: float,
    motor_azimuth: float,
    cylinder_inner_radius: float,
    wall_thickness: float,
    cylinder_extension: float,
    sphere_offset: float,
    disc_diameter: float,
    disc_thickness: float,
    motor_screw_count: int,
    motor_screw_start_angle: float,
    motor_screw_depth: float,
    center_hole_diameter: float,
) -> cq.Workplane:
    """Create a hollow arm tube with a motor disc at its tip."""
    del motor_tilt, motor_azimuth, motor_screw_depth

    outer_radius = max(cylinder_inner_radius + wall_thickness, cylinder_inner_radius + 0.2)
    tube_height = max(1.0, arm_length + sphere_offset + cylinder_extension)

    tube = (
        cq.Workplane("XY")
        .circle(outer_radius)
        .circle(max(0.1, cylinder_inner_radius))
        .extrude(tube_height)
    )

    disc = (
        cq.Workplane("XY")
        .workplane(offset=tube_height)
        .circle(max(1.0, disc_diameter / 2.0))
        .extrude(max(0.5, disc_thickness))
    )
    arm = tube.union(disc)

    if center_hole_diameter > 0:
        arm = arm.faces(">Z").workplane().hole(center_hole_diameter)

    if motor_screw_count > 0:
        screw_r = max(0.1, disc_diameter * 0.35)
        screw_d = max(0.6, wall_thickness)
        wp = arm.faces(">Z").workplane()
        for i in range(motor_screw_count):
            angle_deg = motor_screw_start_angle + i * (360.0 / motor_screw_count)
            wp = wp.pushPoints([
                (
                    screw_r * math.cos(math.radians(angle_deg)),
                    screw_r * math.sin(math.radians(angle_deg)),
                )
            ])
        arm = wp.hole(screw_d)

    return arm
