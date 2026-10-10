"""Arm mount CAD primitive."""

from __future__ import annotations

import cadquery as cq


def create_arm_mount(
    *,
    sphere_radius: float,
    clamp_inset: float,
    arm_plate_diameter: float,
    arm_plate_thickness: float,
    arm_screw_hole_inset: float,
    include_arm_socket: bool = False,
    arm_elevation: float = 0.0,
    arm_azimuth: float = 0.0,
) -> dict[str, cq.Workplane]:
    """Create a simple spherical arm mount with a rim clamp notch.

    Extra parameters are accepted for API compatibility with existing callers.
    """
    del arm_elevation, arm_azimuth, arm_screw_hole_inset

    sphere = cq.Workplane("XY").sphere(sphere_radius)

    notch_depth = max(0.1, min(sphere_radius * 1.5, clamp_inset + arm_plate_thickness))
    notch = (
        cq.Workplane("XY")
        .box(sphere_radius * 2.2, arm_plate_diameter, arm_plate_thickness + 1.0)
        .translate((sphere_radius - notch_depth, 0, 0))
    )
    sphere = sphere.cut(notch)

    if include_arm_socket:
        socket = (
            cq.Workplane("YZ")
            .circle(max(0.6, sphere_radius * 0.35))
            .extrude(sphere_radius * 2.5)
            .translate((0.0, 0.0, 0.0))
        )
        sphere = sphere.cut(socket)

    return {"sphere": sphere}
