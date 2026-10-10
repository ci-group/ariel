"""Core plate CAD primitive."""

from __future__ import annotations

import cadquery as cq


def create_core_plate(
    *,
    plate_diameter: float,
    plate_thickness: float,
    outer_ring_width: float,
    include_hollow_square: bool = True,
    hollow_square_outer_size: float = 33.0,
    hollow_square_wall_thickness: float = 2.0,
) -> cq.Workplane:
    """Create a printable core plate as a ring-like disc.

    The geometry is intentionally simple and robust for export.
    """
    plate_radius = plate_diameter / 2.0
    inner_radius = max(0.1, plate_radius - outer_ring_width)

    ring = (
        cq.Workplane("XY")
        .circle(plate_radius)
        .circle(inner_radius)
        .extrude(plate_thickness)
    )

    if include_hollow_square:
        inner_square = max(0.1, hollow_square_outer_size - 2.0 * hollow_square_wall_thickness)
        if inner_square > 0.0:
            ring = ring.faces(">Z").workplane().rect(inner_square, inner_square).cutBlind(-plate_thickness)

    return ring
