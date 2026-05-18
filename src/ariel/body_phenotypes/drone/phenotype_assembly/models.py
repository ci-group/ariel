"""
Data models for the phenotype assembly pipeline.

All dataclasses used across genome_adapter, assembler, and generator live here
so there is a single source of truth for each type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from . import config


# ─────────────────────────────────────────────────────────────────────────────
# Arm / drone parameters produced by genome_adapter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArmCADParameters:
    """Physical parameters for one arm, ready for CAD generation.

    Angles are all in **degrees** (converted from genome radians by
    ``genome_adapter``).

    Attributes:
        attachment_angle: Azimuthal angle (0–360°) the arm tube points toward
            in the world XY plane.  0° = +X axis, 90° = +Y axis, etc.
            When ``snap_mounts=False`` this equals the genome's ``arm_rotation``
            directly.  When ``snap_mounts=True`` it is the angle from the
            snapped mount position toward the motor tip.
        mount_angle: Azimuthal angle (0–360°) of the arm mount sphere on the
            plate rim.  When ``snap_mounts=False`` this equals
            ``attachment_angle``.  When ``snap_mounts=True`` this is snapped
            to the nearest discrete mount position (e.g. multiples of 45° for
            8 positions).
        arm_elevation: Elevation of the arm above horizontal (degrees).
            0° = arm lies in the XY plane, positive = arm tilts upward.
            When snapping is active this is recomputed from the vector between
            the snapped mount and the original motor tip.
        arm_length: Length of the arm tube in mm.  When snapping is active
            this is recomputed as the distance from the snapped mount to the
            motor tip minus ``sphere_offset``.
        motor_tilt: Motor disc tilt in **local arm space** (degrees), passed
            directly to ``create_motor_arm``.  0° = disc normal aligns with
            +Z (arm tube axis); 90° = disc normal points radially outward.
            Computed by ``genome_adapter`` by inverting the assembler's
            Ry/Rz rotations so that ``motor_pitch=0`` in the genome means
            the disc faces straight up in world space regardless of arm
            orientation.
        motor_azimuth: Motor disc azimuth in **local arm space** (degrees,
            rotation around Z after tilt), chosen so that the resulting
            world-space disc normal matches the genome's ``motor_rotation``
            (which is interpreted as a world-frame azimuth).
        direction: Propeller spin direction.  0 = CCW, 1 = CW.
    """
    attachment_angle: float   # degrees, 0–360 — direction arm tube points
    mount_angle: float        # degrees, 0–360 — where sphere sits on plate rim
    arm_elevation: float      # degrees, 0 = horizontal, positive = up
    arm_length: float         # mm
    motor_tilt: float         # degrees (local arm space)
    motor_azimuth: float      # degrees
    direction: int            # 0 = CCW, 1 = CW


@dataclass
class DroneCADParameters:
    """Complete set of per-arm CAD parameters for one drone individual."""
    arms: List[ArmCADParameters]

    @property
    def num_arms(self) -> int:
        return len(self.arms)


# ─────────────────────────────────────────────────────────────────────────────
# Assembly configuration (replaces the stringly-typed part_config dict)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AssemblyConfig:
    """Physical dimensions used during part generation and assembly.

    All measurements in **mm**.  Defaults mirror the values previously
    scattered across ``config.py`` and ``stl_generator.py``.
    """
    # Core plate (must match the actual plate being used)
    plate_diameter: float = config.plate_diameter
    plate_thickness: float = config.plate_thickness
    outer_ring_width: float = config.outer_ring_width
    include_hollow_square: bool = True
    hollow_square_outer_size: float = 33.0
    hollow_square_wall_thickness: float = 2.0

    # Arm mount (sphere clamp)
    sphere_radius: float = 12.0
    clamp_inset: float = 10.5
    arm_screw_hole_inset: float = config.outer_ring_width / 2  # = 3.25 mm

    # Motor arm tube
    cylinder_inner_radius: float = config.arm_cylinder_inner_radius
    wall_thickness: float = 1.0
    cylinder_extension: float = 13.428
    sphere_offset: float = 5.0

    # Motor disc
    disc_diameter: float = 23.0
    disc_thickness: float = 3.0
    motor_screw_count: int = 4
    motor_screw_start_angle: float = 45.0
    motor_screw_depth: float = 25.0
    center_hole_diameter: float = 0.0

    @property
    def plate_radius(self) -> float:
        return self.plate_diameter / 2.0

    @property
    def outer_ring_inner_radius(self) -> float:
        return self.plate_radius - self.outer_ring_width

    @property
    def outer_ring_middle_radius(self) -> float:
        return (self.plate_radius + self.outer_ring_inner_radius) / 2.0

    @property
    def arm_attach_radial_distance(self) -> float:
        """Radial distance from drone centre to the arm mount sphere centre.

        The sphere overlaps the plate rim inward by ``clamp_inset``.  Therefore:

            sphere_centre_r = plate_radius + (sphere_radius - clamp_inset)

        With defaults: 30 + (12 - 10.5) = 31.5 mm.

        Screw holes are at sphere_centre_r - arm_screw_hole_inset from centre,
        i.e. 31.5 - 3.25 = 28.25 mm, landing in the middle of the outer ring.
        """
        return self.plate_radius + (self.sphere_radius - self.clamp_inset)


# ─────────────────────────────────────────────────────────────────────────────
# Result type returned by generate_stl_files
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class STLGenerationResult:
    """Paths to every file produced by a single generate_stl_files() call."""
    output_dir: Path
    core_plate_file: Optional[Path] = None
    arm_files: List[Path] = field(default_factory=list)
    motor_mount_files: List[Path] = field(default_factory=list)
    arm_mount_files: List[Path] = field(default_factory=list)
    assembly_file: Optional[Path] = None
    landing_leg_file: Optional[Path] = None
    step_files: List[Path] = field(default_factory=list)
