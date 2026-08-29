
"""
Physical constants and default dimensions for phenotype assembly.

All measurements in millimeters unless otherwise noted.
"""

# ══════════════════════════════════════════════════════════════════════════════
# Screw / Fastener Parameters
# ══════════════════════════════════════════════════════════════════════════════

screw_size = 2.0
screw_size_tolerance = 0.3

# ══════════════════════════════════════════════════════════════════════════════
# Core Plate Parameters
# ══════════════════════════════════════════════════════════════════════════════

# Overall dimensions
plate_diameter = 60.0
plate_thickness = 2.0
plate_thickness_tolerance = 0.2

# Centre hub (solid disc at centre)
center_hub_radius = 20.0
central_hole_diameter = 25.0

# Outer ring (annular ring at perimeter)
outer_ring_width = 6.5
outer_ring_inner_radius = plate_diameter / 2 - outer_ring_width

# Radial struts (connecting hub to outer ring)
strut_width = 12.0
number_struts = 4

# Screw holes
screw_pattern_radius = plate_diameter / 2 - 7.5  # 7.5mm inset from edge
number_holes = 32
screw_square_size = 25.5

# ══════════════════════════════════════════════════════════════════════════════
# Arm Mount Parameters (sphere clamp on plate rim)
# ══════════════════════════════════════════════════════════════════════════════

# Inner cutout cylinders (above/below plate slot to reduce sphere overhang)
arm_mount_inner_cutout_height = 8.0

# Screw hole angles relative to slot centre (degrees)
arm_mount_screw_angles = [180.0 - 11.25, 180.0 + 11.25]

# ══════════════════════════════════════════════════════════════════════════════
# Motor Arm Parameters (tube + motor disc)
# ══════════════════════════════════════════════════════════════════════════════

# Arm tube dimensions
arm_cylinder_inner_radius = 8.0 / 2
arm_cylinder_inner_radius_tolerance = 0.3

# Motor disc mounting
intermediary_outer_screw_pattern_radius = 18.0 / 2

# Flattening disc (removes overhang material beyond motor disc face)
motor_flattening_disc_thickness = 10.0

# ══════════════════════════════════════════════════════════════════════════════
# Motor Mount Parameters (modular separate printable mount)
# ══════════════════════════════════════════════════════════════════════════════

# Socket cylinder dimensions
motor_mount_cylinder_height = 20.0
motor_mount_cylinder_bottom_thickness = 20.0

# Mounting screw
motor_mount_screw_inset = 5.0
