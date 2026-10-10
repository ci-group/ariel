"""
Phenotype Assembly

Tools for generating physical drone parts (STL / STEP files) from evolved genomes.

Typical usage
-------------
    from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
        SphericalAngularDroneGenomeHandler,
    )
    from ariel.body_phenotypes.drone.phenotype_assembly import generate_stl_files

    handler = SphericalAngularDroneGenomeHandler(...)
    # ... evolve ...
    result = generate_stl_files(handler, output_dir="./my_drone")
    print(result.assembly_file)

Module layout
-------------
models.py           — dataclasses: ArmCADParameters, DroneCADParameters,
                      AssemblyConfig, STLGenerationResult
genome_adapter.py   — genome → DroneCADParameters conversion
assembler.py        — positions and orients parts onto the plate
generator.py        — orchestrates the full pipeline; file I/O
parts/
    core_plate.py   — central hub-and-spoke plate
    arm_mount.py    — sphere-clamp that grips the plate rim
    motor_arm.py    — arm tube + motor-mounting disc
"""

from .generator import generate_stl_files, quick_visualize_genome
from .genome_adapter import genome_to_cad_parameters
from .models import (
    ArmCADParameters,
    DroneCADParameters,
    AssemblyConfig,
    STLGenerationResult,
)
from .assembler import place_arm_on_plate, assemble_drone
from .parts import create_core_plate, create_arm_mount, create_motor_arm

__all__ = [
    # ── Main API ───────────────────────────────────────────────────────────
    "generate_stl_files",
    "quick_visualize_genome",
    # ── Data models ────────────────────────────────────────────────────────
    "ArmCADParameters",
    "DroneCADParameters",
    "AssemblyConfig",
    "STLGenerationResult",
    # ── Conversion ─────────────────────────────────────────────────────────
    "genome_to_cad_parameters",
    # ── Assembly helpers (advanced usage) ──────────────────────────────────
    "place_arm_on_plate",
    "assemble_drone",
    # ── Part generators (advanced usage) ───────────────────────────────────
    "create_core_plate",
    "create_arm_mount",
    "create_motor_arm",
]

__version__ = "0.2.0"
