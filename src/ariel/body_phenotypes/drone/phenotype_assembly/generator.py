"""
STL / STEP Generator

Top-level orchestration: converts a genome handler into physical files.

Pipeline
--------
genome_handler
    → genome_adapter.genome_to_cad_parameters()   → DroneCADParameters
    → assembler.assemble_drone()                   → compound solid + named dict
    → export (STL / STEP)

Public API
----------
    generate_stl_files(genome_handler, ...) -> STLGenerationResult
    quick_visualize_genome(genome_handler, output_file) -> Path
"""

import cadquery as cq
from pathlib import Path
from typing import Optional

from .models import AssemblyConfig, DroneCADParameters, STLGenerationResult
from .genome_adapter import genome_to_cad_parameters
from .assembler import assemble_drone
from .parts.core_plate import create_core_plate

# Part colours for STEP export
_PART_COLORS = {
    "sphere": cq.Color("lightblue"),
    "motor_arm": cq.Color("orange"),
}


def generate_stl_files(
    genome_handler,
    output_dir: str = "./drone_stls",
    include_assembly: bool = True,
    include_individual_parts: bool = True,
    include_motor_mounts: bool = False,
    include_arm_mounts: bool = False,
    include_step_files: bool = True,
    include_landing_leg: bool = False,
    distribute_arms_evenly: bool = False,
    magnitude_to_length_scale: float = 100.0,
    assembly_config: Optional[AssemblyConfig] = None,
    snap_mounts: bool = False,
    num_mount_positions: int = 8,
) -> STLGenerationResult:
    """
    Generate STL (and optionally STEP) files for a drone from a genome handler.

    Args:
        genome_handler: ``SphericalAngularDroneGenomeHandler`` with a loaded genome.
        output_dir: Directory to write output files.
        include_assembly: If True, write a combined ``full_drone_assembly.stl``.
        include_individual_parts: If True, write ``core_plate.stl`` and one
            ``arm_N.stl`` per arm (integrated tube + motor disc).
        include_motor_mounts: If True, write separate ``motor_mount_N.stl`` files
            (hollow socket with motor disc, for modular assembly).
        include_arm_mounts: If True, write separate ``arm_mount_N.stl`` files
            (sphere with arm socket hole, for modular assembly).
        include_step_files: If True, also write STEP equivalents.
        include_landing_leg: Reserved for future use (no-op currently).
        distribute_arms_evenly: When True, arm attachment angles are evenly
            spaced regardless of the genome's ``arm_rotation`` values.
        magnitude_to_length_scale: Genome magnitude → arm length in mm.
        assembly_config: Physical dimensions.  Defaults to ``AssemblyConfig()``.
        snap_mounts: When True, snap each arm mount to the nearest of
            ``num_mount_positions`` discrete positions on the plate rim.
            The motor tip position is preserved.  Default False.
        num_mount_positions: Number of discrete mount positions (default 8).

    Returns:
        ``STLGenerationResult`` with paths to every generated file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if assembly_config is None:
        assembly_config = AssemblyConfig()

    result = STLGenerationResult(output_dir=output_path)

    # ── Convert genome ───────────────────────────────────────────────────────
    cad_params: DroneCADParameters = genome_to_cad_parameters(
        genome_handler,
        magnitude_to_length_scale=magnitude_to_length_scale,
        distribute_arms_evenly=distribute_arms_evenly,
        assembly_config=assembly_config,
        snap_mounts=snap_mounts,
        num_mount_positions=num_mount_positions,
    )

    if cad_params.num_arms == 0:
        print("Warning: no valid arms in genome — skipping STL generation.")
        return result

    print(f"Generating STL files for drone with {cad_params.num_arms} arms")
    print(f"Output directory: {output_path}")

    # ── Core plate ───────────────────────────────────────────────────────────
    print("\nGenerating core plate…")
    core_plate = create_core_plate(
        plate_diameter=assembly_config.plate_diameter,
        plate_thickness=assembly_config.plate_thickness,
        outer_ring_width=assembly_config.outer_ring_width,
        include_hollow_square=assembly_config.include_hollow_square,
        hollow_square_outer_size=assembly_config.hollow_square_outer_size,
        hollow_square_wall_thickness=assembly_config.hollow_square_wall_thickness,
    )

    if include_individual_parts:
        core_plate_file = output_path / "core_plate.stl"
        cq.exporters.export(core_plate, str(core_plate_file))
        result.core_plate_file = core_plate_file
        print(f"  Saved: {core_plate_file.name}")

    if include_step_files:
        step_file = output_path / "core_plate.step"
        core_plate.val().exportStep(str(step_file))
        result.step_files.append(step_file)

    # ── Arms (individual STLs) ───────────────────────────────────────────────
    if include_individual_parts:
        from .assembler import place_arm_on_plate

        for i, arm_params in enumerate(cad_params.arms):
            print(f"\nGenerating arm {i + 1}…")
            print(f"  Attachment angle: {arm_params.attachment_angle:.1f}°  "
                  f"Elevation: {arm_params.arm_elevation:.1f}°  "
                  f"Length: {arm_params.arm_length:.1f} mm")

            placed = place_arm_on_plate(arm_params, assembly_config)
            arm_compound = cq.Compound.makeCompound(
                [p.val() for p in placed.values()]
            )
            arm_file = output_path / f"arm_{i + 1}.stl"
            cq.exporters.export(arm_compound, str(arm_file))
            result.arm_files.append(arm_file)
            print(f"  Saved: {arm_file.name}")

    # ── Motor mounts (separate STLs for modular assembly) ────────────────────
    if include_motor_mounts:
        from .parts.motor_mount import create_motor_mount

        for i, arm_params in enumerate(cad_params.arms):
            print(f"\nGenerating motor mount {i + 1}…")
            print(f"  Motor tilt: {arm_params.motor_tilt:.1f}°  "
                  f"Motor azimuth: {arm_params.motor_azimuth:.1f}°")

            motor_mount = create_motor_mount(
                motor_tilt=arm_params.motor_tilt,
                motor_azimuth=arm_params.motor_azimuth,
                cylinder_inner_radius=assembly_config.cylinder_inner_radius,
                pocket_thickness=assembly_config.wall_thickness,
                sphere_offset=assembly_config.sphere_offset,
                cylinder_extension=assembly_config.cylinder_extension,
                disc_diameter=assembly_config.disc_diameter,
                disc_thickness=assembly_config.disc_thickness,
                motor_screw_count=assembly_config.motor_screw_count,
                motor_screw_start_angle=assembly_config.motor_screw_start_angle,
                motor_screw_depth=assembly_config.motor_screw_depth,
                center_hole_diameter=assembly_config.center_hole_diameter,
            )

            motor_mount_file = output_path / f"motor_mount_{i + 1}.stl"
            cq.exporters.export(motor_mount, str(motor_mount_file))
            result.motor_mount_files.append(motor_mount_file)
            print(f"  Saved: {motor_mount_file.name}")

    # ── Arm mounts (separate STLs for modular assembly) ──────────────────────
    if include_arm_mounts:
        from .parts.arm_mount import create_arm_mount

        for i, arm_params in enumerate(cad_params.arms):
            print(f"\nGenerating arm mount {i + 1}…")
            print(f"  Mount angle: {arm_params.mount_angle:.1f}°  "
                  f"Arm elevation: {arm_params.arm_elevation:.1f}°  "
                  f"Arm azimuth: {arm_params.attachment_angle:.1f}°")

            mount_parts = create_arm_mount(
                sphere_radius=assembly_config.sphere_radius,
                clamp_inset=assembly_config.clamp_inset,
                arm_plate_diameter=assembly_config.plate_diameter,
                arm_plate_thickness=assembly_config.plate_thickness,
                arm_screw_hole_inset=assembly_config.arm_screw_hole_inset,
                include_arm_socket=True,  # Enable socket for separate printing
                arm_elevation=arm_params.arm_elevation,
                arm_azimuth=arm_params.attachment_angle,
            )

            # Extract sphere from dict and export
            arm_mount_file = output_path / f"arm_mount_{i + 1}.stl"
            cq.exporters.export(mount_parts["sphere"], str(arm_mount_file))
            result.arm_mount_files.append(arm_mount_file)
            print(f"  Saved: {arm_mount_file.name}")

    # ── Full assembly ────────────────────────────────────────────────────────
    if include_assembly:
        print("\nGenerating full assembly…")
        compound, named = assemble_drone(cad_params.arms, assembly_config, core_plate)

        assembly_file = output_path / "full_drone_assembly.stl"
        cq.exporters.export(compound, str(assembly_file))
        result.assembly_file = assembly_file
        print(f"  Saved: {assembly_file.name}")

        if include_step_files:
            step_assembly = cq.Assembly()
            step_assembly.add(
                named["core_plate"],
                name="core_plate",
                color=cq.Color("gray"),
            )
            for i, arm_parts in enumerate(named["arms"]):
                for part_name, part_wp in arm_parts.items():
                    step_assembly.add(
                        part_wp,
                        name=f"arm{i + 1}_{part_name}",
                        color=_PART_COLORS.get(part_name, cq.Color("white")),
                    )

            step_file = output_path / "full_drone_assembly.step"
            step_assembly.save(str(step_file))
            result.step_files.append(step_file)
            print(f"  Saved STEP: {step_file.name}")

    print("\n" + "=" * 70)
    print("STL generation completed.")
    print("=" * 70)
    print(f"\nGenerated files in: {output_path}")
    if result.core_plate_file:
        print(f"  Core plate: {result.core_plate_file.name}")
    if result.arm_files:
        print(f"  Arms: {len(result.arm_files)} files")
    if result.motor_mount_files:
        print(f"  Motor mounts: {len(result.motor_mount_files)} files")
    if result.arm_mount_files:
        print(f"  Arm mounts: {len(result.arm_mount_files)} files")
    if result.assembly_file:
        print(f"  Assembly: {result.assembly_file.name}")
    if result.step_files:
        print(f"  STEP files: {len(result.step_files)} files")

    return result


def quick_visualize_genome(
    genome_handler,
    output_file: str = "quick_drone.stl",
) -> Optional[Path]:
    """
    Quickly generate a single combined STL for visual inspection.

    Args:
        genome_handler: ``SphericalAngularDroneGenomeHandler`` with a loaded genome.
        output_file: Path for the output STL.

    Returns:
        ``Path`` to the generated file, or ``None`` if generation failed.
    """
    import os

    out_dir = os.path.dirname(output_file) or "."
    result = generate_stl_files(
        genome_handler,
        output_dir=out_dir,
        include_assembly=True,
        include_individual_parts=False,
        include_step_files=False,
        include_landing_leg=False,
    )

    if result.assembly_file:
        dest = Path(output_file)
        result.assembly_file.rename(dest)
        return dest

    return None
