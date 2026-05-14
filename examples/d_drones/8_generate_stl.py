"""Generate fabrication-ready STL/STEP files from an evolved drone genome.

Creates a physical assembly (arms + motor mounts + propeller discs) from a
spherical drone genome and exports it to STL (3-D printing) and optionally
STEP (CAD editing) formats. Also renders a before/after comparison of the
evolved genome vs. the evenly-distributed fabrication layout.

Run:
    # Randomly generated 4-arm genome:
    python examples/d_drones/8_generate_stl.py

    # Specify arm count and output directory:
    python examples/d_drones/8_generate_stl.py --arms 6 --out __data__/my_drone_stl

    # Headless (no interactive matplotlib windows):
    python examples/d_drones/8_generate_stl.py --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import DroneVisualizer
from airevolve.phenotype_assembly import generate_stl_files, AssemblyConfig

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Generate STL files from a drone genome")
parser.add_argument("--arms", type=int, default=4,
                    help="Number of rotor arms (default 4)")
parser.add_argument("--prop-size", type=int, default=6,
                    help="Propeller size in inches for STL geometry (default 6)")
parser.add_argument("--out", default="__data__/drone_stl",
                    help="Output directory for STL/STEP files (default __data__/drone_stl)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-show", action="store_true",
                    help="Save visualizations without showing interactive windows")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

output_dir = Path(args.out)
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Generate a random genome
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [0, 1],
])

handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(args.arms, args.arms),
    parameter_limits=PARAMETER_LIMITS,
    rnd=np.random.default_rng(args.seed),
)
handler._generate_random_genome(innovation_ids=np.arange(args.arms))

valid_arms = handler.get_valid_arms()
num_arms = len(valid_arms)

print("=" * 70)
print(f"Generate STL — {num_arms}-arm drone")
print("=" * 70)
print("\nArm parameters:")
for i, arm in enumerate(valid_arms):
    print(
        f"  Arm {i + 1}: mag={arm[0]:.3f}  "
        f"arm_az={np.degrees(arm[1]):.1f}°  arm_el={np.degrees(arm[2]):.1f}°  "
        f"motor_az={np.degrees(arm[3]):.1f}°  motor_pitch={np.degrees(arm[4]):.1f}°  "
        f"spin={'CW' if arm[5] > 0.5 else 'CCW'}"
    )

# ---------------------------------------------------------------------------
# Generate STL files
# ---------------------------------------------------------------------------

print(f"\nGenerating STL files → {output_dir}")

assembly_config = AssemblyConfig(propeller_size=args.prop_size)
result = generate_stl_files(
    handler,
    output_dir=str(output_dir),
    assembly_config=assembly_config,
)

print("Generation complete!")
print(f"Output directory: {result.output_dir}")

# ---------------------------------------------------------------------------
# Genome comparison: evolved vs. fabrication (evenly distributed) layout
# ---------------------------------------------------------------------------

print("\n" + "-" * 70)
print("GENOME COMPARISON: evolved vs. fabrication layout")
print("-" * 70)

viz_genome = valid_arms.copy()
for i in range(num_arms):
    viz_genome[i, 1] = (2 * np.pi / num_arms) * i  # evenly distribute azimuth

print("\nOriginal genome (as evolved):")
for i, arm in enumerate(valid_arms):
    print(f"  Arm {i + 1}: arm_az={np.degrees(arm[1]):.1f}°  mag={arm[0]:.3f}")

print("\nFabrication genome (evenly distributed arm azimuth):")
for i, arm in enumerate(viz_genome):
    print(f"  Arm {i + 1}: arm_az={np.degrees(arm[1]):.1f}°  mag={arm[0]:.3f}")

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

visualizer = DroneVisualizer()

# Fabrication layout blueprint
fig, _ = visualizer.plot_blueprint(
    viz_genome,
    title=f"Fabrication Layout — {num_arms} Arms ({args.prop_size}-inch props)",
)
viz_path = output_dir / "drone_visualization_fabrication.png"
fig.savefig(viz_path, dpi=150, bbox_inches="tight")
print(f"\nFabrication visualisation saved: {viz_path}")
if not args.no_show:
    plt.show()
plt.close(fig)

# Original genome blueprint
fig2, _ = visualizer.plot_blueprint(
    handler,
    title=f"Original Evolved Genome — {num_arms} Arms",
)
viz_path_orig = output_dir / "drone_visualization_original.png"
fig2.savefig(viz_path_orig, dpi=150, bbox_inches="tight")
print(f"Original genome visualisation saved: {viz_path_orig}")
if not args.no_show:
    plt.show()
plt.close(fig2)

# 6-panel analysis dashboard
fig3 = plt.figure(figsize=(16, 12))
fig3.suptitle(f"Drone Genome Analysis — {num_arms} Arms", fontsize=16)
for idx, (elev, azim, title) in enumerate([
    (30, 45, "Isometric"),
    (0,  0,  "Front"),
    (0,  90, "Side"),
    (90, 0,  "Top (3D)"),
    (20, 60, "Alternate"),
], start=1):
    ax = fig3.add_subplot(2, 3, idx, projection="3d")
    visualizer.plot_3d(handler, ax=ax, title=title, elevation=elev, azimuth=azim)
ax6 = fig3.add_subplot(2, 3, 6)
visualizer.plot_2d(handler, ax=ax6, title="Top View (2D)")
plt.tight_layout()
analysis_path = output_dir / "drone_detailed_analysis.png"
fig3.savefig(analysis_path, dpi=150, bbox_inches="tight")
print(f"Detailed analysis saved: {analysis_path}")
if not args.no_show:
    plt.show()
plt.close(fig3)

print("\n" + "=" * 70)
print("You can now:")
print("  1. Open the STL files in MeshLab, Blender, or your slicer")
print("  2. Open the STEP files in FreeCAD or Fusion 360")
print("  3. Send the STL files to a 3-D printer")
print("=" * 70)
