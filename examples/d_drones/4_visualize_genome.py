"""Visualise a drone genome as 3D blueprint and propeller diagrams.

Mirrors src/airevolve/examples/visualization/genome_visualizer_demo.py and
draw_blueprint.py using ARIEL imports.

Usage:
    # Visualise a random 2-inch quad genome:
    uv run examples/d_drones/4_visualize_genome.py

    # Visualise a genome saved to .npy:
    uv run examples/d_drones/4_visualize_genome.py \\
        --genome-file __data__/drone_evolution/genome.npy

    # Save figures without displaying:
    uv run examples/d_drones/4_visualize_genome.py --save --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drone genome visualizer")
parser.add_argument("--genome-file", default=None,
                    help="Path to .npy genome file (optional; uses random genome if omitted)")
parser.add_argument("--save", action="store_true",
                    help="Save figures to the output directory")
parser.add_argument("--no-show", action="store_true",
                    help="Do not display interactive figures (useful for headless runs)")
parser.add_argument("--save-dir", default="__data__/genome_viz",
                    help="Output directory for saved figures (default __data__/genome_viz)")
args = parser.parse_args()

save_dir = Path(args.save_dir)
if args.save:
    save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load or generate genome
# ---------------------------------------------------------------------------

if args.genome_file:
    individual = np.load(args.genome_file, allow_pickle=True).astype(np.float64)
    # Remove NaN rows to get valid arms
    valid_mask = ~np.isnan(individual[:, 0]) if individual.ndim == 2 else ~np.isnan(individual)
    individual = individual[valid_mask]
    print(f"Loaded genome from {args.genome_file}  ({len(individual)} arms)")
else:
    # Generate a random 4-arm spherical genome
    handler = SphericalAngularDroneGenomeHandler(min_max_narms=(4, 8))
    genome = handler._generate_random_genome()
    # Extract the valid (non-NaN) arms
    valid_mask = ~np.isnan(genome.arms[:, 0])
    individual = genome.arms[valid_mask]
    print(f"Generated random genome  ({len(individual)} arms)")

print(f"Genome shape: {individual.shape}")
print(f"Arms (mag, arm_rot, arm_pitch, mot_rot, mot_pitch, dir):")
for i, arm in enumerate(individual):
    print(f"  [{i}]  {np.round(arm, 3)}")

# ---------------------------------------------------------------------------
# Blueprint visualisation
# ---------------------------------------------------------------------------

visualizer = DroneVisualizer()

print("\nRendering blueprint …")
fig_bp, axes_bp = visualizer.plot_blueprint(individual, title="Drone Blueprint")

if args.save:
    path = save_dir / "blueprint.png"
    fig_bp.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Blueprint saved: {path}")

# ---------------------------------------------------------------------------
# 3D visualisation
# ---------------------------------------------------------------------------

print("Rendering 3D view …")
fig_3d = plt.figure(figsize=(8, 7))
ax_3d = fig_3d.add_subplot(111, projection="3d")
visualizer.plot_3d(individual, ax=ax_3d, title="Drone 3D View")

if args.save:
    path = save_dir / "3d_view.png"
    fig_3d.savefig(path, dpi=150, bbox_inches="tight")
    print(f"3D view saved: {path}")

if not args.no_show:
    plt.show()
else:
    plt.close("all")
