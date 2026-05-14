"""Drone genome visualisation demo — 2D, 3D, and blueprint views.

Shows how to visualise evolved (spherical) and hand-crafted (cartesian) drone
genomes using airevolve's DroneVisualizer. Demonstrates:

  • 2D top-down view
  • 3D isometric view
  • Blueprint (4-panel) view
  • Coordinate-system comparison (cartesian vs spherical)
  • Multi-panel analysis dashboard

Run:
    python examples/d_drones/7_visualize_genome.py
    python examples/d_drones/7_visualize_genome.py --no-show   # headless, save only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from airevolve.evolution_tools.genome_handlers.cartesian_euler_genome_handler import (
    CartesianEulerDroneGenomeHandler,
)
from airevolve.evolution_tools.inspection_tools.drone_visualizer import (
    DroneVisualizer,
    VisualizationConfig,
)
import airevolve.evolution_tools.inspection_tools.utils as u

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drone genome visualisation demo")
parser.add_argument("--no-show", action="store_true",
                    help="Save figures without showing interactive windows")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

OUTPUT_DIR = Path.cwd() / "__data__" / "visualizations" / "genome_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

visualizer = DroneVisualizer()

# ---------------------------------------------------------------------------
# Helper genomes
# ---------------------------------------------------------------------------

def make_quad_cartesian() -> CartesianEulerDroneGenomeHandler:
    """Classic + quad in Cartesian encoding."""
    data = np.array([
        [ 0.3,  0.0, 0.0, 0.0, 0.0, 0.0, 1],
        [-0.3,  0.0, 0.0, 0.0, 0.0, 0.0, 1],
        [ 0.0,  0.3, 0.0, 0.0, 0.0, 0.0, 0],
        [ 0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0],
    ])
    return CartesianEulerDroneGenomeHandler(genome=data, min_max_narms=(4, 4))


def make_hex_cartesian() -> CartesianEulerDroneGenomeHandler:
    """Regular hexacopter in Cartesian encoding."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    r = 0.35
    data = np.array([
        [r * np.cos(a), r * np.sin(a), 0.0, 0.0, 0.0, 0.0, float(i % 2)]
        for i, a in enumerate(angles)
    ])
    return CartesianEulerDroneGenomeHandler(genome=data, min_max_narms=(6, 6))


def make_random_spherical() -> np.ndarray:
    """Random 4-arm drone in spherical encoding (phenotype array)."""
    rng = np.random.default_rng(0)
    handler = SphericalAngularDroneGenomeHandler(
        min_max_narms=(4, 4),
        parameter_limits=np.array([
            [0.055, 0.17], [-np.pi, np.pi], [-np.pi / 2, np.pi / 2],
            [-np.pi, np.pi], [-np.pi, np.pi], [0, 1],
        ]),
        rnd=rng,
    )
    g = handler._generate_random_genome(innovation_ids=np.arange(4))
    valid = ~np.isnan(g.arms[:, 0])
    return g.arms[valid]


def make_tilted_cartesian() -> CartesianEulerDroneGenomeHandler:
    """Quad with tilted motors (30°) in Cartesian encoding."""
    tilt = np.pi / 6
    data = np.array([
        [ 0.25,  0.25, 0.0, 0.0,  tilt, 0.0, 1],
        [ 0.25, -0.25, 0.0, 0.0,  tilt, 0.0, 1],
        [-0.25, -0.25, 0.0, 0.0, -tilt, 0.0, 0],
        [-0.25,  0.25, 0.0, 0.0, -tilt, 0.0, 0],
    ])
    return CartesianEulerDroneGenomeHandler(genome=data, min_max_narms=(4, 4))


# ---------------------------------------------------------------------------
# Demo 1: Basic 3D and 2D
# ---------------------------------------------------------------------------

print("=== Demo 1: Basic 3D and 2D views ===")

fig, ax = visualizer.plot_3d(make_quad_cartesian(), title="Quadcopter — 3D View", fitness=0.95)
fig.savefig(OUTPUT_DIR / "demo_3d_quadcopter.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

fig, ax = visualizer.plot_2d(make_hex_cartesian(), title="Hexacopter — Top View", fitness=0.87)
fig.savefig(OUTPUT_DIR / "demo_2d_hexacopter.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

print(f"  Saved to {OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Demo 2: Coordinate systems — Cartesian vs spherical phenotype
# ---------------------------------------------------------------------------

print("=== Demo 2: Coordinate system comparison ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
visualizer.plot_2d(make_quad_cartesian(), ax=axes[0], title="Cartesian coordinates")
visualizer.plot_2d(make_random_spherical(), ax=axes[1], title="Spherical phenotype")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "demo_coordinate_systems.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 3: Blueprint (4-panel)
# ---------------------------------------------------------------------------

print("=== Demo 3: Blueprint views ===")

fig, _ = visualizer.plot_blueprint(
    make_tilted_cartesian(), title="Tilted Motors Drone — Blueprint"
)
fig.savefig(OUTPUT_DIR / "demo_blueprint.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 4: Complex geometry grid
# ---------------------------------------------------------------------------

print("=== Demo 4: Complex geometry grid ===")

configs = [
    (make_tilted_cartesian(), "Tilted Motors"),
    (make_hex_cartesian(), "Hexacopter"),
    (make_quad_cartesian(), "Quadcopter"),
    (make_random_spherical(), "Random Spherical"),
]

fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                         subplot_kw={"projection": "3d"})
for ax, (genome, title) in zip(axes.flat, configs):
    visualizer.plot_3d(genome, ax=ax, title=title, elevation=45, azimuth=45)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "demo_complex_geometries.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 5: Multi-panel analysis dashboard for a single genome
# ---------------------------------------------------------------------------

print("=== Demo 5: Multi-panel analysis dashboard ===")

genome = make_tilted_cartesian()
fig = plt.figure(figsize=(15, 10))
fig.suptitle(f"Drone Analysis Dashboard — Tilted Motors Quad", fontsize=16)

ax1 = plt.subplot(2, 3, 1, projection="3d")
visualizer.plot_3d(genome, ax=ax1, title="Isometric (default)", elevation=30, azimuth=45)

ax2 = plt.subplot(2, 3, 2)
visualizer.plot_2d(genome, ax=ax2, title="Top View (2D)")

ax3 = plt.subplot(2, 3, 3, projection="3d")
visualizer.plot_3d(genome, ax=ax3, title="Front View", elevation=0, azimuth=0)

ax4 = plt.subplot(2, 3, 4)
cfg_detail = VisualizationConfig(include_motor_orientation_2d=1)
DroneVisualizer(cfg_detail).plot_2d(genome, ax=ax4, title="With Motor Orientations")

ax5 = plt.subplot(2, 3, 5)
cfg_minimal = VisualizationConfig(show_limits=False)
DroneVisualizer(cfg_minimal).plot_2d(genome, ax=ax5, title="Minimalist")

ax6 = plt.subplot(2, 3, 6, projection="3d")
visualizer.plot_3d(genome, ax=ax6, title="Alternate Angle", elevation=20, azimuth=60)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "demo_dashboard.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 6: Coordinate conversion utilities
# ---------------------------------------------------------------------------

print("=== Demo 6: Coordinate conversion utilities ===")

mag, azi, pit = 0.12, np.pi / 4, np.pi / 6
x, y, z = u.convert_to_cartesian(mag, azi, pit)
mag2, azi2, pit2 = u.convert_to_spherical(x, y, z)
print(
    f"  Spherical ({mag:.2f}, {np.degrees(azi):.1f}°, {np.degrees(pit):.1f}°)"
    f" → Cartesian ({x:.3f}, {y:.3f}, {z:.3f})"
    f" → back ({mag2:.2f}, {np.degrees(azi2):.1f}°, {np.degrees(pit2):.1f}°)"
)

print(f"\nAll figures saved to: {OUTPUT_DIR}")
