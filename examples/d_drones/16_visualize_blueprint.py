"""Visualise a DroneBlueprint with the same DroneVisualizer views as
``8_visualize_genome.py``.

``8_visualize_genome.py`` visualises *genomes*. ``11_blueprint_demo.py``
shows two encodings decoding into a shared :class:`DroneBlueprint`. This
example closes the loop: it takes the blueprints produced exactly as in
``11_blueprint_demo.py`` and renders them with ``DroneVisualizer``'s 2D /
3D / blueprint (4-panel) views.

``DroneVisualizer`` consumes genome handlers or phenotype arrays — it does
not accept a ``DroneBlueprint`` directly. The bridge is one small helper,
:func:`blueprint_to_cartesian_array`, which flattens the blueprint tree
(via :func:`blueprint_to_propellers`) into the cartesian ``(n, 7)`` array
``[x, y, z, roll, pitch, yaw, direction]`` the visualizer understands.

Run:
    uv run examples/d_drones/16_visualize_blueprint.py
    uv run examples/d_drones/16_visualize_blueprint.py --no-show   # headless, save only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.body_phenotypes.drone.decoders import (
    cartesian_euler_to_blueprint,
    spherical_angular_to_blueprint,
)
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Visualise a DroneBlueprint")
parser.add_argument("--no-show", action="store_true",
                    help="Save figures without showing interactive windows")
args = parser.parse_args()

if args.no_show:
    matplotlib.use("Agg")

OUTPUT_DIR = Path.cwd() / "__data__" / "visualizations" / "blueprint_demo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Blueprint → visualizer-friendly phenotype array
# ---------------------------------------------------------------------------

def blueprint_to_cartesian_array(bp: DroneBlueprint) -> np.ndarray:
    """Flatten a DroneBlueprint to the cartesian ``(n, 7)`` array
    ``[x, y, z, roll, pitch, yaw, direction]`` accepted by ``DroneVisualizer``.

    Motor world positions and thrust normals come from
    :func:`blueprint_to_propellers`; the thrust normal ``n`` is turned back
    into a ``(roll, pitch, yaw)`` motor orientation (yaw fixed at 0, the free
    DOF about the thrust axis). ``direction`` follows the decoder convention
    ``0 = CCW, 1 = CW``.
    """
    rows: list[list[float]] = []
    for prop in blueprint_to_propellers(bp, convention="z_up"):
        x, y, z = prop["loc"]
        nx, ny, nz, spin = prop["dir"]
        # n = Ry(pitch) · Rx(roll) · [0, 0, 1]  ⇒  invert for roll, pitch.
        roll = float(np.arcsin(np.clip(-ny, -1.0, 1.0)))
        pitch = float(np.arctan2(nx, nz))
        direction = 1.0 if spin == "cw" else 0.0
        rows.append([float(x), float(y), float(z), roll, pitch, 0.0, direction])
    return np.asarray(rows, dtype=float)


# ---------------------------------------------------------------------------
# Build the two blueprints exactly as in 11_blueprint_demo.py
# ---------------------------------------------------------------------------

mag = 0.11

# Spherical-angular genome: 4 arms at 90°, level, alternating spin.
spherical_genome = np.array([
    [mag, 0.00,        0.0, 0.0, 0.0, 0],
    [mag, np.pi / 2,   0.0, 0.0, 0.0, 1],
    [mag, np.pi,       0.0, 0.0, 0.0, 0],
    [mag, -np.pi / 2,  0.0, 0.0, 0.0, 1],
])

# Cartesian-Euler genome: the same four motors as direct XYZ + thrust RPY.
cartesian_genome = np.array([
    [ mag,  0.0,  0.0, 0.0, 0.0, 0.0, 0],
    [ 0.0,  mag,  0.0, 0.0, 0.0, 0.0, 1],
    [-mag,  0.0,  0.0, 0.0, 0.0, 0.0, 0],
    [ 0.0, -mag,  0.0, 0.0, 0.0, 0.0, 1],
])

bp_spherical = spherical_angular_to_blueprint(spherical_genome)
bp_cartesian = cartesian_euler_to_blueprint(cartesian_genome)

# A blueprint can equally be loaded from disk (11_blueprint_demo.py saves one):
#     bp = DroneBlueprint.load_json("__data__/blueprint_demo/blueprint_A.json")

print("=== Blueprint (from spherical-angular genome) ===")
print(bp_spherical.summary())
print("\n=== Blueprint (from cartesian-Euler genome) ===")
print(bp_cartesian.summary())

arr_spherical = blueprint_to_cartesian_array(bp_spherical)
arr_cartesian = blueprint_to_cartesian_array(bp_cartesian)

visualizer = DroneVisualizer()

# ---------------------------------------------------------------------------
# Demo 1: 3D view
# ---------------------------------------------------------------------------

print("\n=== Demo 1: 3D view ===")

fig, _ = visualizer.plot_3d(
    arr_spherical, title="Blueprint (spherical genome) — 3D View",
)
fig.savefig(OUTPUT_DIR / "blueprint_3d.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 2: 2D top-down view
# ---------------------------------------------------------------------------

print("=== Demo 2: 2D top-down view ===")

fig, _ = visualizer.plot_2d(
    arr_spherical, title="Blueprint (spherical genome) — Top View",
)
fig.savefig(OUTPUT_DIR / "blueprint_2d.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 3: Blueprint (4-panel) view
# ---------------------------------------------------------------------------

print("=== Demo 3: Blueprint (4-panel) view ===")

fig, _ = visualizer.plot_blueprint(
    arr_spherical, title="Blueprint (spherical genome) — Blueprint View",
)
fig.savefig(OUTPUT_DIR / "blueprint_4panel.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

# ---------------------------------------------------------------------------
# Demo 4: Same blueprint format, two source encodings — side by side
# ---------------------------------------------------------------------------

print("=== Demo 4: Two encodings → one blueprint format ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
visualizer.plot_2d(arr_spherical, ax=axes[0], title="Blueprint ← spherical-angular genome")
visualizer.plot_2d(arr_cartesian, ax=axes[1], title="Blueprint ← cartesian-Euler genome")
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "blueprint_two_encodings.png", dpi=150, bbox_inches="tight")
if not args.no_show:
    plt.show()
plt.close(fig)

print(f"\nAll figures saved to: {OUTPUT_DIR}")
