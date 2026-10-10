"""Emit a URDF file from a DroneBlueprint and inspect the result.

Pipeline:

    spherical_angular genome
      → spherical_angular_to_blueprint   (decoder)
      → DroneBlueprint                   (saved as JSON for inspection)
      → blueprint_to_urdf                (NEW backend — this example)
      → quad.urdf                        (rigid drone, fixed joints)

The emitted URDF is the intermediate for the Isaac Lab pipeline. Hand
it to ``scripts/urdf_to_usd.py`` in an Isaac Lab env to produce a USD:

    /path/to/isaaclab/python scripts/urdf_to_usd.py --headless \\
        --input examples/d_drones/__data__/blueprint_demo/quad.urdf \\
        --output_dir examples/d_drones/__data__/blueprint_demo/usd

Cross-section dispatch: ``--rect-arms`` swaps the default hollow-tube
arms (8mm OD / 6mm ID, matching ariel's ``BEAM_DENSITY``) for solid
rectangular box arms, which exercises ``blueprint_to_urdf``'s
``isinstance(arm.cross_section, RectangularCrossSection)`` branch.

Run:
    uv run examples/d_drones/16_blueprint_to_urdf.py
    uv run examples/d_drones/16_blueprint_to_urdf.py --arm-len 0.18 --prop-size 4
    uv run examples/d_drones/16_blueprint_to_urdf.py --rect-arms
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.backends import blueprint_to_urdf
from ariel.body_phenotypes.drone.blueprint import (
    ArmNode,
    RectangularCrossSection,
)
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--arm-len", type=float, default=0.12,
                    help="Arm length in metres (default 0.12).")
parser.add_argument("--prop-size", type=int, default=5,
                    help="Propeller size in inches (default 5).")
parser.add_argument("--rect-arms", action="store_true",
                    help="Swap default HollowTube arms for solid rectangular "
                         "(10 mm wide × 5 mm thick) — exercises the URDF box "
                         "geometry path.")
parser.add_argument("--robot-name", type=str, default="quad")
parser.add_argument("--out", type=Path,
                    default=Path("examples/d_drones/__data__/blueprint_demo/quad.urdf"),
                    help="Destination URDF path.")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# 1. Build an X-quad spherical-angular genome and decode to a blueprint
# ---------------------------------------------------------------------------

mag = float(args.arm_len)
genome = np.array([
    [mag,  np.pi / 4,        0.0, 0.0, 0.0, 0],
    [mag,  3 * np.pi / 4,    0.0, 0.0, 0.0, 1],
    [mag, -3 * np.pi / 4,    0.0, 0.0, 0.0, 0],
    [mag, -np.pi / 4,        0.0, 0.0, 0.0, 1],
])

bp = spherical_angular_to_blueprint(genome, propsize=args.prop_size)

if args.rect_arms:
    # Re-author each ArmNode's cross_section in place.
    for arm_id in bp.children(bp.root_id):  # type: ignore[arg-type]
        arm = bp.payload(arm_id)
        if isinstance(arm, ArmNode):
            arm.cross_section = RectangularCrossSection(width=0.010, thickness=0.005)


# ---------------------------------------------------------------------------
# 2. Emit URDF
# ---------------------------------------------------------------------------

args.out.parent.mkdir(parents=True, exist_ok=True)
bp.save_json(args.out.with_suffix(".json"))
urdf_path = blueprint_to_urdf(bp, str(args.out), robot_name=args.robot_name)

print("=" * 70)
print(bp.summary())
print(f"\nBlueprint JSON: {args.out.with_suffix('.json')}")
print(f"URDF:           {urdf_path}")


# ---------------------------------------------------------------------------
# 3. Quick structural inspection — parse the URDF back and tally
# ---------------------------------------------------------------------------

tree = ET.parse(urdf_path)
robot = tree.getroot()
links = robot.findall("link")
joints = robot.findall("joint")

print("\nURDF structure:")
print(f"  links  : {len(links)} (expect 1 core + N arms + N motors)")
print(f"  joints : {len(joints)} (all fixed in v1)")

total_mass = 0.0
for link in links:
    mass_el = link.find("inertial/mass")
    if mass_el is not None:
        total_mass += float(mass_el.attrib["value"])
print(f"  Σ mass : {total_mass:.4f} kg")

geom_tags = sorted({
    g.tag for L in links for g in L.findall("visual/geometry/*")
})
print(f"  arm/motor geometry types in URDF: {geom_tags}")

print("\nNext step (Isaac Lab env):")
print(f"  $ISAAC_PYTHON scripts/urdf_to_usd.py --headless \\")
print(f"      --input {urdf_path} --output_dir {args.out.parent}/usd")
