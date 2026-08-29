#!/usr/bin/env python3
"""Build a DroneBlueprint and emit a URDF file.

Ariel-env side of the two-step Blueprint → URDF → USD pipeline. The
emitted ``.urdf`` is consumed by ``scripts/urdf_to_usd.py`` in an Isaac
Lab environment (the two envs are separated because ariel requires
Python 3.12 PEP 695 syntax and the local isaaclab env runs 3.11).

Examples:

    # Quad with default propsize=5
    python scripts/blueprint_to_urdf.py --preset quad --out /tmp/quad.urdf

    # Load a saved blueprint
    python scripts/blueprint_to_urdf.py --blueprint_json bp.json --out drone.urdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make ariel importable when run from a checkout without `pip install -e .`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from ariel.body_phenotypes.drone.backends import blueprint_to_urdf  # noqa: E402
from ariel.body_phenotypes.drone.blueprint import DroneBlueprint  # noqa: E402
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint  # noqa: E402


# (mag, az, pitch, motor_az, motor_pitch, direction) per arm.
PRESETS: dict[str, np.ndarray] = {
    "quad": np.array([
        [0.18, 0.0,            0.0, 0.0, 0.0, 1.0],
        [0.18, np.pi / 2.0,    0.0, 0.0, 0.0, 0.0],
        [0.18, np.pi,          0.0, 0.0, 0.0, 1.0],
        [0.18, 3 * np.pi / 2,  0.0, 0.0, 0.0, 0.0],
    ]),
    "hex": np.array([
        [0.18, i * np.pi / 3.0, 0.0, 0.0, 0.0, float(i % 2)]
        for i in range(6)
    ]),
}


def _build_blueprint(args: argparse.Namespace) -> DroneBlueprint:
    if args.blueprint_json:
        bp = DroneBlueprint.load_json(args.blueprint_json)
        print(f"[blueprint_to_urdf] Loaded blueprint from {args.blueprint_json}")
        return bp
    genome = PRESETS[args.preset]
    bp = spherical_angular_to_blueprint(genome, propsize=args.propsize)
    print(f"[blueprint_to_urdf] Built '{args.preset}' preset (propsize={args.propsize})")
    return bp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=list(PRESETS), default="quad")
    parser.add_argument("--blueprint_json", type=str, default=None,
                        help="Optional: load a saved DroneBlueprint JSON instead of a preset.")
    parser.add_argument("--propsize", type=int, default=5,
                        help="Propeller size for the preset (ignored if --blueprint_json given).")
    parser.add_argument("--out", type=str, required=True,
                        help="Destination .urdf path.")
    parser.add_argument("--robot_name", type=str, default="drone")
    args = parser.parse_args()

    bp = _build_blueprint(args)
    urdf_path = blueprint_to_urdf(bp, args.out, robot_name=args.robot_name)
    print(f"[blueprint_to_urdf] URDF written: {urdf_path}")


if __name__ == "__main__":
    main()
