"""Demonstration of the Drone Blueprint pipeline.

Two completely different genome encodings (spherical-angular and
cartesian-Euler) flow through the SAME downstream pipeline:

    genome → DroneBlueprint → propellers list → DroneConfiguration

The point: ARIEL's blueprint layer decouples encoding from embodiment,
so the consortium can converge on a shared phenotype path without
forcing partners to converge on a shared genotype.

Run:
    uv run examples/d_drones/11_blueprint_demo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ariel.body_phenotypes.drone.blueprint import DroneBlueprint
from ariel.body_phenotypes.drone.decoders import (
    spherical_angular_to_blueprint,
    cartesian_euler_to_blueprint,
)
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.simulation.drone.drone_configuration import DroneConfiguration


def banner(s: str) -> None:
    print("\n" + "=" * 70 + f"\n  {s}\n" + "=" * 70)


# ---- 1. Two encodings of (roughly) the same 4-arm drone ---------------------

# Spherical-angular: 4 arms evenly spaced at 90°, level, motors thrust +Z, alt spins.
mag = 0.11
spherical_genome = np.array([
    [mag, 0.00,        0.0, 0.0, 0.0, 0],   # +X
    [mag, np.pi / 2,   0.0, 0.0, 0.0, 1],   # +Y
    [mag, np.pi,       0.0, 0.0, 0.0, 0],   # -X
    [mag, -np.pi / 2,  0.0, 0.0, 0.0, 1],   # -Y
])

# Cartesian-Euler: same four motors expressed directly as XYZ + thrust RPY.
cartesian_genome = np.array([
    [ mag,  0.0,   0.0,  0.0, 0.0, 0.0, 0],
    [ 0.0,  mag,   0.0,  0.0, 0.0, 0.0, 1],
    [-mag,  0.0,   0.0,  0.0, 0.0, 0.0, 0],
    [ 0.0, -mag,   0.0,  0.0, 0.0, 0.0, 1],
])


# ---- 2. Decode each genome through ITS OWN decoder to a shared Blueprint ----

bp_a = spherical_angular_to_blueprint(spherical_genome)
bp_b = cartesian_euler_to_blueprint(cartesian_genome)

banner("Blueprint A (from spherical-angular genome)")
print(bp_a.summary())

banner("Blueprint B (from cartesian-Euler genome)")
print(bp_b.summary())


# ---- 3. Persist Blueprint A as JSON, round-trip it ---------------------------

out_dir = Path("__data__") / "blueprint_demo"
out_dir.mkdir(parents=True, exist_ok=True)

json_path = out_dir / "blueprint_A.json"
bp_a.save_json(json_path)
bp_a_roundtrip = DroneBlueprint.load_json(json_path)
banner(f"JSON round-trip → {json_path}")
print(f"  nodes before: {bp_a.g.number_of_nodes()}   after: "
      f"{bp_a_roundtrip.g.number_of_nodes()}")
assert bp_a.g.number_of_nodes() == bp_a_roundtrip.g.number_of_nodes()


# ---- 4. Same backend instantiates BOTH blueprints into the SAME format ------

props_a = blueprint_to_propellers(bp_a)
props_b = blueprint_to_propellers(bp_b)

banner("Propellers extracted from Blueprint A (spherical genome)")
for i, p in enumerate(props_a):
    print(f"  motor {i}: loc={[round(v,3) for v in p['loc']]}  "
          f"dir={[round(v,3) if isinstance(v,float) else v for v in p['dir']]}  "
          f"propsize={p['propsize']}")

banner("Propellers extracted from Blueprint B (cartesian genome)")
for i, p in enumerate(props_b):
    print(f"  motor {i}: loc={[round(v,3) for v in p['loc']]}  "
          f"dir={[round(v,3) if isinstance(v,float) else v for v in p['dir']]}  "
          f"propsize={p['propsize']}")


# ---- 5. Both feed the existing DroneConfiguration unchanged -----------------

cfg_a = DroneConfiguration(props_a)
cfg_b = DroneConfiguration(props_b)

banner("Downstream phenotype properties (same DroneConfiguration code path)")
print(f"  Blueprint A  →  mass={cfg_a.mass:.4f} kg   "
      f"Ix={cfg_a.Ix:.4e}  Iy={cfg_a.Iy:.4e}  Iz={cfg_a.Iz:.4e}")
print(f"  Blueprint B  →  mass={cfg_b.mass:.4f} kg   "
      f"Ix={cfg_b.Ix:.4e}  Iy={cfg_b.Iy:.4e}  Iz={cfg_b.Iz:.4e}")
print("\n  → Two distinct encodings, one shared phenotype pipeline. "
      "That's the blueprint.\n")
