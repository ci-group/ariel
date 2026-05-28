"""Recover best individual from a completed overnight run database.

Reads the best individual directly from the SQLite DB (no fetch_population,
so it won't OOM on large runs).  Saves:
  - best_blueprint_<RUN_ID>.json
  - gate_pos_<RUN_ID>.npy
  - gate_yaw_<RUN_ID>.npy
  - best_policy_<RUN_ID>.zip  (if policy_b64 tag is present)

Usage:
    uv run examples/spear/10b_recover_best.py --db __data__/drone_evo_overnight/<RUN_ID>/database_<RUN_ID>.db
"""
import argparse
import base64
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rich.console import Console

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome

# ── Gate track parameters — must match the original run ───────────────────────
GATE_PATH_STEPS = 15
GATE_PATH_SCALE = 5.0
GATE_Z_HEIGHT   = -1.5
SEED            = 42
PROP_SIZE       = 2

_COEFFS_PATH = (
    _REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator" / "quintic_coeffs.npy"
)

console = Console()

parser = argparse.ArgumentParser(description="Recover best individual from overnight DB")
parser.add_argument("--db", required=True, help="Path to the database_*.db file")
args = parser.parse_args()

db_path = Path(args.db)
if not db_path.exists():
    console.log(f"[red]DB not found: {db_path}[/red]")
    raise SystemExit(1)

out_dir = db_path.parent
run_id  = db_path.stem.removeprefix("database_")

# ── Load best row directly from DB ────────────────────────────────────────────
con = sqlite3.connect(db_path)
cur = con.cursor()
cur.execute(
    "SELECT genotype_, tags_, fitness_ FROM individual "
    "WHERE fitness_ IS NOT NULL ORDER BY fitness_ DESC LIMIT 1"
)
row = cur.fetchone()
con.close()

if row is None:
    console.log("[red]No evaluated individuals in DB.[/red]")
    raise SystemExit(1)

genotype_json, tags_json, fitness = row
console.log(f"Best fitness: {fitness:.4f}")

genotype_data = json.loads(genotype_json)
tags          = json.loads(tags_json) if tags_json else {}

# ── Save blueprint ─────────────────────────────────────────────────────────────
genome  = deserialize_genome(genotype_data)
best_bp = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)

bp_path = out_dir / f"best_blueprint_{run_id}.json"
best_bp.save_json(bp_path)
console.log(f"Blueprint → {bp_path}")

# ── Save gate positions ────────────────────────────────────────────────────────
coeffs = np.load(_COEFFS_PATH)

OVERSAMPLE  = max(GATE_PATH_STEPS * 8, 64)
MIN_SPACING = 0.3
paths, yaws_dense_arr = generate_paths_from_coefficients(
    coeffs, num_generate=1, steps=OVERSAMPLE, seed=SEED, clip_range=(-1.0, 1.0),
)
xy_dense  = paths[0] * GATE_PATH_SCALE
yaw_dense = yaws_dense_arr[0]

kept = [0]
for idx in range(1, len(xy_dense)):
    if np.linalg.norm(xy_dense[idx] - xy_dense[kept[-1]]) >= MIN_SPACING:
        kept.append(idx)
    if len(kept) == GATE_PATH_STEPS:
        break
if len(kept) < GATE_PATH_STEPS:
    kept = list(np.linspace(0, len(xy_dense) - 1, GATE_PATH_STEPS, dtype=int))

xy       = xy_dense[kept]
gate_pos = np.column_stack([xy, np.full(GATE_PATH_STEPS, GATE_Z_HEIGHT)]).astype(np.float32)
gate_yaw = yaw_dense[kept].astype(np.float32)

np.save(out_dir / f"gate_pos_{run_id}.npy", gate_pos)
np.save(out_dir / f"gate_yaw_{run_id}.npy", gate_yaw)
console.log(f"Gate pos  → {out_dir / f'gate_pos_{run_id}.npy'}")
console.log(f"Gate yaw  → {out_dir / f'gate_yaw_{run_id}.npy'}")

# ── Save policy ────────────────────────────────────────────────────────────────
if "policy_b64" in tags:
    policy_path = out_dir / f"best_policy_{run_id}.zip"
    policy_path.write_bytes(base64.b64decode(tags["policy_b64"]))
    console.log(f"Policy    → {policy_path}")
else:
    console.log("[yellow]No policy_b64 tag found — policy not saved.[/yellow]")

console.log(f"\nTo visualise:\n  uv run examples/e_drones_ec/6_visualize_evo_results.py {out_dir}")
