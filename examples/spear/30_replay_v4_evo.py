"""Replay a v4-warmstarted EA individual from its database.

Reads ``examples/spear/29_drone_evo_v4_warmstart.py``'s SQLite DB, extracts
one individual (by rank or by id), unpacks its blueprint + policy.zip +
vecnormalize.pkl to a working directory, and hands off to
``27_eval_rl_hex_mtrl_v4.py`` for the actual MuJoCo viewer / MP4 render.

The reads talk directly to sqlite (mirrors ``10b_recover_best.py``), so
this stays cheap even on very large runs — no full ``fetch_population``.

Usage
-----
    # Visualise the best individual in the MuJoCo viewer:
    uv run examples/spear/30_replay_v4_evo.py \\
        --db __data__/drone_evo_warmstart_v4/<RUN>/database_<RUN_ID>.db

    # Pick by rank (best | median | worst) or by raw rank index (0 = best):
    uv run examples/spear/30_replay_v4_evo.py --db <…>.db --rank 3

    # Pick a specific individual by its SQL id:
    uv run examples/spear/30_replay_v4_evo.py --db <…>.db --id 42

    # Render an MP4 per task instead of opening the viewer:
    uv run examples/spear/30_replay_v4_evo.py --db <…>.db --no-view

NOTE: do NOT add ``from __future__ import annotations`` to this file.
"""

import argparse
import base64
import json
import sqlite3
import subprocess
import time
from pathlib import Path

from rich.console import Console

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome

console = Console()

# Must match CONFIG 2 of the EA run that produced the DB.
PROP_SIZE    = 2
GATE_DENSITY = 3

# 29_drone_evo_v4_warmstart launches each fine-tune from this v4 base; if a
# row in the DB doesn't have vecnorm_b64 (e.g. an older run), we fall back
# to the v4 base stats.
WARMSTART_DIR_FALLBACK = Path("__data__/spear_rl_hex_mtrl_v4/20260616_163031")

_EVAL_SCRIPT = Path(__file__).with_name("27_eval_rl_hex_mtrl_v4.py")


def _resolve_row(con: sqlite3.Connection, ind_id: int | None,
                 rank: str) -> tuple[int, str, dict, float]:
    """Return (id, genotype_json, tags_dict, fitness) for the requested row.

    ``rank`` may be ``"best"``, ``"worst"``, ``"median"``, or an integer
    string interpreted as a 0-indexed rank from the top.
    """
    cur = con.cursor()
    if ind_id is not None:
        cur.execute(
            "SELECT id, genotype_, tags_, fitness_ FROM individual WHERE id = ?",
            (ind_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"No individual with id={ind_id} in DB.")
        return row[0], row[1], json.loads(row[2] or "{}"), row[3]

    # Fitness-ordered queries
    if rank == "best":
        order = "DESC"; offset = 0
    elif rank == "worst":
        order = "ASC"; offset = 0
    elif rank == "median":
        cur.execute("SELECT COUNT(*) FROM individual WHERE fitness_ IS NOT NULL")
        n = cur.fetchone()[0]
        if n == 0:
            raise SystemExit("No evaluated individuals in DB.")
        order = "DESC"; offset = n // 2
    else:
        try:
            offset = int(rank)
        except ValueError as exc:
            raise SystemExit(
                f"--rank must be 'best', 'worst', 'median', or an integer; "
                f"got {rank!r}"
            ) from exc
        if offset < 0:
            raise SystemExit("--rank must be ≥ 0")
        order = "DESC"

    cur.execute(
        f"SELECT id, genotype_, tags_, fitness_ FROM individual "
        f"WHERE fitness_ IS NOT NULL ORDER BY fitness_ {order} LIMIT 1 OFFSET ?",
        (offset,),
    )
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"No individual at rank offset {offset} in DB.")
    return row[0], row[1], json.loads(row[2] or "{}"), row[3]


def _extract_artifacts(genotype_data: dict, tags: dict, out_dir: Path) -> dict:
    """Write blueprint.json + policy.zip + vecnormalize.pkl into out_dir.

    Returns the dict of paths actually written so the caller can build the
    eval-script CLI.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict = {}

    genome = deserialize_genome(genotype_data)
    bp     = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)
    bp_path = out_dir / "blueprint.json"
    bp.save_json(bp_path)
    paths["blueprint"] = bp_path

    if "policy_b64" in tags:
        policy_path = out_dir / "policy.zip"
        policy_path.write_bytes(base64.b64decode(tags["policy_b64"]))
        paths["policy"] = policy_path
    else:
        raise SystemExit(
            "Selected individual has no policy_b64 tag — was it evaluated?"
        )

    if "vecnorm_b64" in tags:
        vn_path = out_dir / "vecnormalize.pkl"
        vn_path.write_bytes(base64.b64decode(tags["vecnorm_b64"]))
        paths["vecnormalize"] = vn_path
    else:
        fallback = WARMSTART_DIR_FALLBACK / "vecnormalize.pkl"
        if fallback.exists():
            vn_path = out_dir / "vecnormalize.pkl"
            vn_path.write_bytes(fallback.read_bytes())
            paths["vecnormalize"] = vn_path
            console.log(
                f"[yellow]No per-candidate vecnorm_b64 in DB row; "
                f"falling back to v4 base: {fallback}[/yellow]"
            )
        else:
            console.log(
                "[yellow]No vecnorm available — the policy will see "
                "unnormalised obs and likely fly poorly.[/yellow]"
            )
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Replay a v4-warmstarted EA individual from its database"
    )
    parser.add_argument("--db", required=True,
                        help="Path to database_*.db file written by 29_drone_evo_v4_warmstart.py")
    parser.add_argument("--id", type=int, default=None,
                        help="Pick the individual with this exact SQL id.")
    parser.add_argument("--rank", default="best",
                        help="If --id is not given: 'best' | 'worst' | 'median' | "
                             "integer 0-based offset from best. Default: best.")
    parser.add_argument("--task", default="all",
                        help="Forwarded to 27_eval_rl_hex_mtrl_v4 (figure8/slalom/"
                             "shuttle-run/hover/all). Default: all.")
    parser.add_argument("--rollout-time", type=float, default=15.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-view", action="store_true",
                        help="Render MP4s into <out-dir>/viz/ instead of opening the viewer.")
    parser.add_argument("--out-dir", default=None,
                        help="Where to unpack artifacts. Default: <db_dir>/replay_<id>_<ts>/.")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        row_id, genotype_json, tags, fitness = _resolve_row(con, args.id, args.rank)
    finally:
        con.close()

    label = f"id={row_id}"
    if args.id is None:
        label = f"id={row_id} (rank={args.rank})"
    console.log(f"Selected individual {label} — fitness={fitness:.4f}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        db_path.parent / f"replay_id{row_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    paths = _extract_artifacts(json.loads(genotype_json), tags, out_dir)
    console.log(f"Unpacked artifacts → {out_dir}")
    for k, v in paths.items():
        console.log(f"  {k:>13} → {v}")

    cmd = [
        "uv", "run", str(_EVAL_SCRIPT),
        "--policy",        str(paths["policy"]),
        "--blueprint-json", str(paths["blueprint"]),
        "--rollout-time",  str(args.rollout_time),
        "--gate-density",  str(GATE_DENSITY),
        "--device",        args.device,
        "--task",          args.task,
    ]
    if "vecnormalize" in paths:
        cmd += ["--vecnormalize", str(paths["vecnormalize"])]
    if args.no_view:
        cmd += ["--no-view", "--out-dir", str(out_dir / "viz")]

    console.log(f"Launching: {' '.join(cmd)}")
    # No capture — let the eval script's viewer / progress go straight to the
    # user's terminal.
    rc = subprocess.call(cmd)
    if rc != 0:
        console.log(f"[red]Eval script exited with code {rc}[/red]")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
