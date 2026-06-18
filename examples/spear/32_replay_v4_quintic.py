"""Replay a 31_drone_evo_v4_quintic.py EA individual from its database.

Mirrors ``30_replay_v4_evo.py``. The extra piece: each row's
``quintic_tracks_b64`` tag is decoded back into the exact per-task gate
arrays the candidate was trained on, and ``27_eval_rl_hex_mtrl_v4`` is
invoked in-process with its ``_task_config`` monkey-patched so the viewer
uses those tracks (otherwise it would default back to GATE_CONFIGS).

Usage
-----
    # Best individual, MuJoCo viewer:
    uv run examples/spear/32_replay_v4_quintic.py \\
        --db __data__/drone_evo_quintic_v4/<RUN>/database_<RUN_ID>.db

    # Pick by rank or by id:
    uv run examples/spear/32_replay_v4_quintic.py --db <…>.db --rank 3
    uv run examples/spear/32_replay_v4_quintic.py --db <…>.db --id 42

    # MP4 per task instead of viewer:
    uv run examples/spear/32_replay_v4_quintic.py --db <…>.db --no-view
"""

import argparse
import base64
import importlib.util
import io
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome

console = Console()

PROP_SIZE    = 2
GATE_DENSITY = 3

# Same v4 base used by 31_drone_evo_v4_quintic.py.
WARMSTART_DIR_FALLBACK = Path("__data__/spear_rl_hex_mtrl_v4/20260616_163031")

# Tasks that were quintic-replaced at training time. Hover keeps GATE_CONFIGS.
QUINTIC_TASKS = ("figure8", "slalom", "shuttle-run")

_EVAL_SCRIPT = Path(__file__).with_name("27_eval_rl_hex_mtrl_v4.py")


def _resolve_row(con: sqlite3.Connection, ind_id, rank: str):
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
    """Write blueprint.json + policy.zip + vecnormalize.pkl + quintic_tracks.npz."""
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
                "[yellow]No vecnorm available — policy will see unnormalised obs.[/yellow]"
            )

    if "quintic_tracks_b64" in tags:
        tracks_path = out_dir / "quintic_tracks.npz"
        tracks_path.write_bytes(base64.b64decode(tags["quintic_tracks_b64"]))
        paths["tracks"] = tracks_path
    else:
        console.log(
            "[yellow]No quintic_tracks_b64 in DB row — falling back to "
            "GATE_CONFIGS for the racing tasks.[/yellow]"
        )
    return paths


def _load_tracks(npz_path: Path) -> dict:
    """Reload {task_name: (gate_pos, gate_yaw, start_pos)} from the npz."""
    data = np.load(npz_path)
    tracks: dict = {}
    for name in QUINTIC_TASKS:
        gp = data[f"{name}__gate_pos"]
        gy = data[f"{name}__gate_yaw"]
        sp = data[f"{name}__start_pos"]
        tracks[name] = (np.asarray(gp), np.asarray(gy), np.asarray(sp))
    return tracks


def _run_eval_in_process(eval_paths: dict, quintic_tracks: dict | None,
                         task: str, rollout_time: float, device: str,
                         no_view: bool, viz_dir: Path | None,
                         combined_mp4: Path | None = None,
                         mp4_width: int = 720, mp4_height: int = 540,
                         mp4_fps: int = 30) -> int:
    """Import 27_eval_rl_hex_mtrl_v4 in-process, monkey-patch its
    ``_task_config`` to return our quintic tracks (when available), then
    invoke its ``main()`` with the right sys.argv.

    Patching the eval module's binding directly (not the v4 train module's)
    is what matters: ``_rollout`` reads ``_task_config`` via module globals
    on the eval module itself.
    """
    spec = importlib.util.spec_from_file_location("mtrl_eval_v4", _EVAL_SCRIPT)
    eval_mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["mtrl_eval_v4"] = eval_mod
    spec.loader.exec_module(eval_mod)  # type: ignore[union-attr]

    if quintic_tracks is not None:
        orig_task_config = eval_mod._task_config

        def _patched(name, density=1):
            if name in quintic_tracks:
                return quintic_tracks[name]
            return orig_task_config(name, density=density)

        eval_mod._task_config = _patched

    argv = [
        str(_EVAL_SCRIPT),
        "--policy",         str(eval_paths["policy"]),
        "--blueprint-json", str(eval_paths["blueprint"]),
        "--rollout-time",   str(rollout_time),
        "--gate-density",   str(GATE_DENSITY),
        "--device",         device,
        "--task",           task,
    ]
    if "vecnormalize" in eval_paths:
        argv += ["--vecnormalize", str(eval_paths["vecnormalize"])]
    if no_view:
        argv += ["--no-view"]
        if viz_dir is not None:
            argv += ["--out-dir", str(viz_dir)]
    if combined_mp4 is not None:
        argv += [
            "--combined-mp4", str(combined_mp4),
            "--mp4-width",  str(mp4_width),
            "--mp4-height", str(mp4_height),
            "--mp4-fps",    str(mp4_fps),
        ]

    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = eval_mod.main()
    finally:
        sys.argv = old_argv
    return int(rc) if rc is not None else 0


def main():
    parser = argparse.ArgumentParser(
        description="Replay a quintic-EA v4 individual from its database"
    )
    parser.add_argument("--db", required=True,
                        help="Path to database_*.db written by 31_drone_evo_v4_quintic.py")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--rank", default="best",
                        help="'best' | 'worst' | 'median' | int offset. Default: best.")
    parser.add_argument("--task", default="all",
                        help="figure8 | slalom | shuttle-run | hover | all. Default: all.")
    parser.add_argument("--rollout-time", type=float, default=15.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-view", action="store_true",
                        help="Render MP4s into <out-dir>/viz/ instead of opening the viewer.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--combined-mp4", default=None,
                        help="Write a single MP4 with all selected tasks "
                             "concatenated. If a relative path is given, it is "
                             "resolved under <out-dir>.")
    parser.add_argument("--mp4-width",  type=int, default=720)
    parser.add_argument("--mp4-height", type=int, default=540)
    parser.add_argument("--mp4-fps",    type=int, default=30)
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

    quintic_tracks = _load_tracks(paths["tracks"]) if "tracks" in paths else None
    if quintic_tracks is not None:
        console.log(
            f"Loaded quintic tracks for {list(quintic_tracks.keys())} — "
            f"viewer will use the exact gates this candidate was trained on."
        )

    viz_dir = (out_dir / "viz") if args.no_view else None

    combined_mp4 = None
    if args.combined_mp4:
        cm = Path(args.combined_mp4)
        combined_mp4 = cm if cm.is_absolute() else (out_dir / cm)

    rc = _run_eval_in_process(
        eval_paths=paths,
        quintic_tracks=quintic_tracks,
        task=args.task,
        rollout_time=args.rollout_time,
        device=args.device,
        no_view=args.no_view,
        viz_dir=viz_dir,
        combined_mp4=combined_mp4,
        mp4_width=args.mp4_width,
        mp4_height=args.mp4_height,
        mp4_fps=args.mp4_fps,
    )
    if rc != 0:
        console.log(f"[red]Eval main() returned non-zero: {rc}[/red]")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
