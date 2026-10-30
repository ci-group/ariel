"""Probe how many PPO steps the v4 warm-start needs to adapt to ONE
evolved hexacopter blueprint.

Takes a 31_drone_evo_v4_quintic.py database, picks one individual
(best by default), reuses its exact quintic tracks + vecnorm + the
v4 warm-start, and runs a longer PPO fine-tune (default 10M steps)
with periodic evaluation across all four tasks.  Result: a
gates/s-vs-PPO-steps curve so we can see where (or whether) the
policy actually starts adapting.

Usage
-----
    uv run examples/spear/33_probe_warmstart_adaptation.py \\
        --db __data__/drone_evo_quintic_v4/<RUN>/database_<RUN_ID>.db

    # Override which individual:
    --rank best | median | worst | <int offset>
    --id <individual_id>

    # Override budgets:
    --ppo-steps 10000000  --eval-every 500000

NOTE: do NOT add ``from __future__ import annotations`` — ariel's
@EAOperation decorator inspects real annotation objects at decoration time.
"""

import argparse
import base64
import csv
import importlib.util
import io
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Reuse v4 multi-task env + helpers (same trick as 31_drone_evo_v4_quintic).
_V4_TRAIN_PATH = Path(__file__).with_name("27_train_rl_hex_mtrl_v4.py")
_v4_spec = importlib.util.spec_from_file_location("mtrl_train_v4", _V4_TRAIN_PATH)
_v4 = importlib.util.module_from_spec(_v4_spec)  # type: ignore[arg-type]
sys.modules["mtrl_train_v4"] = _v4
_v4_spec.loader.exec_module(_v4)  # type: ignore[union-attr]
MTRLActorCriticPolicy = _v4.MTRLActorCriticPolicy
MultiTaskHexVecEnv    = _v4.MultiTaskHexVecEnv
EntCoefAnneal         = _v4.EntCoefAnneal
_eval_per_task        = _v4._eval_per_task
TASK_NAMES            = _v4.TASK_NAMES
NUM_TASKS             = _v4.NUM_TASKS


# ─── CONFIG (mirrors 31_drone_evo_v4_quintic) ────────────────────────────────

GATE_DENSITY     = 3
PROP_SIZE        = 2

PPO_NUM_ENVS     = 128
PPO_N_STEPS      = 4096
PPO_N_EPOCHS     = 10
PPO_GAMMA        = 0.99
PPO_GAE_LAMBDA   = 0.95
PPO_LR           = 3e-4
PPO_CLIP_RANGE   = 0.2
PPO_ENT_START    = 0.005
PPO_ENT_END      = 1e-4
PPO_MAX_GRAD_NORM = 0.5

EVAL_STEPS       = 1500

WARMSTART_DIR    = "__data__/spear_rl_hex_mtrl_v4/20260616_163031"
QUINTIC_TASKS    = ("figure8", "slalom", "shuttle-run")


# ─── Quintic env reused from 31 ──────────────────────────────────────────────

class QuinticMultiTaskHexVecEnv(MultiTaskHexVecEnv):
    def __init__(self, *, quintic_tracks: dict, **kwargs):
        self._quintic_tracks = quintic_tracks
        orig_task_config = _v4._task_config

        def _patched(name, density=1):
            if name in quintic_tracks:
                return quintic_tracks[name]
            return orig_task_config(name, density=density)

        _v4._task_config = _patched
        try:
            super().__init__(**kwargs)
        finally:
            _v4._task_config = orig_task_config


# ─── DB row resolution (mirrors 32_replay_v4_quintic) ────────────────────────

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
        order, offset = "DESC", 0
    elif rank == "worst":
        order, offset = "ASC", 0
    elif rank == "median":
        cur.execute("SELECT COUNT(*) FROM individual WHERE fitness_ IS NOT NULL")
        n = cur.fetchone()[0]
        if n == 0:
            raise SystemExit("No evaluated individuals in DB.")
        order, offset = "DESC", n // 2
    else:
        try:
            offset = int(rank)
        except ValueError as exc:
            raise SystemExit(f"--rank must be best/worst/median/int, got {rank!r}") from exc
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


def _decode_tracks(tags: dict) -> dict | None:
    if "quintic_tracks_b64" not in tags:
        return None
    buf = io.BytesIO(base64.b64decode(tags["quintic_tracks_b64"]))
    data = np.load(buf)
    tracks: dict = {}
    for name in QUINTIC_TASKS:
        tracks[name] = (
            np.asarray(data[f"{name}__gate_pos"]),
            np.asarray(data[f"{name}__gate_yaw"]),
            np.asarray(data[f"{name}__start_pos"]),
        )
    return tracks


# ─── Periodic eval callback ──────────────────────────────────────────────────

class PeriodicEval(BaseCallback):
    """Run _eval_per_task every ``eval_every`` env steps and append a row to ``log_rows``."""

    def __init__(self, eval_every: int, eval_steps: int, log_rows: list, console: Console):
        super().__init__()
        self.eval_every = int(eval_every)
        self.eval_steps = int(eval_steps)
        self.log_rows   = log_rows
        self.console    = console
        self._next_at   = self.eval_every

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_at:
            return True
        self._next_at = self.num_timesteps + self.eval_every
        env = self.model.get_env()  # the VecNormalize we trained against
        was_training = env.training
        env.training = False  # freeze obs-norm stats during eval
        try:
            ep_r, ep_g, total_g, live_steps = _eval_per_task(
                env, self.model, n_steps=self.eval_steps,
            )
        finally:
            env.training = was_training

        elapsed_s = float(live_steps.sum()) * 0.01
        gps_total = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0
        per_task = {}
        for ti, name in enumerate(TASK_NAMES):
            tsec = float(live_steps[ti]) * 0.01
            per_task[name] = (float(total_g[ti]) / tsec) if tsec > 0 else 0.0

        row = {"steps": int(self.num_timesteps), "gates_per_s_total": gps_total, **per_task}
        self.log_rows.append(row)
        self.console.log(
            f"[cyan]eval @ {self.num_timesteps:>10,} steps[/cyan]  "
            f"total={gps_total:.4f}  " +
            "  ".join(f"{n}={per_task[n]:.3f}" for n in TASK_NAMES)
        )
        return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Probe warmstart-policy adaptation on a single evolved blueprint"
    )
    parser.add_argument("--db", required=True,
                        help="database_*.db written by 31_drone_evo_v4_quintic.py")
    parser.add_argument("--id", type=int, default=None)
    parser.add_argument("--rank", default="best")
    parser.add_argument("--ppo-steps", type=int, default=10_000_000)
    parser.add_argument("--eval-every", type=int, default=500_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    console = Console()

    if PPO_NUM_ENVS % NUM_TASKS != 0:
        raise SystemExit(f"PPO_NUM_ENVS ({PPO_NUM_ENVS}) % NUM_TASKS ({NUM_TASKS}) != 0")

    warm_dir     = Path(WARMSTART_DIR)
    warm_policy  = warm_dir / "policy.zip"
    warm_vecnorm = warm_dir / "vecnormalize.pkl"
    if not warm_policy.exists() or not warm_vecnorm.exists():
        raise SystemExit(f"Warmstart dir missing files: {warm_dir}")

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = sqlite3.connect(db_path)
    try:
        row_id, genotype_json, tags, fitness = _resolve_row(con, args.id, args.rank)
    finally:
        con.close()

    label = f"id={row_id}" if args.id is not None else f"id={row_id} (rank={args.rank})"
    console.log(f"Selected {label}  fitness_at_3M={fitness:.4f}")

    genotype = json.loads(genotype_json)
    genome   = deserialize_genome(genotype)
    bp       = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)
    propellers = blueprint_to_propellers(bp, convention="ned")
    n_motors = len(propellers)
    console.log(f"Motors: {n_motors}")
    if n_motors != 6:
        raise SystemExit("This probe expects 6-motor candidates (v4 policy is locked to 6).")

    tracks = _decode_tracks(tags)
    if tracks is None:
        raise SystemExit(
            "Selected individual has no quintic_tracks_b64 tag — pick another row."
        )
    console.log(f"Reusing trained quintic tracks for: {list(tracks.keys())}")

    out_dir = Path(args.out_dir) if args.out_dir else (
        db_path.parent / f"probe_id{row_id}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    console.log(f"Out dir → {out_dir}")

    raw_env = QuinticMultiTaskHexVecEnv(
        quintic_tracks=tracks,
        propellers=propellers,
        num_envs=PPO_NUM_ENVS,
        device=args.device,
        dt=0.01,
        seed=args.seed,
        gate_density=GATE_DENSITY,
    )
    env = VecNormalize.load(str(warm_vecnorm), raw_env)
    env.training = True
    env.norm_reward = False

    rollout_size = PPO_N_STEPS * PPO_NUM_ENVS
    batch_size   = rollout_size // 8

    model = PPO.load(
        str(warm_policy), env=env, device=args.device,
        custom_objects={
            "policy_class":  MTRLActorCriticPolicy,
            "n_steps":       PPO_N_STEPS,
            "batch_size":    batch_size,
            "n_epochs":      PPO_N_EPOCHS,
            "learning_rate": PPO_LR,
            "clip_range":    PPO_CLIP_RANGE,
            "ent_coef":      PPO_ENT_START,
            "gamma":         PPO_GAMMA,
            "gae_lambda":    PPO_GAE_LAMBDA,
            "max_grad_norm": PPO_MAX_GRAD_NORM,
        },
    )
    model.ent_coef = PPO_ENT_START
    model.policy.optimizer = torch.optim.Adam(model.policy.parameters(), lr=PPO_LR)
    model.lr_schedule = lambda _p: PPO_LR

    log_rows: list[dict[str, Any]] = []

    # Step-0 eval (warmstart baseline before any updates).
    env.training = False
    try:
        ep_r, ep_g, total_g, live_steps = _eval_per_task(env, model, n_steps=EVAL_STEPS)
    finally:
        env.training = True
    elapsed_s = float(live_steps.sum()) * 0.01
    gps_total = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0
    per_task = {}
    for ti, name in enumerate(TASK_NAMES):
        tsec = float(live_steps[ti]) * 0.01
        per_task[name] = (float(total_g[ti]) / tsec) if tsec > 0 else 0.0
    log_rows.append({"steps": 0, "gates_per_s_total": gps_total, **per_task})
    console.log(
        f"[magenta]warmstart baseline[/magenta]  total={gps_total:.4f}  " +
        "  ".join(f"{n}={per_task[n]:.3f}" for n in TASK_NAMES)
    )

    callbacks = [
        EntCoefAnneal(PPO_ENT_START, PPO_ENT_END, args.ppo_steps),
        PeriodicEval(args.eval_every, EVAL_STEPS, log_rows, console),
    ]

    t0 = time.time()
    model.learn(total_timesteps=args.ppo_steps, callback=callbacks, progress_bar=False)
    console.log(f"PPO done in {time.time() - t0:.1f}s")

    # Final eval (in case the last periodic eval landed mid-rollout).
    env.training = False
    try:
        ep_r, ep_g, total_g, live_steps = _eval_per_task(env, model, n_steps=EVAL_STEPS)
    finally:
        env.training = True
    elapsed_s = float(live_steps.sum()) * 0.01
    gps_total = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0
    per_task = {}
    for ti, name in enumerate(TASK_NAMES):
        tsec = float(live_steps[ti]) * 0.01
        per_task[name] = (float(total_g[ti]) / tsec) if tsec > 0 else 0.0
    log_rows.append({"steps": int(args.ppo_steps), "gates_per_s_total": gps_total, **per_task})
    console.log(
        f"[bold green]final[/bold green]  total={gps_total:.4f}  " +
        "  ".join(f"{n}={per_task[n]:.3f}" for n in TASK_NAMES)
    )

    # Persist curve + artifacts.
    csv_path = out_dir / "adaptation_curve.csv"
    with csv_path.open("w", newline="") as f:
        fieldnames = ["steps", "gates_per_s_total", *TASK_NAMES]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in log_rows:
            w.writerow(r)
    console.log(f"Curve → {csv_path}")

    model.save(str(out_dir / "policy_final.zip"))
    env.save(str(out_dir / "vecnormalize_final.pkl"))
    bp.save_json(out_dir / "blueprint.json")
    with (out_dir / "config.json").open("w") as f:
        json.dump({
            "db": str(db_path), "row_id": row_id, "rank": args.rank, "id": args.id,
            "ppo_steps": args.ppo_steps, "eval_every": args.eval_every,
            "ppo_num_envs": PPO_NUM_ENVS, "ppo_n_steps": PPO_N_STEPS,
            "ent_start": PPO_ENT_START, "ent_end": PPO_ENT_END,
            "lr": PPO_LR, "device": args.device, "seed": args.seed,
            "warmstart_dir": WARMSTART_DIR,
        }, f, indent=2)
    console.log(f"Artifacts → {out_dir}")


if __name__ == "__main__":
    main()
