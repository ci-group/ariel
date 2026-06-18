"""Drone morphology evolution with per-candidate multi-task PPO fine-tune,
trained on quintic-generated gate tracks (warm-started from the v4 MTRL
generalist policy).

Mirrors ``29_drone_evo_v4_warmstart.py`` but replaces the three racing-task
gate configs (figure8 / slalom / shuttle-run) with tracks sampled from the
LTU quintic goal generator (see ``goal_generator_ltu/polynomial_goal_generator``).
Each task slot keeps its v4 task-encoder identity (so the warm-started
weights stay meaningful), but the gates underneath it are quintic-sampled.
The 'hover' task is left unchanged because it needs a stationary target.

Per-candidate inner loop:
    * QuinticMultiTaskHexVecEnv with this candidate's propellers and a
      candidate-specific quintic seed (so each morphology fine-tunes on a
      different set of tracks — robust to seed luck).
    * VecNormalize obs-norm loaded from the v4 base run.
    * PPO weights loaded from the v4 base policy.
    * EntCoefAnneal callback during ``model.learn``.
    * _eval_per_task across all four tasks for fitness.

Fitness = total gate-passes / second across all four tasks (same as v4
warmstart). Each evaluated row stores policy.zip + vecnormalize.pkl +
the exact per-task quintic gates (b64 npz) so ``32_replay_v4_quintic.py``
can reconstruct the env this candidate was trained on.

Quick start
-----------
    uv run examples/spear/31_drone_evo_v4_quintic.py
    uv run examples/spear/31_drone_evo_v4_quintic.py --device cuda:0 --seed 7

NOTE: do NOT add ``from __future__ import annotations`` — ariel's
@EAOperation decorator inspects real annotation objects at decoration time.
"""

import argparse
import base64
import gc
import importlib.util
import io
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

_REPO_ROOT = Path(__file__).resolve().parents[2]

from ariel.body_phenotypes.drone import (
    crossover_drones,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.body_phenotypes.drone.genome import deserialize_genome
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)

# Reuse v4's MultiTaskHexVecEnv + helpers. We monkey-patch _task_config on
# this module right before constructing a vec-env so the racing-task slots
# get quintic gates while hover keeps its stationary target.
_V4_TRAIN_PATH = Path(__file__).with_name("27_train_rl_hex_mtrl_v4.py")
_v4_spec = importlib.util.spec_from_file_location("mtrl_train_v4", _V4_TRAIN_PATH)
_v4 = importlib.util.module_from_spec(_v4_spec)  # type: ignore[arg-type]
sys.modules["mtrl_train_v4"] = _v4
_v4_spec.loader.exec_module(_v4)  # type: ignore[union-attr]
MTRLActorCriticPolicy = _v4.MTRLActorCriticPolicy
MultiTaskHexVecEnv    = _v4.MultiTaskHexVecEnv
EntCoefAnneal         = _v4.EntCoefAnneal
_eval_per_task        = _v4._eval_per_task
_format_eval          = _v4._format_eval
TASK_NAMES            = _v4.TASK_NAMES
NUM_TASKS             = _v4.NUM_TASKS

# Quintic goal generator (LTU)
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLI  (run-level overrides — not the main config surface)
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Drone morphology evolution + multi-task PPO fine-tune on "
                "quintic gate tracks, warm-started from v4 MTRL"
)
parser.add_argument("--device",  default="cpu",
                    help="Torch device: 'cpu' or 'cuda:0'  (default: cpu)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_quintic_v4/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE  = 1
EA_GENS      = 1

# v4 policy is locked to 6 motors. Keep min == max == 6.
N_ARMS_MIN = 6
N_ARMS_MAX = 6

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [0, 1],
])

APPEND_ARM_CHANCE   = 0.0
BILATERAL_SYMMETRY  = None
REPAIR              = False
MUTATION_SCALES     = None
CUSTOM_GENERATION_OPS = None


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — MULTI-TASK ENV  (mirrors 27_train_rl_hex_mtrl_v4 config)
# ═════════════════════════════════════════════════════════════════════════════

GATE_DENSITY = 3
PROP_SIZE    = 2

# Disable v4's tilt-based episode termination. Evolved morphologies can
# spin out under the warm-started hexacopter policy; resetting on tilt
# kills fine-tune episodes before the policy can adapt.
_v4.TILT_TERMINATE_COS = -1.0


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — PPO / RL  (smaller budget than base — this is a fine-tune)
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS         = 10_000_000
PPO_NUM_ENVS      = 128
PPO_N_STEPS       = 4096
PPO_N_EPOCHS      = 10
PPO_GAMMA         = 0.99
PPO_GAE_LAMBDA    = 0.95
PPO_LR            = 3e-4
PPO_CLIP_RANGE    = 0.2
PPO_ENT_START     = 0.005
PPO_ENT_END       = 1e-4
PPO_MAX_GRAD_NORM = 0.5

EVAL_STEPS = 1500

# Periodic in-training gate-pass probe. Set MID_EVAL_EVERY=0 to disable.
MID_EVAL_EVERY = 524_288   # env-steps between probes (≈ one PPO rollout)
MID_EVAL_STEPS = 1500


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 4 — WARM START  (v4 policy + obs-norm stats)
# ═════════════════════════════════════════════════════════════════════════════

WARMSTART_DIR       = "__data__/spear_rl_hex_mtrl_v4/20260616_163031"
RESET_VECNORM_TRAIN = True
WARMSTART_REQUIRED  = True


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 5 — QUINTIC GATE GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

QUINTIC_COEFFS_PATH = str(
    _REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator" / "quintic_coeffs.npy"
)
# Racing tasks that get their gates replaced by quintic tracks. Hover is
# intentionally left out — it needs a stationary target.
QUINTIC_TASKS       = ("figure8", "slalom", "shuttle-run")
QUINTIC_N_GATES     = 15
QUINTIC_PATH_SCALE  = 5.0
QUINTIC_Z_HEIGHT    = -1.5


# ─────────────────────────────────────────────────────────────────────────────
# Below this line: experiment logic — edit only if you know what you're doing
# ─────────────────────────────────────────────────────────────────────────────

DATA   = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

_warm_dir = Path(WARMSTART_DIR)
_warm_policy = _warm_dir / "policy.zip"
_warm_vecnorm = _warm_dir / "vecnormalize.pkl"
if WARMSTART_REQUIRED and (not _warm_policy.exists() or not _warm_vecnorm.exists()):
    console.log(
        f"[red]Warm-start dir missing required files: {_warm_dir}\n"
        f"  need policy.zip ({'OK' if _warm_policy.exists() else 'MISSING'}) "
        f"and vecnormalize.pkl ({'OK' if _warm_vecnorm.exists() else 'MISSING'})[/red]"
    )
    raise SystemExit(1)

if PPO_NUM_ENVS % NUM_TASKS != 0:
    raise SystemExit(
        f"PPO_NUM_ENVS ({PPO_NUM_ENVS}) must be divisible by NUM_TASKS ({NUM_TASKS})"
    )

QUINTIC_COEFFS = np.load(QUINTIC_COEFFS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Quintic gate sampler  (copied + reduced from 7_drone_evo_rl_quintic.py)
# ─────────────────────────────────────────────────────────────────────────────

def _quintic_to_gates(
    coeffs: np.ndarray,
    n_gates: int,
    scale: float,
    z: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one quintic track, return (gate_pos N×3, gate_yaw N).

    Yaw comes from the planner's smooth tangent, not the gate-to-gate
    difference vector (which is noisy for close-spaced consecutive gates).
    """
    OVERSAMPLE = max(n_gates * 8, 64)
    MIN_SPACING = 0.3
    paths, yaws_dense_arr = generate_paths_from_coefficients(
        coeffs, num_generate=1, steps=OVERSAMPLE, seed=seed, clip_range=(-1.0, 1.0),
    )
    xy_dense  = paths[0] * scale
    yaw_dense = yaws_dense_arr[0]

    kept = [0]
    for idx in range(1, len(xy_dense)):
        if np.linalg.norm(xy_dense[idx] - xy_dense[kept[-1]]) >= MIN_SPACING:
            kept.append(idx)
        if len(kept) == n_gates:
            break
    if len(kept) < n_gates:
        kept = list(np.linspace(0, len(xy_dense) - 1, n_gates, dtype=int))
    xy = xy_dense[kept]

    gate_pos = np.column_stack(
        [xy, np.full(len(kept), z, dtype=np.float64)]
    ).astype(np.float64)
    gate_yaw = yaw_dense[kept].astype(np.float64)
    return gate_pos, gate_yaw


def _quintic_track_set(base_seed: int) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return {task_name: (gate_pos, gate_yaw, starting_pos)} for each racing
    task, plus the unchanged hover config. Each racing task uses a distinct
    sub-seed so the policy sees three different tracks per fine-tune.
    """
    tracks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for ti, name in enumerate(QUINTIC_TASKS):
        gp, gy = _quintic_to_gates(
            QUINTIC_COEFFS, QUINTIC_N_GATES, QUINTIC_PATH_SCALE,
            QUINTIC_Z_HEIGHT, seed=base_seed * 100 + ti,
        )
        # Start 1m "behind" the first gate (in -y direction), height matched.
        start_pos = (gp[0] + np.array([0.0, -1.0, 0.0])).astype(np.float64)
        tracks[name] = (gp, gy, start_pos)
    return tracks


def _serialize_tracks_b64(tracks: dict) -> str:
    """Save tracks as an in-memory npz, return base64 string."""
    arrs = {}
    for name, (gp, gy, sp) in tracks.items():
        # Keys can't contain hyphens cleanly in npz access patterns,
        # but np.savez/load tolerate them. Use stable join anyway.
        arrs[f"{name}__gate_pos"] = gp
        arrs[f"{name}__gate_yaw"] = gy
        arrs[f"{name}__start_pos"] = sp
    buf = io.BytesIO()
    np.savez(buf, **arrs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─────────────────────────────────────────────────────────────────────────────
# QuinticMultiTaskHexVecEnv — same VecEnv as v4, but with patched gate source
# ─────────────────────────────────────────────────────────────────────────────

class QuinticMultiTaskHexVecEnv(MultiTaskHexVecEnv):
    """v4's MultiTaskHexVecEnv with the racing tasks' gate configs replaced
    by quintic-sampled tracks. Achieved by temporarily monkey-patching
    ``_v4._task_config`` for the duration of ``super().__init__``."""

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


# ─────────────────────────────────────────────────────────────────────────────
# Genome handler  (built from CONFIG 1)
# ─────────────────────────────────────────────────────────────────────────────

genome_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(N_ARMS_MIN, N_ARMS_MAX),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=APPEND_ARM_CHANCE,
    bilateral_plane_for_symmetry=BILATERAL_SYMMETRY,
    repair=REPAIR,
    mutation_scales_percentage=MUTATION_SCALES,
    rnd=np.random.default_rng(args.seed),
)


# ─────────────────────────────────────────────────────────────────────────────
# PPO training + evaluation  (mirrors v4 fine-tune loop with quintic env)
# ─────────────────────────────────────────────────────────────────────────────

def _model_to_b64(model: PPO) -> str:
    tmp = Path(tempfile.mktemp(suffix=".zip"))
    try:
        model.save(str(tmp))
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


def _vecnorm_to_b64(env: VecNormalize) -> str:
    tmp = Path(tempfile.mktemp(suffix=".pkl"))
    try:
        env.save(str(tmp))
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


cfg: dict[str, Any] = {
    "ppo_steps":  PPO_STEPS,
    "num_envs":   PPO_NUM_ENVS,
    "eval_steps": EVAL_STEPS,
    "prop_size":  PROP_SIZE,
    "device":     args.device,
}

# Per-candidate sub-seed counter — gives every fine-tune its own track set.
_quintic_seed_counter = {"n": 0}


class MidTrainingGateEval(BaseCallback):
    """Run _eval_per_task every ``eval_every`` env steps and print gates/s
    so we can watch adaptation progress during PPO fine-tune."""

    def __init__(self, eval_every: int, eval_steps: int, console: Console):
        super().__init__()
        self.eval_every = int(eval_every)
        self.eval_steps = int(eval_steps)
        self.console    = console
        self._next_at   = self.eval_every

    def _on_step(self) -> bool:
        if self.eval_every <= 0 or self.num_timesteps < self._next_at:
            return True
        self._next_at = self.num_timesteps + self.eval_every
        env = self.model.get_env()
        was_training = env.training
        env.training = False
        try:
            _ep_r, _ep_g, total_g, live_steps = _eval_per_task(
                env, self.model, n_steps=self.eval_steps,
            )
        finally:
            env.training = was_training

        elapsed_s = float(live_steps.sum()) * 0.01
        gps_total = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0
        per_task_parts = []
        for ti, name in enumerate(TASK_NAMES):
            tsec = float(live_steps[ti]) * 0.01
            gps = (float(total_g[ti]) / tsec) if tsec > 0 else 0.0
            per_task_parts.append(f"{name}={int(total_g[ti])}g/{gps:.3f}gps")
        self.console.log(
            f"[cyan]gate-probe @ {self.num_timesteps:>10,} steps[/cyan]  "
            f"total={int(total_g.sum())}g / {gps_total:.4f} gps  "
            + "  ".join(per_task_parts)
        )
        return True


def _train_and_eval(genotype: dict, cfg: dict) -> tuple[float, Any, str | None, str | None, int]:
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, None, None, 0

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), None, None, None, 0

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, None, None, 0

    n_motors = len(propellers)
    if n_motors != 6:
        return float("-inf"), None, None, None, n_motors

    _quintic_seed_counter["n"] += 1
    quintic_seed = args.seed * 10_000 + _quintic_seed_counter["n"]
    tracks = _quintic_track_set(quintic_seed)
    tracks_b64 = _serialize_tracks_b64(tracks)

    raw_env = QuinticMultiTaskHexVecEnv(
        quintic_tracks=tracks,
        propellers=propellers,
        num_envs=cfg["num_envs"],
        device=cfg["device"],
        dt=0.01,
        seed=args.seed,
        gate_density=GATE_DENSITY,
    )

    env = VecNormalize.load(str(_warm_vecnorm), raw_env)
    env.training = bool(RESET_VECNORM_TRAIN)
    env.norm_reward = False

    rollout_size = PPO_N_STEPS * cfg["num_envs"]
    batch_size   = rollout_size // 8

    model = PPO.load(
        str(_warm_policy), env=env, device=cfg["device"],
        custom_objects={
            "policy_class":   MTRLActorCriticPolicy,
            "n_steps":        PPO_N_STEPS,
            "batch_size":     batch_size,
            "n_epochs":       PPO_N_EPOCHS,
            "learning_rate":  PPO_LR,
            "clip_range":     PPO_CLIP_RANGE,
            "ent_coef":       PPO_ENT_START,
            "gamma":          PPO_GAMMA,
            "gae_lambda":     PPO_GAE_LAMBDA,
            "max_grad_norm":  PPO_MAX_GRAD_NORM,
        },
    )
    model.ent_coef = PPO_ENT_START

    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(), lr=PPO_LR,
    )
    model.lr_schedule = lambda _progress_remaining: PPO_LR

    _callbacks: list = [EntCoefAnneal(PPO_ENT_START, PPO_ENT_END, cfg["ppo_steps"])]
    if MID_EVAL_EVERY > 0:
        _callbacks.append(MidTrainingGateEval(MID_EVAL_EVERY, MID_EVAL_STEPS, console))

    model.learn(
        total_timesteps=cfg["ppo_steps"],
        callback=_callbacks,
        progress_bar=False,
    )

    _ep_r, _ep_g, total_g, live_steps = _eval_per_task(
        env, model, n_steps=cfg["eval_steps"],
    )

    elapsed_s = float(live_steps.sum()) * 0.01
    fitness   = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0

    vecnorm_b64 = _vecnorm_to_b64(env)

    try:
        env.close()
    except Exception:
        pass
    del env, raw_env
    if cfg["device"].startswith("cuda"):
        torch.cuda.empty_cache()

    return fitness, model, vecnorm_b64, tracks_b64, n_motors


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_ppo(population: Population) -> Population:
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="Quintic MTRL fine-tune…", console=console)):
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.rule(f"[cyan]candidate {display_id} — blueprint #{_i + 1}/{len(to_eval)}")

        t0 = time.time()
        fit, model, vecnorm_b64, tracks_b64, n_motors = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        if model is not None:
            ind.tags["policy_b64"] = _model_to_b64(model)
        if vecnorm_b64 is not None:
            ind.tags["vecnorm_b64"] = vecnorm_b64
        if tracks_b64 is not None:
            ind.tags["quintic_tracks_b64"] = tracks_b64

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        console.log(
            f"  id={display_id}  motors={n_motors}  "
            f"gates/s={fit:.3f}  ({time.time()-t0:.1f}s)"
        )

    finite = [ind.fitness_ for ind in population.alive.evaluated if np.isfinite(ind.fitness_)]  # type: ignore[arg-type]
    if finite:
        console.log(f"    batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}")
    return population


eval_op = EAOperation(evaluate_with_ppo)


# ─────────────────────────────────────────────────────────────────────────────
# Build generation pipeline
# ─────────────────────────────────────────────────────────────────────────────

if CUSTOM_GENERATION_OPS is not None:
    generation_ops = [eval_op if op is None else op for op in CUSTOM_GENERATION_OPS]
else:
    generation_ops = [
        parent_tag(n=EA_POP_SIZE),
        crossover_drones(template_handler=genome_handler),
        mutate_drones(template_handler=genome_handler),
        eval_op,
        truncation_select(n=EA_POP_SIZE),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

console.rule("[bold blue]Drone Body Evolution + Multi-Task PPO on Quintic Tracks (v4 warm-start)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"ppo_steps={PPO_STEPS:,}  num_envs={PPO_NUM_ENVS}  "
    f"tasks={list(TASK_NAMES)}  quintic_tasks={list(QUINTIC_TASKS)}  "
    f"quintic_n_gates={QUINTIC_N_GATES}  quintic_scale={QUINTIC_PATH_SCALE}  "
    f"warm_dir={WARMSTART_DIR}  device={args.device}"
)
console.log(f"DB → {DB_PATH}")

initial_pop = Population([Individual() for _ in range(EA_POP_SIZE)])
init_op     = initialize_drones(template_handler=genome_handler)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if finite0:
    console.log(f"Initial — max={max(finite0):.4f}  mean={np.mean(finite0):.4f}")

console.rule("[bold green]Evolving")
ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=EA_GENS,
    db_file_path=DB_PATH,
    db_handling="delete",
)
ea.run()


# ─────────────────────────────────────────────────────────────────────────────
# Save best individual
# ─────────────────────────────────────────────────────────────────────────────

ea.fetch_population(only_alive=False, requires_eval=False)
finite = [ind for ind in ea.population if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if not finite:
    console.log("[red]No valid individuals — aborting.[/red]")
    raise SystemExit(1)

best        = sorted(finite, key=lambda ind: ind.fitness_, reverse=True)[0]
best_genome = deserialize_genome(best.genotype)
best_bp     = spherical_angular_to_blueprint(best_genome.arms, propsize=PROP_SIZE)

bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"Best blueprint → {bp_path}")

if "policy_b64" in best.tags:
    policy_path = DATA / f"best_policy_{RUN_ID}.zip"
    policy_path.write_bytes(base64.b64decode(best.tags["policy_b64"]))
    console.log(f"Best policy    → {policy_path}")

if "vecnorm_b64" in best.tags:
    vn_path = DATA / f"best_vecnormalize_{RUN_ID}.pkl"
    vn_path.write_bytes(base64.b64decode(best.tags["vecnorm_b64"]))
    console.log(f"Best vecnorm   → {vn_path}")

if "quintic_tracks_b64" in best.tags:
    tracks_path = DATA / f"best_quintic_tracks_{RUN_ID}.npz"
    tracks_path.write_bytes(base64.b64decode(best.tags["quintic_tracks_b64"]))
    console.log(f"Best tracks    → {tracks_path}")

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best fitness (gates/s across 4 tasks): {best.fitness_:.4f}  DB → {DB_PATH}"
)
console.log(
    f"\nTo visualise the best individual (or any other), use the replay script:\n"
    f"  uv run examples/spear/32_replay_v4_quintic.py --db {DB_PATH}\n"
    f"  uv run examples/spear/32_replay_v4_quintic.py --db {DB_PATH} --rank best\n"
    f"  uv run examples/spear/32_replay_v4_quintic.py --db {DB_PATH} --id <individual_id>"
)
