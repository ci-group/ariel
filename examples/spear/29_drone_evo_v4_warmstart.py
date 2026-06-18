"""Drone morphology evolution with per-candidate multi-task PPO fine-tune.

Per-candidate inner loop mirrors ``27_train_rl_hex_mtrl_v4.py``: each
morphology is dropped into a 4-task ``MultiTaskHexVecEnv`` (figure8,
slalom, shuttle-run, hover) seeded from the v4 actor-critic + v4
VecNormalize obs-norm stats, then fine-tuned for ``PPO_STEPS`` with the
same entropy-annealing callback and per-task reward normalisation. Fitness
is aggregated across all four tasks (gates per second over the full
evaluation rollout — robust to whether episodes terminate or time out).

EA scaffolding follows ``11_drone_evo_curvature.py``: CONFIG-clustered,
no inline comments inside CONFIG blocks.

The v4 actor-critic is locked to 6 motors. Candidates with a different
motor count are rejected; keep N_ARMS_MIN == N_ARMS_MAX == 6.

Quick start
-----------
    uv run examples/spear/29_drone_evo_v4_warmstart.py
    uv run examples/spear/29_drone_evo_v4_warmstart.py --device cuda:0 --seed 7

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports  (do not edit)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import base64
import gc
import importlib.util
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

# Re-use v4's MultiTaskHexVecEnv, evaluator, entropy-annealing callback and
# custom MTRL actor-critic policy. Loading the module also pulls in all
# v4-side env shaping constants so the EA fine-tune matches base training.
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


# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLI  (run-level overrides — not the main config surface)
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Drone morphology evolution + multi-task PPO fine-tune, "
                "warm-started from v4 MTRL"
)
parser.add_argument("--device",  default="cpu",
                    help="Torch device: 'cpu' or 'cuda:0'  (default: cpu)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_warmstart_v4/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE  = 2
EA_GENS      = 2

# v4 policy is locked to 6 motors. Keep min == max == 6.
N_ARMS_MIN = 6
N_ARMS_MAX = 6

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm length
    [-np.pi, np.pi],          # arm azimuth
    [-np.pi / 2, np.pi / 2],  # arm elevation
    [-np.pi, np.pi],          # motor disc azimuth
    [-np.pi, np.pi],          # motor disc pitch
    [0, 1],                   # spin direction (binary)
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

# Final per-task evaluation rollout length (≥ TorchDroneGateEnv.max_steps so
# every per-task env has a chance to time out at least once).
EVAL_STEPS = 1500


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 4 — WARM START  (v4 policy + obs-norm stats)
# ═════════════════════════════════════════════════════════════════════════════

WARMSTART_DIR       = "__data__/spear_rl_hex_mtrl_v4/20260616_163031"
RESET_VECNORM_TRAIN = True
WARMSTART_REQUIRED  = True


# ─────────────────────────────────────────────────────────────────────────────
# Below this line: experiment logic  — edit only if you know what you're doing
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
# PPO training + evaluation  (mirrors the v4 fine-tune loop)
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


def _train_and_eval(genotype: dict, cfg: dict) -> tuple[float, Any, str | None, int]:
    """Decode genome → multi-task fine-tune PPO from v4 weights → evaluate.

    Mirrors the inner loop of 27_train_rl_hex_mtrl_v4.main:
        * MultiTaskHexVecEnv with this candidate's propellers
        * VecNormalize obs-norm loaded from the v4 run
        * PPO weights loaded from the v4 policy.zip
        * EntCoefAnneal callback during ``model.learn``
        * _eval_per_task across all four tasks for fitness

    Fitness is the sum of gate-passes per second across the four tasks.
    """
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, None, 0

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), None, None, 0

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, None, 0

    n_motors = len(propellers)
    if n_motors != 6:
        # v4 actor-critic is locked to 6 motors; reject mismatched morphologies.
        return float("-inf"), None, None, n_motors

    raw_env = MultiTaskHexVecEnv(
        propellers=propellers,
        num_envs=cfg["num_envs"],
        device=cfg["device"],
        dt=0.01,
        seed=args.seed,
        gate_density=GATE_DENSITY,
    )

    # Obs-norm only; the per-task reward normalizer lives inside MultiTaskHexVecEnv.
    env = VecNormalize.load(str(_warm_vecnorm), raw_env)
    env.training = bool(RESET_VECNORM_TRAIN)
    env.norm_reward = False

    rollout_size = PPO_N_STEPS * cfg["num_envs"]
    batch_size   = rollout_size // 8

    # Load v4 PPO weights against this candidate's env. custom_objects
    # overrides per-run hyperparameters so the fine-tune uses CONFIG-3 values
    # regardless of what the base checkpoint shipped with.
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
    # Reset ent_coef in case the base checkpoint annealed it to near-zero.
    model.ent_coef = PPO_ENT_START

    # Each candidate has its own dynamics, so PPO.load's restored Adam state
    # (mean/variance from the v4 base training) is stale and biased toward
    # the original morphology. Reset the optimiser to a fresh Adam keyed to
    # the loaded weights + this run's learning rate so the fine-tune behaves
    # like an independent RL session warm-started from the v4 actor-critic.
    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(), lr=PPO_LR,
    )
    # PPO uses model.lr_schedule to drive learning_rate during learn(); pin
    # it to a constant so the reset optimiser doesn't get re-overwritten.
    model.lr_schedule = lambda _progress_remaining: PPO_LR

    model.learn(
        total_timesteps=cfg["ppo_steps"],
        callback=[EntCoefAnneal(PPO_ENT_START, PPO_ENT_END, cfg["ppo_steps"])],
        progress_bar=False,
    )

    # Multi-task evaluation rollout (1500 steps so every per-task env can
    # time out at least once; matches v4's main-script eval window).
    _ep_r, _ep_g, total_g, live_steps = _eval_per_task(
        env, model, n_steps=cfg["eval_steps"],
    )

    # Fitness: total gate-passes per second across all four tasks. Robust to
    # n=0 completed episodes (since total_g counts every increment seen
    # during the rollout regardless of whether the episode finished).
    elapsed_s = float(live_steps.sum()) * 0.01
    fitness   = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0

    # Capture the per-candidate VecNormalize obs-norm stats so the replay
    # script can reconstruct the exact observation pipeline this individual
    # was evaluated against. (env.save serialises the whole VecNormalize.)
    vecnorm_b64 = _vecnorm_to_b64(env)

    # The caller still needs the model to extract weights, so cleanup of the
    # env + cached tensors happens here; the model is released after the
    # b64 dump in the evaluator loop.
    try:
        env.close()
    except Exception:
        pass
    del env, raw_env
    if cfg["device"].startswith("cuda"):
        torch.cuda.empty_cache()

    return fitness, model, vecnorm_b64, n_motors


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_ppo(population: Population) -> Population:
    """Fine-tune the v4 policy on every unevaluated body; assign fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="MTRL fine-tune…", console=console)):
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        # Per-candidate header — makes each rerun visually distinct in the log,
        # since each blueprint is a separate RL session warm-started from v4.
        console.rule(f"[cyan]candidate {display_id} — blueprint #{_i + 1}/{len(to_eval)}")

        t0  = time.time()
        fit, model, vecnorm_b64, n_motors = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        if model is not None:
            ind.tags["policy_b64"] = _model_to_b64(model)
        if vecnorm_b64 is not None:
            ind.tags["vecnorm_b64"] = vecnorm_b64

        # Release the per-candidate model + GPU tensors before the next rerun.
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
# Build generation pipeline  (CUSTOM_GENERATION_OPS or default)
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

console.rule("[bold blue]Drone Body Evolution + Multi-Task PPO (v4 warm-start)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"ppo_steps={PPO_STEPS:,}  num_envs={PPO_NUM_ENVS}  "
    f"tasks={list(TASK_NAMES)}  gate_density={GATE_DENSITY}  "
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

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best fitness (gates/s across 4 tasks): {best.fitness_:.4f}  DB → {DB_PATH}"
)
console.log(
    f"\nTo visualise the best individual (or any other), use the replay script:\n"
    f"  uv run examples/spear/30_replay_v4_evo.py --db {DB_PATH}\n"
    f"  uv run examples/spear/30_replay_v4_evo.py --db {DB_PATH} --rank best\n"
    f"  uv run examples/spear/30_replay_v4_evo.py --db {DB_PATH} --id <individual_id>"
)
