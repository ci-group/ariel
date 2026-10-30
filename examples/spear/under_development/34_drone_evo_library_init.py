"""Library-based initialization for EA-wrapped hex controller training,
following Rehberg et al. (RA-L 2026) — "Efficient Knowledge Transfer
for Jump-Starting Control Policy Learning of Multirotors".

Paper-faithful pieces implemented here:

    * GROWING LIBRARY. We maintain a fixed-capacity library of trained
      (policy, vecnorm, blueprint) tuples. Bootstrapped from the v4 hex
      generalist, then every accepted EA individual is appended (FIFO
      eviction once at capacity).

    * REWARD-BASED SIMILARITY m_r. For each new candidate morphology we
      load every library policy against the candidate's env, do a short
      probe rollout, and pick the entry with the highest accumulated
      reward as the init. Paper §III-C-3 + Tab. I: this is the *only*
      similarity measure that correlated significantly with sample-
      efficiency improvement; m_c / m_wd had no statistical significance.

    * ACTOR + CRITIC + ADAM TRANSFER. Paper §III-C explicitly notes that
      transferring optimizer state (Adam first/second moments) on top of
      actor+critic weights gave "significantly better performance".
      ``PPO.load`` restores the optimizer state from the saved policy
      zip; we deliberately do NOT reset it (unlike 29_drone_evo_v4_
      warmstart.py, which resets Adam — that script predates this finding).

What we deliberately DON'T implement (and why):

    * Per-candidate allocation network. The paper splits the controller
      into a wrench-output RL policy + a supervised allocation network.
      v4's actor-critic is end-to-end (state → motor commands), so the
      allocation network has no counterpart here. Cross-morphology
      transfer in this script therefore relies entirely on the RL
      policy's robustness, which is weaker than the paper's split but
      keeps this script drop-in compatible with the v4 stack.

Weekend run guidance:
    uv run examples/spear/34_drone_evo_library_init.py --device cuda:0

NOTE: do NOT add ``from __future__ import annotations`` — ariel's
@EAOperation decorator needs real (not stringified) type hints at
import time.
"""
# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import base64
import gc
import importlib.util
import json
import sys
import tempfile
import time
from collections import deque
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
HOVER_ID              = _v4.HOVER_ID


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Library-based EA init for hex controller (Rehberg et al. 2026)"
)
parser.add_argument("--device",  default="cpu")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_library/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA
# ═════════════════════════════════════════════════════════════════════════════

EA_POP_SIZE = 8
EA_GENS     = 12

# v4 actor-critic is locked to 6 motors. Library-based transfer in this script
# is only validated within the hexarotor family; keep min == max == 6.
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

APPEND_ARM_CHANCE     = 0.0
BILATERAL_SYMMETRY    = None
REPAIR                = False
MUTATION_SCALES       = None
CUSTOM_GENERATION_OPS = None


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — MULTI-TASK ENV
# ═════════════════════════════════════════════════════════════════════════════

GATE_DENSITY = 3
PROP_SIZE    = 2


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — PPO fine-tune
# ═════════════════════════════════════════════════════════════════════════════

PPO_STEPS         = 4_000_000
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


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 4 — LIBRARY + SIMILARITY (paper §III-C)
# ═════════════════════════════════════════════════════════════════════════════

# Bootstrap policy: the v4 hex generalist. Library[0] is always this entry.
WARMSTART_DIR      = "__data__/spear_rl_hex_mtrl_v4/20260616_163031"
WARMSTART_REQUIRED = True

# Library capacity (excl. the locked bootstrap entry). FIFO eviction when full.
LIBRARY_CAPACITY = 16

# m_r probe (paper §III-C-3). Cheap by design:
#   * PROBE_NUM_ENVS = NUM_TASKS  (one env per task slot — the legal minimum
#     for MultiTaskHexVecEnv; 32× cheaper than the 128-env training vec).
#   * Hover-task envs only contribute to the score, giving a position-only
#     similarity signal (paper §IV-A uses position-only reward in the lower-
#     fidelity library sim). Hover-task envs in v4 use _hover_reward whose
#     dominant term is exactly the position offset.
PROBE_STEPS    = 400
PROBE_NUM_ENVS = NUM_TASKS

# Only admit a freshly-trained policy into the library if its fitness exceeds
# this fraction of the best fitness seen so far. Prevents library poisoning
# by collapsed / pathological morphologies.
LIBRARY_ADMIT_FRAC = 0.6

# Reset VecNormalize.training when loading. Same as 29.
RESET_VECNORM_TRAIN = True


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 5 — EARLY-STOP (paper measures "interactions to reach goal reward")
# ═════════════════════════════════════════════════════════════════════════════

# Evaluate every this many env-steps during training. Triggers a short
# in-place eval rollout against the same env (no model copy).
EARLYSTOP_EVAL_EVERY  = 200_000
EARLYSTOP_EVAL_STEPS  = 300
# Stop once fitness clears this target (gates/s, summed across tasks).
EARLYSTOP_TARGET      = 6.0
# Or stop after this many consecutive evals without > MIN_DELTA improvement.
EARLYSTOP_PATIENCE    = 4
EARLYSTOP_MIN_DELTA   = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Below: experiment logic
# ─────────────────────────────────────────────────────────────────────────────

DATA   = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH      = DATA / f"database_{RUN_ID}.db"
LIBRARY_LOG  = DATA / f"library_{RUN_ID}.jsonl"
METRICS_LOG  = DATA / f"metrics_{RUN_ID}.jsonl"

_warm_dir     = Path(WARMSTART_DIR)
_warm_policy  = _warm_dir / "policy.zip"
_warm_vecnorm = _warm_dir / "vecnormalize.pkl"
if WARMSTART_REQUIRED and (not _warm_policy.exists() or not _warm_vecnorm.exists()):
    console.log(
        f"[red]Warm-start dir missing required files: {_warm_dir}\n"
        f"  policy.zip ({'OK' if _warm_policy.exists() else 'MISSING'}), "
        f"vecnormalize.pkl ({'OK' if _warm_vecnorm.exists() else 'MISSING'})[/red]"
    )
    raise SystemExit(1)

if PPO_NUM_ENVS % NUM_TASKS != 0:
    raise SystemExit(
        f"PPO_NUM_ENVS ({PPO_NUM_ENVS}) must be divisible by NUM_TASKS ({NUM_TASKS})"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Genome handler
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
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _b64_read(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


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


def _b64_to_tmpfile(b64: str, suffix: str) -> Path:
    tmp = Path(tempfile.mktemp(suffix=suffix))
    tmp.write_bytes(base64.b64decode(b64))
    return tmp


def _build_raw_env(propellers, seed: int, num_envs: int = PPO_NUM_ENVS) -> MultiTaskHexVecEnv:
    return MultiTaskHexVecEnv(
        propellers=propellers,
        num_envs=num_envs,
        device=args.device,
        dt=0.01,
        seed=seed,
        gate_density=GATE_DENSITY,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Library (paper §III-C, Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each entry: {"tag": str, "policy_b64": str, "vecnorm_b64": str}
# library[0] is the v4 hex bootstrap and is never evicted.
# library[1:] is a FIFO deque capped at LIBRARY_CAPACITY.

_bootstrap_entry = {
    "tag":         "v4_hex_bootstrap",
    "policy_b64":  _b64_read(_warm_policy),
    "vecnorm_b64": _b64_read(_warm_vecnorm),
}
library: list[dict[str, str]] = [_bootstrap_entry]
library_tail: deque[dict[str, str]] = deque(maxlen=LIBRARY_CAPACITY)


def _library_view() -> list[dict[str, str]]:
    return [library[0], *library_tail]


def _library_admit(tag: str, policy_b64: str, vecnorm_b64: str) -> None:
    entry = {"tag": tag, "policy_b64": policy_b64, "vecnorm_b64": vecnorm_b64}
    library_tail.append(entry)
    with LIBRARY_LOG.open("a") as f:
        f.write(json.dumps({"tag": tag, "t": time.time()}) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# m_r: reward-based similarity (paper §III-C, eq. 17)
# ─────────────────────────────────────────────────────────────────────────────

def _probe_score(entry: dict[str, str], raw_env: MultiTaskHexVecEnv) -> float:
    """Load a library policy + its vecnorm onto raw_env and return the mean
    per-step reward accumulated *only over hover-task envs* (position-only
    similarity signal — see CONFIG 4 comment).
    """
    pol_path = _b64_to_tmpfile(entry["policy_b64"],  ".zip")
    vn_path  = _b64_to_tmpfile(entry["vecnorm_b64"], ".pkl")
    try:
        env = VecNormalize.load(str(vn_path), raw_env)
        env.training    = False
        env.norm_reward = False

        per = raw_env.per_task
        hover_mask = (raw_env.task_ids == HOVER_ID)
        n_hover = int(hover_mask.sum())
        if n_hover == 0:
            return 0.0

        model = PPO.load(str(pol_path), env=env, device=args.device,
                         custom_objects={"policy_class": MTRLActorCriticPolicy})
        obs = env.reset()
        total_r = 0.0
        for _ in range(PROBE_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, rews, _dones, _infos = env.step(action)
            total_r += float(np.asarray(rews)[hover_mask].sum())
        del model
        # Normalize so scores are comparable across different PROBE_NUM_ENVS.
        return total_r / (PROBE_STEPS * n_hover)
    finally:
        pol_path.unlink(missing_ok=True)
        vn_path.unlink(missing_ok=True)


def _pick_init(propellers) -> tuple[dict[str, str], list[tuple[str, float]]]:
    """Build a cheap probe env, score every library entry on it, return
    (best_entry, scoreboard). The probe env is destroyed afterwards so the
    fine-tune step builds a fresh full-size env.
    """
    scores: list[tuple[str, float]] = []
    best_score = -np.inf
    best_entry = library[0]
    for entry in _library_view():
        probe_raw = _build_raw_env(propellers, seed=args.seed,
                                   num_envs=PROBE_NUM_ENVS)
        try:
            s = _probe_score(entry, probe_raw)
        finally:
            try:
                probe_raw.close()
            except Exception:
                pass
            del probe_raw
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        scores.append((entry["tag"], s))
        if s > best_score:
            best_score = s
            best_entry = entry
    return best_entry, scores


# ─────────────────────────────────────────────────────────────────────────────
# Early-stop callback (paper measures interactions-to-target, not fixed budget)
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopOnFitness(BaseCallback):
    """Periodically eval `_eval_per_task` on the live env, stop when fitness
    clears EARLYSTOP_TARGET or plateaus for EARLYSTOP_PATIENCE consecutive
    evals (improvement < EARLYSTOP_MIN_DELTA).

    Records (env_steps, fitness) pairs in self.history so the caller can log
    interactions-to-target for the m_r ↔ savings correlation analysis.
    """
    def __init__(self, eval_env, target: float, patience: int,
                 min_delta: float, eval_every: int, eval_steps: int):
        super().__init__()
        self.eval_env   = eval_env
        self.target     = target
        self.patience   = patience
        self.min_delta  = min_delta
        self.eval_every = eval_every
        self.eval_steps = eval_steps
        self._next_eval = eval_every
        self._stale     = 0
        self._best      = -np.inf
        self.history: list[tuple[int, float]] = []
        self.steps_to_target: int | None = None

    def _on_step(self) -> bool:
        if self.num_timesteps < self._next_eval:
            return True
        self._next_eval = self.num_timesteps + self.eval_every
        _epr, _epg, total_g, live_steps = _eval_per_task(
            self.eval_env, self.model, n_steps=self.eval_steps,
        )
        elapsed_s = float(live_steps.sum()) * 0.01
        fit = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0
        self.history.append((int(self.num_timesteps), fit))
        if fit >= self.target and self.steps_to_target is None:
            self.steps_to_target = int(self.num_timesteps)
            return False
        if fit > self._best + self.min_delta:
            self._best = fit
            self._stale = 0
        else:
            self._stale += 1
            if self._stale >= self.patience:
                return False
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Per-candidate fine-tune (paper Algorithm 1, lines after init pick)
# ─────────────────────────────────────────────────────────────────────────────

# Tracks the best fitness seen so far so library admission stays meaningful.
_global_best_fitness: float = 0.0


def _train_and_eval(genotype: dict) -> tuple[
    float, PPO | None, str | None, int, str | None, dict[str, Any]
]:
    metrics: dict[str, Any] = {}
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, None, 0, None, metrics

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), None, None, 0, None, metrics

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=PROP_SIZE)
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, None, 0, None, metrics

    n_motors = len(propellers)
    if n_motors != 6:
        return float("-inf"), None, None, n_motors, None, metrics

    # --- Step 1: m_r — pick init from library (uses its own cheap probe envs)
    init_entry, scoreboard = _pick_init(propellers)
    console.log(
        "    m_r scores:  " +
        ", ".join(f"{tag}={s:.3f}" for tag, s in scoreboard) +
        f"  → picked [bold]{init_entry['tag']}[/bold]"
    )
    metrics["mr_scores"]   = {tag: s for tag, s in scoreboard}
    metrics["mr_winner"]   = init_entry["tag"]
    metrics["mr_winner_score"] = next(s for tag, s in scoreboard
                                      if tag == init_entry["tag"])

    # --- Step 2: build full-size training env
    raw_env = _build_raw_env(propellers, seed=args.seed)

    vn_path = _b64_to_tmpfile(init_entry["vecnorm_b64"], ".pkl")
    pol_path = _b64_to_tmpfile(init_entry["policy_b64"],  ".zip")
    try:
        env = VecNormalize.load(str(vn_path), raw_env)
        env.training    = bool(RESET_VECNORM_TRAIN)
        env.norm_reward = False

        rollout_size = PPO_N_STEPS * PPO_NUM_ENVS
        batch_size   = rollout_size // 8

        # PPO.load restores actor, critic AND Adam state from the policy zip.
        # Paper §III-C: keeping the optimizer state (first/second moments)
        # gives "significantly better performance". We deliberately do NOT
        # rebuild model.policy.optimizer here.
        model = PPO.load(
            str(pol_path), env=env, device=args.device,
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
        model.ent_coef    = PPO_ENT_START
        model.lr_schedule = lambda _p: PPO_LR
    finally:
        pol_path.unlink(missing_ok=True)
        vn_path.unlink(missing_ok=True)

    early = EarlyStopOnFitness(
        eval_env=env,
        target=EARLYSTOP_TARGET,
        patience=EARLYSTOP_PATIENCE,
        min_delta=EARLYSTOP_MIN_DELTA,
        eval_every=EARLYSTOP_EVAL_EVERY,
        eval_steps=EARLYSTOP_EVAL_STEPS,
    )
    model.learn(
        total_timesteps=PPO_STEPS,
        callback=[EntCoefAnneal(PPO_ENT_START, PPO_ENT_END, PPO_STEPS), early],
        progress_bar=False,
    )
    metrics["steps_used"]      = int(model.num_timesteps)
    metrics["steps_to_target"] = early.steps_to_target  # None if never hit
    metrics["fitness_curve"]   = early.history

    _ep_r, _ep_g, total_g, live_steps = _eval_per_task(env, model, n_steps=EVAL_STEPS)
    elapsed_s = float(live_steps.sum()) * 0.01
    fitness   = float(total_g.sum()) / elapsed_s if elapsed_s > 0 else 0.0

    vecnorm_b64 = _vecnorm_to_b64(env)

    try:
        env.close()
    except Exception:
        pass
    del env, raw_env
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return fitness, model, vecnorm_b64, n_motors, init_entry["tag"], metrics


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation op
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_library(population: Population) -> Population:
    global _global_best_fitness

    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="library fine-tune…", console=console)):
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.rule(
            f"[cyan]candidate {display_id} — {_i + 1}/{len(to_eval)}  "
            f"library_size={len(_library_view())}"
        )

        t0 = time.time()
        fit, model, vecnorm_b64, n_motors, init_tag, metrics = _train_and_eval(ind.genotype)
        ind.fitness = fit
        ind.tags["init_tag"] = init_tag if init_tag is not None else ""

        metrics_row = {
            "t": time.time(),
            "id": str(display_id),
            "fitness": fit,
            "n_motors": n_motors,
            "init_tag": init_tag,
            "wall_s": time.time() - t0,
            **metrics,
        }
        with METRICS_LOG.open("a") as f:
            f.write(json.dumps(metrics_row, default=float) + "\n")

        if model is not None:
            policy_b64 = _model_to_b64(model)
            ind.tags["policy_b64"]  = policy_b64
            ind.tags["vecnorm_b64"] = vecnorm_b64

            # Library admission policy: must clear LIBRARY_ADMIT_FRAC * best.
            # The very first accepted individual seeds the threshold.
            admit_floor = LIBRARY_ADMIT_FRAC * max(_global_best_fitness, 1e-6)
            if fit >= admit_floor and fit > 0:
                tag = f"ind_{display_id}_f{fit:.2f}"
                _library_admit(tag, policy_b64, vecnorm_b64)
                console.log(f"    [green]→ admitted to library as {tag}[/green]")
            else:
                console.log(
                    f"    [yellow]skip library admit  "
                    f"(fit {fit:.3f} < floor {admit_floor:.3f})[/yellow]"
                )
            _global_best_fitness = max(_global_best_fitness, fit)

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        console.log(
            f"  id={display_id}  motors={n_motors}  init={init_tag}  "
            f"gates/s={fit:.3f}  ({time.time()-t0:.1f}s)"
        )

    finite = [ind.fitness_ for ind in population.alive.evaluated
              if np.isfinite(ind.fitness_)]  # type: ignore[arg-type]
    if finite:
        console.log(f"    batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}")
    return population


eval_op = EAOperation(evaluate_with_library)


# ─────────────────────────────────────────────────────────────────────────────
# Generation pipeline
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

console.rule("[bold blue]Library-Based EA Init (Rehberg et al. 2026)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}-{N_ARMS_MAX}  "
    f"ppo_steps={PPO_STEPS:,}  num_envs={PPO_NUM_ENVS}  "
    f"library_cap={LIBRARY_CAPACITY}  probe_steps={PROBE_STEPS}  "
    f"warm_dir={WARMSTART_DIR}  device={args.device}"
)
console.log(f"DB → {DB_PATH}")
console.log(f"Library log → {LIBRARY_LOG}")

initial_pop = Population([Individual() for _ in range(EA_POP_SIZE)])
init_op     = initialize_drones(template_handler=genome_handler)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop
           if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
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
# Save best
# ─────────────────────────────────────────────────────────────────────────────

ea.fetch_population(only_alive=False, requires_eval=False)
finite = [ind for ind in ea.population
          if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
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
    (DATA / f"best_policy_{RUN_ID}.zip").write_bytes(
        base64.b64decode(best.tags["policy_b64"])
    )
if "vecnorm_b64" in best.tags:
    (DATA / f"best_vecnormalize_{RUN_ID}.pkl").write_bytes(
        base64.b64decode(best.tags["vecnorm_b64"])
    )

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best fitness (gates/s): {best.fitness_:.4f}  "
    f"final library size: {len(_library_view())}  DB → {DB_PATH}"
)
console.log(
    f"\nReplay:\n"
    f"  uv run examples/spear/30_replay_v4_evo.py --db {DB_PATH} --rank best"
)
