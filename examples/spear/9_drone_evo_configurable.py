"""Configurable drone morphology evolution + PPO gate-racing.

This script is a researcher-friendly wrapper around the full experiment
pipeline from 7_drone_evo_rl_quintic.py.  All tuneable knobs are grouped
into clearly labelled CONFIG sections near the top of the file — scroll
to the one you want to change and edit the Python values directly.

Quick start
-----------
    uv run examples/spear/9_drone_evo_configurable.py

Pass a few CLI flags to override output location, device, and seed
without touching the config sections:

    uv run examples/spear/9_drone_evo_configurable.py \\
        --device cuda:0 --seed 7 --out-dir __data__/my_run

See README_drone_evo.md in this directory for a full explanation of every
setting and links to relevant source files.

NOTE: do NOT add ``from __future__ import annotations`` to this file —
ariel's @EAOperation decorator needs real (not stringified) type hints
at import time.
"""
# ─────────────────────────────────────────────────────────────────────────────
# Standard-library / third-party imports  (do not edit)
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import base64
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator"))
from planner_generator import generate_paths_from_coefficients  # noqa: E402

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
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv
from ariel.simulation.tasks.torch_drone_gate_env import TorchDroneGateEnv

# ─────────────────────────────────────────────────────────────────────────────
# Minimal CLI  (run-level overrides — not the main config surface)
# ─────────────────────────────────────────────────────────────────────────────
curr_time = time.strftime("%Y%m%d_%H%M%S")
parser = argparse.ArgumentParser(
    description="Configurable drone morphology evolution + PPO"
)
parser.add_argument("--device",  default="cpu",
                    help="Torch device: 'cpu' or 'cuda:0'  (default: cpu)")
parser.add_argument("--seed",    type=int, default=42)
parser.add_argument("--out-dir", default=f"__data__/drone_evo_configurable/{curr_time}")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

console = Console()


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 1 — EA  (evolutionary algorithm)
#
#  What controls how bodies are generated, varied, and selected.
#
#  Example scripts to read first:
#    examples/spear/7_drone_evo_rl_quintic.py  ← full pipeline this is based on
#    examples/z_ec_course/                            ← step-by-step EA tutorials
#
#  Key source files:
#    src/ariel/ec/ea.py                              ← EA loop
#    src/ariel/body_phenotypes/drone/operations.py   ← all EAOperation factories
#    src/ariel/ec/drone/genome_handlers/
#      spherical_angular_genome_handler.py           ← mutation / crossover internals
# ═════════════════════════════════════════════════════════════════════════════

# ── Population & generations ──────────────────────────────────────────────────
EA_POP_SIZE  = 10    # individuals alive at any time
EA_GENS      = 50    # number of EA generations to run

# ── Drone body: number of arms ────────────────────────────────────────────────
# All arms share the same propeller size (set in CONFIG 3).
# Keeping this fixed (min == max) focuses evolution on arm geometry only.
# Set min < max to also evolve the number of arms.
N_ARMS_MIN = 6
N_ARMS_MAX = 6       # set > N_ARMS_MIN to allow variable arm count

# ── Genome parameter ranges ───────────────────────────────────────────────────
# Each row is [min, max] for one arm parameter.  Order must match the columns
# produced by SphericalAngularDroneGenomeHandler.
#
#   col 0 — arm length (m)           physical constraint: 0.03–0.25 m
#   col 1 — arm azimuth (rad)        full circle: -π … +π
#   col 2 — arm elevation (rad)      tilt up/down: -π/2 … +π/2
#   col 3 — motor disc azimuth (rad) spin around arm axis: -π … +π
#   col 4 — motor disc pitch (rad)   tilt motor: -π … +π
#   col 5 — propeller spin direction 0 = CCW, 1 = CW  (kept binary; do not widen)
PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm length
    [-np.pi, np.pi],          # arm azimuth
    [-np.pi / 2, np.pi / 2],  # arm elevation
    [-np.pi, np.pi],          # motor disc azimuth
    [-np.pi, np.pi],          # motor disc pitch
    [0, 1],                   # spin direction (binary)
])

# ── Genome handler options ─────────────────────────────────────────────────────
# append_arm_chance: probability of adding or removing one arm during mutation.
#   0.0 = fixed arm count (faster, more stable)
#   0.1 = 10 % chance of ±1 arm per mutation
APPEND_ARM_CHANCE = 0.0

# bilateral_plane_for_symmetry: force mirror symmetry across a body plane.
#   None = no symmetry (fully asymmetric designs allowed)
#   "xz"  = left-right symmetric (arms appear in mirrored pairs)
#   "xy"  = front-back symmetric
#   "yz"  = another axis — rarely used for drones
BILATERAL_SYMMETRY = None

# repair: snap out-of-bounds parameter values back into PARAMETER_LIMITS
#   after mutation.  Useful when APPEND_ARM_CHANCE > 0.
REPAIR = False

# mutation_scales_percentage: per-parameter mutation step size as a fraction
#   of each parameter's range.  None = library default (~5 %).
#   Shape must be (6,) matching the columns of PARAMETER_LIMITS.
#   Example — tighter arm-length mutations, larger angle mutations:
#     MUTATION_SCALES = np.array([0.03, 0.10, 0.10, 0.10, 0.10, 0.50])
MUTATION_SCALES = None   # None → use library defaults

# ── Generation pipeline ────────────────────────────────────────────────────────
# The pipeline runs once per generation in the listed order.
# Each entry is an EAOperation factory call — remove or reorder to experiment.
#
# Available factories (from ariel.body_phenotypes.drone):
#   parent_tag(n)              — mark the top-n alive individuals as parents
#   crossover_drones(...)      — produce one child per parent pair
#   mutate_drones(...)         — perturb all unevaluated individuals
#   truncation_select(n)       — kill everyone outside the top-n by fitness
#
# The evaluate step (eval_op, defined later) MUST remain in the pipeline.
# Its position determines whether selection sees the new or old fitness.
#
# Example — (μ+λ) style: keep parents + evaluate offspring before selecting:
#   generation_ops = [
#       parent_tag(n=EA_POP_SIZE // 2),
#       crossover_drones(template_handler=genome_handler),
#       mutate_drones(template_handler=genome_handler),
#       eval_op,                           # ← inserted automatically below
#       truncation_select(n=EA_POP_SIZE),
#   ]
#
# Leave as None to use the default (μ,λ) pipeline built from the values above.
CUSTOM_GENERATION_OPS = None   # None = use default pipeline


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 2 — GATE TRACK  (the task / environment)
#
#  Controls the shape and difficulty of the racing circuit the drone must fly.
#
#  Key source files:
#    src/ariel/simulation/tasks/torch_drone_gate_env.py  ← GPU-accelerated env
#    src/ariel/simulation/tasks/drone_gate_env.py        ← CPU reference env
#    goal_generator_ltu/polynomial_goal_generator/       ← quintic path planner
#
#  Visualise the track (no drone) with:
#    uv run examples/spear/8_visualize_gate_track.py
# ═════════════════════════════════════════════════════════════════════════════

# Number of gates sampled from the quintic path.
# More gates = longer, harder circuit; fewer = shorter laps, faster evaluation.
GATE_PATH_STEPS = 15    # default: 15

# Spatial scale applied to the unit-square quintic path.
# path_scale=5 → gates spread over ≈ ±5 m in x/y.
# Larger = faster flight required; smaller = tighter turns.
GATE_PATH_SCALE = 5.0   # metres

# Gate altitude in NED z-convention (negative = above ground).
# -1.5 means gates are 1.5 m above the ground plane.
GATE_Z_HEIGHT = -1.5    # NED z (metres)

# Gate evaluation mode:
#   "naive"  — one fixed quintic circuit per run (all individuals fly the same track)
#   "online" — gates are generated on-the-fly as the drone passes each one
GATE_MODE = "naive"


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG 3 — PPO / RL  (reinforcement learning per body)
#
#  Controls how the PPO controller is trained for each candidate body.
#  Longer training = better fitness estimates but slower EA wall-clock time.
#
#  Key source files:
#    stable_baselines3/ppo/  (installed library)
#    src/ariel/simulation/tasks/torch_drone_gate_env.py  ← reward function
#
#  Timing reference (RTX 5070 Ti Laptop, 2000 envs):
#    ppo_steps=1_000_000 → ~9 s per individual
#    ppo_steps=2_000_000 → ~16 s per individual
# ═════════════════════════════════════════════════════════════════════════════

# Total PPO environment steps used to train each candidate body.
# Rule of thumb: ≥ 500 000 for meaningful learning; 2 M+ for fine policies.
PPO_STEPS = 1_000_000

# Parallel environments inside TorchDroneGateEnv.
# More envs = faster wall-clock training (up to GPU memory limit).
# RTX 5070 Ti: up to ~4 000.  A100: up to ~16 000.
PPO_NUM_ENVS = 2000

# Steps used for the final deterministic evaluation rollout (not training).
# Should be large enough for the drone to complete several laps.
PPO_EVAL_STEPS = 40_000

# Propeller size in inches.  Larger props → more thrust per RPM.
# Must be 2 (default), 3, or 4 — values that have a matching rotor model.
PROP_SIZE = 2

# ── PPO network architecture ──────────────────────────────────────────────────
# pi  = actor (policy) network hidden layers
# vf  = critic (value) network hidden layers
# Deeper / wider = more expressive but slower to train.
PPO_NET_ARCH = dict(pi=[256, 256], vf=[256, 256])

# Activation function applied between hidden layers.
# torch.nn.Tanh is the SB3 default; SiLU often trains faster for flight tasks.
PPO_ACTIVATION = torch.nn.SiLU

# ── PPO hyperparameters ───────────────────────────────────────────────────────
# These are standard PPO knobs — see the SB3 docs for explanations.
# The defaults below work well for this drone gate task.
PPO_N_EPOCHS    = 20     # gradient update passes per rollout collection
PPO_GAMMA       = 0.999  # discount factor (close to 1 = long-horizon planning)
PPO_LR          = 1e-3   # learning rate
PPO_CLIP_RANGE  = 0.2    # PPO clipping parameter
PPO_ENT_COEF    = 0.01   # entropy bonus (higher = more exploration)


# ─────────────────────────────────────────────────────────────────────────────
# Below this line: experiment logic  — edit only if you know what you're doing
# ─────────────────────────────────────────────────────────────────────────────

DATA   = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

_COEFFS_PATH = (
    _REPO_ROOT / "goal_generator_ltu" / "polynomial_goal_generator" / "quintic_coeffs.npy"
)
COEFFS = np.load(_COEFFS_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Gate track helpers  (unchanged from script 7)
# ─────────────────────────────────────────────────────────────────────────────

def _quintic_to_gates(
    coeffs: np.ndarray,
    n_gates: int,
    scale: float,
    z: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample one quintic path → gate_pos (N×3) and gate_yaw (N,).

    Uses the planner's smooth tangent yaw (not gate-to-gate differences).
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

    xy       = xy_dense[kept]
    gate_pos = np.column_stack([xy, np.full(n_gates, z)]).astype(np.float32)
    gate_yaw = yaw_dense[kept].astype(np.float32)
    return gate_pos, gate_yaw


# ─────────────────────────────────────────────────────────────────────────────
# Genome handler  (built from CONFIG 1 values)
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
# Environment factory  (built from CONFIG 2 values)
# ─────────────────────────────────────────────────────────────────────────────

if GATE_MODE == "naive":
    _gate_pos, _gate_yaw = _quintic_to_gates(
        COEFFS, GATE_PATH_STEPS, GATE_PATH_SCALE, GATE_Z_HEIGHT, seed=args.seed,
    )
    _start_pos = (_gate_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)

    def _make_env(propellers: Any) -> TorchDroneGateEnv:
        return TorchDroneGateEnv(
            propellers=propellers,
            num_envs=PPO_NUM_ENVS,
            gates_pos=_gate_pos,
            gate_yaw=_gate_yaw,
            start_pos=_start_pos,
            device=args.device,
            dt=0.01,
            seed=args.seed,
        )
else:
    from ariel.simulation.tasks.drone_gate_env import DroneGateEnv as _DGE  # noqa: F811

    class _QuinticOnlineEnv(_DGE):
        """Thin subclass that generates gates on-the-fly from the quintic stream."""
        _BATCH = 64

        def __init__(self, propellers, **kw):
            self._coeffs     = COEFFS
            self._scale      = GATE_PATH_SCALE
            self._z          = GATE_Z_HEIGHT
            self._gate_rng   = np.random.default_rng(kw.pop("seed", None))
            self._gate_queue: list[tuple[np.ndarray, float]] = []
            init_pos, init_yaw = self._pull_n(GATE_PATH_STEPS)
            start = (init_pos[0] + np.array([0.0, -1.0, 0.0])).astype(np.float32)
            super().__init__(propellers=propellers, gates_pos=init_pos,
                             gate_yaw=init_yaw, start_pos=start, **kw)

        def _refill(self):
            paths, yaws_arr = generate_paths_from_coefficients(
                self._coeffs, num_generate=1, steps=self._BATCH,
                seed=int(self._gate_rng.integers(0, 2**31)), clip_range=(-1.0, 1.0),
            )
            xy = paths[0] * self._scale
            for i in range(len(xy)):
                pos = np.array([xy[i, 0], xy[i, 1], self._z], dtype=np.float32)
                self._gate_queue.append((pos, float(yaws_arr[0][i])))

        def _pull_n(self, n):
            while len(self._gate_queue) < n:
                self._refill()
            gates = [self._gate_queue.pop(0) for _ in range(n)]
            return (np.array([g[0] for g in gates], dtype=np.float32),
                    np.array([g[1] for g in gates], dtype=np.float32))

    def _make_env(propellers: Any) -> _QuinticOnlineEnv:
        return _QuinticOnlineEnv(
            propellers=propellers,
            num_envs=PPO_NUM_ENVS,
            device=args.device,
            dt=0.01,
            seed=args.seed,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PPO training + evaluation  (built from CONFIG 3 values)
# ─────────────────────────────────────────────────────────────────────────────

def _model_to_b64(model: PPO) -> str:
    import os
    tmp = Path(tempfile.mktemp(suffix=".zip"))
    try:
        model.save(str(tmp))
        return base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        tmp.unlink(missing_ok=True)


def _b64_to_policy(b64: str, path: Path) -> None:
    path.write_bytes(base64.b64decode(b64))


cfg: dict[str, Any] = {
    "ppo_steps":   PPO_STEPS,
    "num_envs":    PPO_NUM_ENVS,
    "eval_steps":  PPO_EVAL_STEPS,
    "prop_size":   PROP_SIZE,
    "device":      args.device,
    "make_env":    _make_env,
    "policy_bank": {},
    "policy_bank_fitness": {},
}


def _train_and_eval(genotype: dict, cfg: dict) -> tuple[float, Any, int]:
    """Decode genome → train PPO → evaluate deterministically."""
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None, 0

    if not np.isfinite(genome.arms[:, 0]).any():
        return float("-inf"), None, 0

    try:
        bp         = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None, 0

    n_motors     = len(propellers)
    env          = cfg["make_env"](propellers)
    n_steps      = max(64, cfg["ppo_steps"] // (cfg["num_envs"] * 10))
    rollout_size = n_steps * cfg["num_envs"]
    _target      = max(4096, rollout_size // 4)
    batch_size   = next(b for b in range(_target, 0, -1) if rollout_size % b == 0)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=PPO_ACTIVATION,
            net_arch=PPO_NET_ARCH,
            log_std_init=0.0,
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        learning_rate=PPO_LR,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        device=cfg["device"],
        verbose=0,
    )

    # Warm-start: copy weights from best known policy for this motor count.
    warm_b64: str | None = cfg["policy_bank"].get(n_motors)
    if warm_b64 is not None:
        _tmp = Path(tempfile.mktemp(suffix=".zip"))
        try:
            _tmp.write_bytes(base64.b64decode(warm_b64))
            _warm = PPO.load(str(_tmp), env=env, device=cfg["device"])
            model.policy.load_state_dict(_warm.policy.state_dict())
            del _warm
        except Exception:
            pass
        finally:
            _tmp.unlink(missing_ok=True)

    model.learn(total_timesteps=cfg["ppo_steps"])

    # Deterministic evaluation rollout.
    obs      = env.reset()
    ep_rews: list[float] = []
    cur_rews = np.zeros(cfg["num_envs"], dtype=np.float32)
    eval_iters = max(cfg["eval_steps"] // cfg["num_envs"], 500)
    for _ in range(eval_iters):
        action, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, _ = env.step(action)
        cur_rews += rews
        for i, done in enumerate(dones):
            if done:
                ep_rews.append(float(cur_rews[i]))
                cur_rews[i] = 0.0

    return float(np.mean(ep_rews)) if ep_rews else 0.0, model, n_motors


# ─────────────────────────────────────────────────────────────────────────────
# EA evaluation operation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_ppo(population: Population) -> Population:
    """Train a PPO policy for every unevaluated body; assign fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for _i, ind in enumerate(track(to_eval, description="PPO evaluation…", console=console)):
        t0  = time.time()
        fit, model, n_motors = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        warm_tag = ""
        if model is not None:
            ind.tags["policy_b64"] = _model_to_b64(model)
            if np.isfinite(fit):
                prev = cfg["policy_bank_fitness"].get(n_motors, float("-inf"))
                if fit > prev:
                    cfg["policy_bank"][n_motors]         = ind.tags["policy_b64"]
                    cfg["policy_bank_fitness"][n_motors] = fit
                    warm_tag = " [new bank best]"
        ws = "warm" if cfg["policy_bank"].get(n_motors) and not warm_tag else "cold"
        display_id = ind.id if ind.id is not None else f"#{_i + 1}"
        console.log(
            f"  id={display_id}  motors={n_motors}  "
            f"mean_reward={fit:.3f}  start={ws}  ({time.time()-t0:.1f}s){warm_tag}"
        )

    finite = [ind.fitness_ for ind in population.alive.evaluated if np.isfinite(ind.fitness_)]  # type: ignore[arg-type]
    if finite:
        console.log(f"    batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}")
    return population


eval_op = EAOperation(evaluate_with_ppo)

# ─────────────────────────────────────────────────────────────────────────────
# Build generation pipeline  (CONFIG 1 — CUSTOM_GENERATION_OPS or default)
# ─────────────────────────────────────────────────────────────────────────────

if CUSTOM_GENERATION_OPS is not None:
    # Splice eval_op into the user-supplied list if they left a None placeholder.
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

console.rule("[bold blue]Drone Body Evolution + PPO (configurable)")
console.log(
    f"pop={EA_POP_SIZE}  gens={EA_GENS}  arms={N_ARMS_MIN}–{N_ARMS_MAX}  "
    f"ppo_steps={PPO_STEPS:,}  num_envs={PPO_NUM_ENVS}  "
    f"gate_steps={GATE_PATH_STEPS}  scale={GATE_PATH_SCALE}  mode={GATE_MODE}  "
    f"device={args.device}"
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

best       = sorted(finite, key=lambda ind: ind.fitness_, reverse=True)[0]
best_genome = deserialize_genome(best.genotype)
best_bp    = spherical_angular_to_blueprint(best_genome.arms, propsize=PROP_SIZE)

bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"Best blueprint → {bp_path}")

_vis_gate_pos, _vis_gate_yaw = _quintic_to_gates(
    COEFFS, GATE_PATH_STEPS, GATE_PATH_SCALE, GATE_Z_HEIGHT, seed=args.seed,
)
np.save(DATA / f"gate_pos_{RUN_ID}.npy", _vis_gate_pos)
np.save(DATA / f"gate_yaw_{RUN_ID}.npy", _vis_gate_yaw)

if "policy_b64" in best.tags:
    policy_path = DATA / f"best_policy_{RUN_ID}.zip"
    _b64_to_policy(best.tags["policy_b64"], policy_path)
    console.log(f"Best policy    → {policy_path}")

console.log(
    f"[bold green]Done.[/bold green]  "
    f"Best fitness: {best.fitness_:.4f}  DB → {DB_PATH}"
)
console.log(
    f"\nTo visualise:\n"
    f"  uv run examples/e_drones_ec/6_visualize_evo_results.py --run-dir {DATA}\n"
    f"  uv run examples/spear/8_visualize_gate_track.py  --run-dir {DATA}"
)
