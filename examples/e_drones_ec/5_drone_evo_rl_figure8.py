"""Evolve drone morphology with ARIEL EA; evaluate each body by training a
short PPO policy on the figure-8 gate-racing task (DroneGateEnv).

Pipeline:

    SphericalAngularDroneGenomeHandler population
        -> ARIEL EA (parent_tag -> crossover -> mutate -> evaluate -> truncate)
        -> per-individual:
               genome -> Blueprint -> propellers
               -> DroneGateEnv (figure-8, vectorised, num_envs parallel copies)
               -> PPO train --ppo-steps timesteps
               -> deterministic eval rollout -> mean episode reward
        -> fitness = mean_reward  (higher is better; matches parent_tag / truncation_select)
    -> best blueprint JSON + best policy NPZ saved to --out-dir

The EA searches for morphologies that are *easy to control*: bodies whose
PPO-trained policy achieves high reward get high fitness and survive.
PPO provides the controller; the EA finds the body.

Smoke test (CPU, ~2 min):
    uv run examples/e_drones_ec/5_drone_evo_rl_figure8.py \\
        --pop 4 --gens 2 --ppo-steps 20000 --num-envs 20 --device cpu

Moderate run (GPU):
    uv run examples/e_drones_ec/5_drone_evo_rl_figure8.py \\
        --pop 8 --gens 5 --ppo-steps 200000 --num-envs 100 --device cuda:0
"""
# NOTE: no ``from __future__ import annotations`` — ariel's EAOperation
# inspects real annotation objects at decoration time; stringified annotations
# (PEP 563) would break the introspection.

import argparse
import base64
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import track
from stable_baselines3 import PPO

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ariel.body_phenotypes.drone import (
    crossover_drones,
    deserialize_genome,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.body_phenotypes.drone.backends import blueprint_to_propellers
from ariel.body_phenotypes.drone.decoders import spherical_angular_to_blueprint
from ariel.ec import EA, EAOperation, Individual, Population
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from ariel.simulation.tasks.drone_gate_env import DroneGateEnv


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

console = Console()
parser = argparse.ArgumentParser(
    description="Drone morphology evolution evaluated via PPO on the figure-8 task"
)
parser.add_argument("--pop", type=int, default=4,
                    help="Body population size (default 4)")
parser.add_argument("--gens", type=int, default=3,
                    help="EA generations (default 3)")
parser.add_argument("--min-arms", type=int, default=4)
parser.add_argument("--max-arms", type=int, default=6)
parser.add_argument("--ppo-steps", type=int, default=50_000,
                    help="PPO training timesteps per morphology (default 50 000)")
parser.add_argument("--num-envs", type=int, default=50,
                    help="Parallel environments inside DroneGateEnv (default 50)")
parser.add_argument("--eval-steps", type=int, default=2_000,
                    help="Deterministic rollout steps used to score each policy (default 2 000)")
parser.add_argument("--prop-size", type=int, default=2,
                    help="Propeller size in inches (default 2)")
parser.add_argument("--device", default="cpu",
                    help="Torch device: cpu or cuda:0 (default cpu)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--out-dir", default="__data__/drone_evo_rl_figure8")
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

DATA = Path(args.out_dir)
DATA.mkdir(parents=True, exist_ok=True)
RUN_ID = time.strftime("%Y%m%d_%H%M%S")
DB_PATH = DATA / f"database_{RUN_ID}.db"

def _model_to_b64(model: PPO) -> str:
    """Serialize a SB3 PPO model to a base64 string via a temp file.

    Stores the full SB3 ZIP (weights + hyperparams + spaces) so the model
    can be reloaded with PPO.load() without reconstructing the architecture.
    """
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
        tmp_path = f.name
    try:
        model.save(tmp_path)
        return base64.b64encode(Path(tmp_path).read_bytes()).decode("ascii")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _b64_to_policy_file(b64: str, out_path: Path) -> None:
    """Decode a base64 policy string back to a SB3 ZIP file."""
    out_path.write_bytes(base64.b64decode(b64))


# ---------------------------------------------------------------------------
# Per-individual PPO evaluation
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],           # arm length (m)
    [-np.pi, np.pi],         # arm azimuth
    [-np.pi / 2, np.pi / 2], # arm elevation
    [-np.pi, np.pi],         # motor disc azimuth
    [-np.pi, np.pi],         # motor disc pitch
    [0, 1],                  # propeller spin direction
])


def _train_and_eval(
    genotype: dict[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, Any]:
    """Decode genome, train PPO for cfg['ppo_steps'], evaluate deterministically.

    Returns (fitness, trained_model) where fitness = mean_episode_reward
    (higher = better; matches parent_tag/truncation_select which both use
    sort="max"). Returns (-inf, None) for invalid or undecodable genomes.
    """
    try:
        genome = deserialize_genome(genotype)
    except Exception:
        return float("-inf"), None

    valid = np.isfinite(genome.arms[:, 0])
    if not valid.any():
        return float("-inf"), None

    try:
        bp = spherical_angular_to_blueprint(genome.arms, propsize=cfg["prop_size"])
        propellers = blueprint_to_propellers(bp, convention="ned")
    except Exception:
        return float("-inf"), None

    # DroneGateEnv is already a VecEnv — PPO accepts it directly.
    env = DroneGateEnv(
        propellers=propellers,
        num_envs=cfg["num_envs"],
        device=cfg["device"],
        dt=0.01,
        seed=cfg["seed"],
    )

    # Scale n_steps so PPO does ~10 gradient updates within the budget.
    # Each update requires collecting n_steps * num_envs env-steps.
    n_steps = max(64, cfg["ppo_steps"] // (cfg["num_envs"] * 10))
    batch_size = min(5000, n_steps * cfg["num_envs"])

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),
            log_std_init=0.0,
        ),
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.999,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        device=cfg["device"],
        verbose=0,
    )
    model.learn(total_timesteps=cfg["ppo_steps"])

    # Deterministic evaluation: run cfg['eval_steps'] steps, collect episode
    # returns reported by the env in infos["episode"]["r"].
    obs = env.reset()
    ep_rewards: list[float] = []
    cur_rews = np.zeros(cfg["num_envs"], dtype=np.float32)
    for _ in range(cfg["eval_steps"]):
        action, _ = model.predict(obs, deterministic=True)
        obs, rews, dones, infos = env.step(action)
        cur_rews += rews
        for i, done in enumerate(dones):
            if done:
                ep_rewards.append(float(cur_rews[i]))
                cur_rews[i] = 0.0

    mean_reward = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    return mean_reward, model


# ---------------------------------------------------------------------------
# EA genome handler + evaluation operation
# ---------------------------------------------------------------------------

template_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(args.min_arms, args.max_arms),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=0.1,
    bilateral_plane_for_symmetry=None,
    repair=False,
    rnd=np.random.default_rng(args.seed),
)

cfg: dict[str, Any] = {
    "prop_size": args.prop_size,
    "num_envs": args.num_envs,
    "device": args.device,
    "ppo_steps": args.ppo_steps,
    "eval_steps": args.eval_steps,
    "seed": args.seed,
}

console.rule("[bold blue]Drone Body Evolution + PPO Controller (figure-8 task)")
console.log(
    f"pop={args.pop}  gens={args.gens}  ppo_steps={args.ppo_steps:,}  "
    f"num_envs={args.num_envs}  device={args.device}"
)
console.log(f"DB → {DB_PATH}")


def evaluate_with_ppo(population: Population) -> Population:
    """Train a PPO policy for every unevaluated body; assign fitness."""
    to_eval = list(population.alive.unevaluated)
    if not to_eval:
        return population

    for ind in track(to_eval, description="PPO evaluation…", console=console):
        t0 = time.time()
        fit, model = _train_and_eval(ind.genotype, cfg)
        ind.fitness = fit
        if model is not None:
            # Serialize the full SB3 model (weights + spaces + hyperparams)
            # into the individual's tags so it's stored in the DB alongside
            # fitness, genotype, and generation info.
            ind.tags["policy_b64"] = _model_to_b64(model)
        console.log(
            f"  id={ind.id}  mean_reward={fit:.3f}  ({time.time() - t0:.1f}s)"
        )

    finite = [
        ind.fitness_ for ind in population.alive.evaluated
        if ind.fitness_ is not None and np.isfinite(ind.fitness_)
    ]
    if finite:
        console.log(
            f"  batch done — max={max(finite):.4f}  mean={np.mean(finite):.4f}"
        )
    return population


# ---------------------------------------------------------------------------
# Run evolution
# ---------------------------------------------------------------------------

initial_pop = Population([Individual() for _ in range(args.pop)])
init_op = initialize_drones(template_handler=template_handler)
eval_op = EAOperation(evaluate_with_ppo)

console.log("Initializing + evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

finite0 = [ind.fitness_ for ind in initial_pop if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if finite0:
    console.log(f"Initial — max={max(finite0):.4f}  mean={np.mean(finite0):.4f}")

generation_ops = [
    parent_tag(n=args.pop),
    crossover_drones(template_handler=template_handler),
    mutate_drones(template_handler=template_handler),
    eval_op,
    truncation_select(n=args.pop),
]

ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=args.gens,
    db_file_path=DB_PATH,
    db_handling="delete",
)

console.rule("[bold green]Evolving")
ea.run()


# ---------------------------------------------------------------------------
# Save best individual (loaded back from DB — tags round-trip through JSON)
# ---------------------------------------------------------------------------

ea.fetch_population(only_alive=False, requires_eval=False)
finite = [ind for ind in ea.population if ind.fitness_ is not None and np.isfinite(ind.fitness_)]
if not finite:
    console.log("[red]No valid individuals found — aborting.[/red]")
    raise SystemExit(1)

best = sorted(finite, key=lambda ind: ind.fitness_, reverse=True)[0]
console.rule("[bold cyan]Best individual")
console.log(f"  id={best.id}  mean_reward={best.fitness_:.4f}  born={best.time_of_birth}")

best_genome = deserialize_genome(best.genotype)
valid_mask = ~np.isnan(best_genome.arms[:, 0])
console.log(f"  active arms: {int(valid_mask.sum())} / {best_genome.arms.shape[0]}")

best_bp = spherical_angular_to_blueprint(best_genome.arms, propsize=args.prop_size)
bp_path = DATA / f"best_blueprint_{RUN_ID}.json"
best_bp.save_json(bp_path)
console.log(f"  blueprint → {bp_path}")

policy_b64 = best.tags.get("policy_b64", "")
if policy_b64:
    policy_path = DATA / f"best_policy_{RUN_ID}.zip"
    _b64_to_policy_file(policy_b64, policy_path)
    console.log(f"  policy    → {policy_path}")
    console.log(
        f"\n  visualize:\n"
        f"    uv run examples/e_drones_ec/6_visualize_evo_results.py --run-dir {DATA}"
    )
else:
    console.log("[yellow]  policy_b64 tag missing — policy not saved[/yellow]")

console.log(f"[bold green]Done. DB → {DB_PATH}")
