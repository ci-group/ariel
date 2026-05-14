"""Drone morphology evolution — spherical genome, configurable fitness + strategy.

Ports airevolve's run_evolution.py to ARIEL's EA engine (SQLite persistence,
Archive post-hoc analysis). Supports the spherical arm encoding and three
fitness modes via mu+lambda or mu,lambda selection.

Run:
    python examples/d_drones/2_drone_evolution.py
    python examples/d_drones/2_drone_evolution.py --fitness pure_hover --strategy plus
    python examples/d_drones/2_drone_evolution.py --fitness edit_distance --workers 8
    python examples/d_drones/2_drone_evolution.py --pop 20 --budget 30 --min-arms 4 --max-arms 6
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from rich.console import Console

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)
from ariel.body_phenotypes.drone import (
    crossover_drones,
    deserialize_genome,
    evaluate_drones,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)
from ariel.ec import EA, Individual, Population
from ariel.ec.archive import Archive

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drone morphology evolution (spherical genome)")
parser.add_argument("--pop", type=int, default=20, help="Population size (mu)")
parser.add_argument("--budget", type=int, default=50, help="Number of EA generations")
parser.add_argument("--workers", type=int, default=1, help="Parallel evaluation workers")
parser.add_argument("--min-arms", type=int, default=4, help="Minimum rotor arms")
parser.add_argument("--max-arms", type=int, default=6, help="Maximum rotor arms")
parser.add_argument("--append-arm-chance", type=float, default=0.1,
                    help="Probability of adding an arm during mutation (default 0.1)")
parser.add_argument(
    "--fitness",
    choices=["pure_hover", "edit_distance", "zero"],
    default="pure_hover",
    help="Fitness function: pure_hover (analytical hover quality [0,3]), "
         "edit_distance (edit distance from target), zero (always 0)",
)
parser.add_argument(
    "--strategy",
    choices=["plus", "comma"],
    default="plus",
    help="Selection strategy: plus (mu+lambda, parents survive) or comma (mu,lambda)",
)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

POP_SIZE: int = args.pop
BUDGET: int = args.budget
N_WORKERS: int = args.workers
FITNESS_MODE: str = args.fitness
STRATEGY: str = args.strategy

DATA = Path.cwd() / "__data__" / "drone_evolution"
DATA.mkdir(parents=True, exist_ok=True)

np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Genome search space
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm magnitude (metres)
    [-np.pi, np.pi],          # arm azimuth (rad)
    [-np.pi / 2, np.pi / 2],  # arm elevation (rad)
    [-np.pi, np.pi],          # motor disc azimuth (rad)
    [-np.pi, np.pi],          # motor disc pitch (rad)
    [0, 1],                   # propeller spin direction (binary)
])

template_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(args.min_arms, args.max_arms),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=args.append_arm_chance,
    bilateral_plane_for_symmetry=None,
    repair=False,
    rnd=np.random.default_rng(args.seed),
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone Evolution")
console.log(
    f"pop={POP_SIZE}  budget={BUDGET}  workers={N_WORKERS}  "
    f"fitness={FITNESS_MODE}  strategy={STRATEGY}"
)
console.log(f"arms=[{args.min_arms}, {args.max_arms}]  db={DATA / 'database.db'}")

# ---------------------------------------------------------------------------
# Initial population
# ---------------------------------------------------------------------------

initial_pop = Population([Individual() for _ in range(POP_SIZE)])

init_op = initialize_drones(template_handler=template_handler)
eval_op = evaluate_drones(fitness_mode=FITNESS_MODE, n_workers=N_WORKERS)

console.log("Initializing and evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

best_init = initial_pop.best(sort="max", attribute="fitness_")[0]
console.log(f"Initial best fitness: {best_init.fitness_:.4f}")

# ---------------------------------------------------------------------------
# Generation ops
# ---------------------------------------------------------------------------
# For mu,lambda (comma) strategy the offspring pool size equals POP_SIZE, so
# the crossover step produces POP_SIZE//2 children and mutation adds nothing
# extra — truncation_select then picks strictly from offspring, discarding
# all parents. For mu+lambda (plus), parents survive alongside offspring.

offspring_n = POP_SIZE if STRATEGY == "comma" else POP_SIZE // 2

generation_ops = [
    parent_tag(n=POP_SIZE),
    crossover_drones(template_handler=template_handler),
    mutate_drones(template_handler=template_handler),
    evaluate_drones(fitness_mode=FITNESS_MODE, n_workers=N_WORKERS),
    truncation_select(n=POP_SIZE),
]

ea = EA(
    population=initial_pop,
    operations=generation_ops,
    num_steps=BUDGET,
    db_file_path=DATA / "database.db",
    db_handling="delete",
)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

console.rule("[bold green]Running EA")
ea.run()

# ---------------------------------------------------------------------------
# Post-hoc analysis
# ---------------------------------------------------------------------------

archive = Archive(DATA / "database.db")

console.rule("[bold yellow]Results")
console.log(f"Archive size: {archive.size} evaluated individuals")
console.log(f"Generation range: {archive.generation_range}")

stats = archive.fitness_stats()
console.log(
    f"Fitness — min: {stats['min']:.4f}  mean: {stats['mean']:.4f}  "
    f"max: {stats['max']:.4f}  std: {stats['std']:.4f}"
)

best = archive.best_individual(fitness_mode="max")
console.rule("[bold cyan]Best Individual")
console.log(
    f"  id={best.id}  fitness={best.fitness_:.4f}  "
    f"born={best.time_of_birth}  died={best.time_of_death}"
)

genome = deserialize_genome(best.genotype)
valid_mask = ~np.isnan(genome.arms[:, 0])
valid_arms = genome.arms[valid_mask]
console.log(f"  active arms: {int(valid_mask.sum())} / {genome.arms.shape[0]}")
console.log("  arm layout (magnitude | azimuth° | elevation° | motor_az° | motor_pitch° | spin):")
for i, arm in enumerate(valid_arms):
    mag, az, el, maz, mpitch, spin = arm
    console.log(
        f"    arm {i}: mag={mag:.3f}m  az={np.degrees(az):.1f}°  "
        f"el={np.degrees(el):.1f}°  motor_az={np.degrees(maz):.1f}°  "
        f"motor_pitch={np.degrees(mpitch):.1f}°  spin={'CW' if spin > 0.5 else 'CCW'}"
    )

console.rule("[bold]Done")
