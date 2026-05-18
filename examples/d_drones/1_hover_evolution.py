"""Drone morphology evolution using ARIEL's EA engine and airevolve's drone simulator.

Evolves drone arm configurations (number, position, orientation, spin direction)
to maximise hover quality (continuous_hover_fitness ∈ [0, 3]) using a mu+lambda
strategy with NEAT-style crossover.

The simulation backend is airevolve's custom ODE dynamics — no MuJoCo involved.
Genomes are persisted in SQLite via ARIEL's EA engine.

Run:
    python examples/d_drones/1_hover_evolution.py
    python examples/d_drones/1_hover_evolution.py --pop 30 --budget 50 --workers 8
    python examples/d_drones/1_hover_evolution.py --fitness gate --workers 4
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from rich.console import Console

# BLAS thread caps must be set before numpy/torch are imported in sub-processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
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
from ariel.ec import EA, EASettings, Individual, Population
from ariel.ec.archive import Archive

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drone hover-quality evolution")
parser.add_argument("--pop", type=int, default=50, help="Population size (mu)")
parser.add_argument("--budget", type=int, default=50, help="Number of EA generations")
parser.add_argument("--workers", type=int, default=1, help="Parallel evaluation workers")
parser.add_argument("--min-arms", type=int, default=3, help="Minimum number of rotor arms")
parser.add_argument("--max-arms", type=int, default=8, help="Maximum number of rotor arms")
parser.add_argument(
    "--fitness",
    choices=["pure_hover", "edit_distance", "zero"],
    default="pure_hover",
    help="Fitness function (pure_hover: analytical hover quality [0,3])",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--viz", action="store_true",
                    help="Simulate best individual after evolution and save hover video")
parser.add_argument("--viz-duration", type=float, default=10.0,
                    help="Hover video duration in seconds (default 10)")
args = parser.parse_args()

POP_SIZE: int = args.pop
BUDGET: int = args.budget
N_WORKERS: int = args.workers
FITNESS_MODE: str = args.fitness

DATA = Path.cwd() / "__data__" / "drone_hover_evolution"
DATA.mkdir(parents=True, exist_ok=True)

np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Genome search space
# ---------------------------------------------------------------------------
# Columns: [magnitude, arm_azimuth, arm_elevation, motor_azimuth, motor_pitch, spin_dir]

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],            # arm magnitude (metres from centre)
    [-np.pi, np.pi],          # arm azimuth (rad)
    [-np.pi / 2, np.pi / 2],  # arm elevation above horizontal (rad)
    [-np.pi, np.pi],          # motor disc azimuth (rad)
    [-np.pi, np.pi],          # motor disc pitch (rad)
    [0, 1],                   # propeller spin direction (binary)
])

template_handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(args.min_arms, args.max_arms),
    parameter_limits=PARAMETER_LIMITS,
    append_arm_chance=0.1,
    bilateral_plane_for_symmetry=None,
    repair=False,
    rnd=np.random.default_rng(args.seed),
)

# ---------------------------------------------------------------------------
# Pre-loop: build and evaluate the initial population outside the EA loop so
# that parent_tag and crossover_drones have valid fitnesses from generation 1.
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone Hover Evolution")
console.log(f"pop={POP_SIZE}  budget={BUDGET}  workers={N_WORKERS}  fitness={FITNESS_MODE}")
console.log(f"arms=[{args.min_arms}, {args.max_arms}]  db={DATA / 'database.db'}")

initial_pop = Population([Individual() for _ in range(POP_SIZE)])

init_op = initialize_drones(template_handler=template_handler)
eval_op = evaluate_drones(fitness_mode=FITNESS_MODE, n_workers=N_WORKERS)

console.log("Initializing and evaluating initial population …")
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

best_init = initial_pop.best(sort="max", attribute="fitness_")[0]
console.log(f"Initial population best fitness: {best_init.fitness_:.4f}")

# ---------------------------------------------------------------------------
# EA loop: mu+lambda strategy with NEAT crossover
#
# Each generation:
#   1. tag top-POP_SIZE alive individuals as parents
#   2. crossover: one child per parent pair (POP_SIZE // 2 offspring)
#   3. mutate:    perturb all unevaluated offspring
#   4. evaluate:  run drone physics for unevaluated individuals only
#   5. select:    keep best POP_SIZE from parents + offspring (plus strategy)
# ---------------------------------------------------------------------------

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

console.rule("[bold green]Running EA")
ea.run()

if args.viz:
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from _viz_best import viz_best_from_db
    viz_best_from_db(DATA / "database.db", DATA / "best_hover.mp4", duration=args.viz_duration)

# ---------------------------------------------------------------------------
# Post-hoc analysis via Archive
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
console.log(f"  id={best.id}  fitness={best.fitness_:.4f}  "
            f"born={best.time_of_birth}  died={best.time_of_death}")

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
