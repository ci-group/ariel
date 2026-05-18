"""Drone morphology evolution using ARIEL's EA engine.

Evolves drone designs encoded as spherical-coordinate arm arrays, evaluated
with ``continuous_hover_fitness`` (fast, analytical) or ``UnifiedFitness``
for gate/edit-distance modes.

Quick start:
    # Pure-hover fitness (CPU, fast smoke test):
    uv run examples/d_drones/1_run_evolution.py \\
        --fitness pure_hover --population-size 20 --generations 10

    # Gate fitness with 4 workers:
    uv run examples/d_drones/1_run_evolution.py \\
        --fitness gate --population-size 50 --generations 100 --n-workers 4

    # Edit-distance diversity measure:
    uv run examples/d_drones/1_run_evolution.py \\
        --fitness edit_distance --population-size 30 --generations 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from ariel.ec.ea import EA, EASettings
from ariel.ec.population import Population
from ariel.body_phenotypes.drone.operations import (
    initialize_drones,
    crossover_drones,
    mutate_drones,
    evaluate_drones,
    parent_tag,
    truncation_select,
)
from ariel.ec.drone.genome_handlers.spherical_angular_genome_handler import (
    SphericalAngularDroneGenomeHandler,
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Drone morphology evolution")
parser.add_argument("--population-size", type=int, default=50,
                    help="Number of individuals in the population (default 50)")
parser.add_argument("--generations", type=int, default=100,
                    help="Number of EA generations (default 100)")
parser.add_argument("--fitness",
                    choices=["pure_hover", "gate", "edit_distance", "zero"],
                    default="pure_hover",
                    help="Fitness mode (default pure_hover)")
parser.add_argument("--n-workers", type=int, default=1,
                    help="Parallel evaluation workers (default 1)")
parser.add_argument("--seed", type=int, default=None,
                    help="Random seed for reproducibility")
parser.add_argument("--save-dir", default="__data__/drone_evolution",
                    help="Output directory (default __data__/drone_evolution)")
parser.add_argument("--viz", action="store_true",
                    help="Simulate best individual after evolution and save hover video")
parser.add_argument("--viz-duration", type=float, default=10.0,
                    help="Hover video duration in seconds (default 10)")
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
else:
    rng = np.random.default_rng()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

print(f"=== Drone Evolution ===")
print(f"  fitness={args.fitness}  pop={args.population_size}  gens={args.generations}"
      f"  workers={args.n_workers}  seed={args.seed}")
print(f"  save_dir={save_dir}")

# ---------------------------------------------------------------------------
# Genome handler (template for all operations)
# ---------------------------------------------------------------------------

handler = SphericalAngularDroneGenomeHandler(
    min_max_narms=(3, 8),
    append_arm_chance=0.1,
    rnd=rng,
)

# ---------------------------------------------------------------------------
# EA operations
# ---------------------------------------------------------------------------

init_op    = initialize_drones(template_handler=handler)
eval_op    = evaluate_drones(fitness_mode=args.fitness, n_workers=args.n_workers)
tag_op     = parent_tag(n=args.population_size)
xover_op   = crossover_drones(template_handler=handler)
mutate_op  = mutate_drones(template_handler=handler)
select_op  = truncation_select(n=args.population_size)

from ariel.ec.individual import Individual

initial_pop = Population([Individual() for _ in range(args.population_size)])
initial_pop = init_op(initial_pop)
initial_pop = eval_op(initial_pop)

ea = EA(
    population=initial_pop,
    operations=[tag_op, xover_op, mutate_op, eval_op, select_op],
    num_steps=args.generations,
    db_file_path=save_dir / "evolution.db",
    db_handling="delete",
)

print("\nStarting evolution …")
ea.run()
print(f"\nDone. Database saved to: {save_dir / 'evolution.db'}")

if args.viz:
    import sys; sys.path.insert(0, str(Path(__file__).parent))
    from _viz_best import viz_best_from_db
    viz_best_from_db(save_dir / "evolution.db", save_dir / "best_hover.mp4", duration=args.viz_duration)
