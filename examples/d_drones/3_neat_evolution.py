"""NEAT-speciated drone morphology evolution using airevolve's evolve_neat strategy.

Wraps airevolve's ``evolve_neat`` loop (which handles speciation internally)
with the same SphericalAngularDroneGenomeHandler and fitness modes as
example 2, but without ARIEL's EA engine — evolve_neat manages its own
population and does not produce an SQLite archive.

Run:
    python examples/d_drones/3_neat_evolution.py
    python examples/d_drones/3_neat_evolution.py --pop 20 --gens 30
    python examples/d_drones/3_neat_evolution.py --fitness pure_hover --workers 8
    python examples/d_drones/3_neat_evolution.py --compat-threshold 2.5 --target-species 4
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
from airevolve.evolution_tools.evaluators.unified_fitness import UnifiedFitness
from airevolve.evolution_tools.selectors.tournament import tournament_selection
from airevolve.evolution_tools.strategies import evolve_neat

console = Console()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="NEAT-speciated drone evolution")
parser.add_argument("--pop", type=int, default=50, help="Population size")
parser.add_argument("--gens", type=int, default=100, help="Number of generations")
parser.add_argument("--workers", type=int, default=1,
                    help="Parallel workers for evaluation (default: 1)")
parser.add_argument("--min-arms", type=int, default=4, help="Minimum rotor arms")
parser.add_argument("--max-arms", type=int, default=6, help="Maximum rotor arms")
parser.add_argument(
    "--fitness",
    choices=["pure_hover", "edit_distance", "zero"],
    default="pure_hover",
    help="Fitness function",
)
# NEAT-specific
parser.add_argument("--crossover-rate", type=float, default=0.75,
                    help="Fraction of offspring produced by crossover (default 0.75)")
parser.add_argument("--compat-threshold", type=float, default=3.0,
                    help="Initial compatibility distance threshold (default 3.0)")
parser.add_argument("--species-elitism", type=int, default=1,
                    help="Top-N per species copied unchanged each generation (default 1)")
parser.add_argument("--stagnation-limit", type=int, default=15,
                    help="Generations without improvement before species removal (default 15)")
parser.add_argument("--min-species-size", type=int, default=2,
                    help="Minimum offspring per surviving species (default 2)")
parser.add_argument("--target-species", type=int, default=5,
                    help="Target number of species for dynamic threshold (default 5)")
parser.add_argument("--no-adjust-threshold", action="store_true",
                    help="Disable dynamic compatibility threshold adjustment")
parser.add_argument("--interspecies-rate", type=float, default=0.001,
                    help="Probability of cross-species crossover (default 0.001)")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

DATA = Path.cwd() / "__data__" / "drone_neat_evolution"
DATA.mkdir(parents=True, exist_ok=True)

np.random.seed(args.seed)

# ---------------------------------------------------------------------------
# Genome handler + fitness
# ---------------------------------------------------------------------------

PARAMETER_LIMITS = np.array([
    [0.055, 0.17],
    [-np.pi, np.pi],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
    [-np.pi, np.pi],
    [0, 1],
])


fitness_fn = UnifiedFitness(
    brain=None,
    fitness_mode=args.fitness,
    hover_gradient=False,
    per_individual_repair=False,
    is_indirect=False,
    handler_class=SphericalAngularDroneGenomeHandler,
    handler_kwargs={
        "min_max_narms": (args.min_arms, args.max_arms),
        "parameter_limits": PARAMETER_LIMITS,
        "append_arm_chance": 0.1,
        "bilateral_plane_for_symmetry": None,
        "repair": False,
    },
    coordinate_system="spherical",
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

console.rule("[bold blue]Drone NEAT Evolution")
console.log(
    f"pop={args.pop}  gens={args.gens}  fitness={args.fitness}  workers={args.workers}"
)
console.log(
    f"compat_threshold={args.compat_threshold}  target_species={args.target_species}  "
    f"crossover_rate={args.crossover_rate}"
)
console.log(f"arms=[{args.min_arms}, {args.max_arms}]")

# ---------------------------------------------------------------------------
# Run evolve_neat
# evolve_neat manages speciation, crossover, and mutation internally.
# It returns a DataFrame with columns: generation, id, fitness, species_id.
# ---------------------------------------------------------------------------

console.log("Starting NEAT evolution …")

all_individuals = evolve_neat(
    fitness_function=fitness_fn,
    population_size=args.pop,
    num_generations=args.gens,
    crossover_rate=args.crossover_rate,
    parent_selection=tournament_selection,
    genome_handler=SphericalAngularDroneGenomeHandler,
    num_workers=args.workers,
    compatibility_threshold=args.compat_threshold,
    species_elitism=args.species_elitism,
    stagnation_limit=args.stagnation_limit,
    min_species_size=args.min_species_size,
    target_species_count=args.target_species,
    adjust_threshold=not args.no_adjust_threshold,
    interspecies_mating_rate=args.interspecies_rate,
    log_dir=str(DATA),
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

console.rule("[bold yellow]Results")
console.log(f"Total individuals evaluated: {len(all_individuals)}")

last_gen = int(all_individuals["generation"].max())
last_gen_df = all_individuals[all_individuals["generation"] == last_gen]
best_row = last_gen_df.sort_values("fitness", ascending=False).iloc[0]

console.log(
    f"Best (gen {last_gen}): id={best_row['id']}  "
    f"fitness={best_row['fitness']:.4f}  "
    f"species={best_row.get('species_id', 'n/a')}"
)

# Per-generation summary
console.rule("[bold cyan]Per-generation best fitness")
for gen in sorted(all_individuals["generation"].unique()):
    gen_df = all_individuals[all_individuals["generation"] == gen]
    console.log(
        f"  gen {gen:3d}: best={gen_df['fitness'].max():.4f}  "
        f"mean={gen_df['fitness'].mean():.4f}  "
        f"species={gen_df['species_id'].nunique() if 'species_id' in gen_df else '?'}"
    )

console.rule("[bold]Done")
