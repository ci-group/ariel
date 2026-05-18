"""Two-stage CMA-ES tuning of Lee controller gains for a drone morphology.

Mirrors src/airevolve/examples/tuning/tune_lee_controller_gates.py using
ARIEL imports.

Usage:
    # Tune default 2-inch quad on figure-8 (100 CMA evaluations, 4 workers):
    uv run examples/d_drones/3_tune_lee.py --gates figure8

    # Tune individual loaded from a .npy file:
    uv run examples/d_drones/3_tune_lee.py \\
        --genome-file __data__/drone_evolution/genome.npy \\
        --gates circle --max-evals 200 --workers 8

Requires: cma  (uv pip install cma)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ariel.ec.drone.evaluators.lee_tune_evaluator import (
    evaluate_individual_with_tuning,
    optimize_controller_for_morphology,
)
from ariel.simulation.drone.controllers.utils.gate_configs import GATE_CONFIGS
from ariel.simulation.drone import create_standard_propeller_config

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Two-stage CMA-ES Lee controller tuning")
parser.add_argument("--gates", choices=["figure8", "circle", "slalom", "backandforth"],
                    default="figure8", help="Gate configuration (default figure8)")
parser.add_argument("--genome-file", default=None,
                    help="Path to .npy genome file. If omitted, uses a default 2-inch X-quad.")
parser.add_argument("--max-evals", type=int, default=100,
                    help="Maximum CMA-ES evaluations (default 100)")
parser.add_argument("--workers", type=int, default=4,
                    help="Parallel workers for CMA-ES (default 4)")
parser.add_argument("--sim-time", type=float, default=20.0,
                    help="Simulation time per evaluation in seconds (default 20)")
parser.add_argument("--dt", type=float, default=0.005,
                    help="Simulation time step (default 0.005)")
parser.add_argument("--timeout", type=float, default=30.0,
                    help="Timeout per evaluation in seconds (default 30)")
parser.add_argument("--save-dir", default="__data__/tuning",
                    help="Output directory (default __data__/tuning)")
args = parser.parse_args()

save_dir = Path(args.save_dir)
save_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

if args.genome_file:
    individual = np.load(args.genome_file, allow_pickle=True).astype(np.float32)
    print(f"Loaded genome from {args.genome_file}  shape={individual.shape}")
else:
    # Default: 2-inch X-quad as a (4, 6) spherical genome
    # [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    ARM_LENGTH = 0.07
    individual = np.array([
        [ARM_LENGTH,  np.pi/4,  0.0, 0.0, 0.0, 1.0],
        [ARM_LENGTH,  3*np.pi/4, 0.0, 0.0, 0.0, 0.0],
        [ARM_LENGTH, -3*np.pi/4, 0.0, 0.0, 0.0, 1.0],
        [ARM_LENGTH, -np.pi/4,  0.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)
    print(f"Using default 2-inch X-quad individual  shape={individual.shape}")

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

gate_config = GATE_CONFIGS[args.gates]
print(f"\nGate config: {args.gates}  ({len(gate_config.gate_pos)} gates)")
print(f"CMA-ES: max_evals={args.max_evals}  workers={args.workers}")
print(f"Output: {save_dir}\n")

result = optimize_controller_for_morphology(
    individual=individual,
    gate_config=gate_config,
    max_evaluations=args.max_evals,
    num_workers=args.workers,
    sim_time=args.sim_time,
    dt=args.dt,
    timeout_per_eval=args.timeout,
    save_dir=str(save_dir),
)

print("\n=== Tuning Results ===")
print(f"  Gates passed : {result.get('gates_passed', 0)}")
print(f"  Best gains   : {result.get('best_gains')}")
print(f"  Tuning time  : {result.get('tuning_time_seconds', 0):.1f}s")
print(f"  Success      : {result.get('success', False)}")
print(f"\nResults saved to: {save_dir}")
