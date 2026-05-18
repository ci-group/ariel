"""Demonstration of the optimization-based genome repair operator.

Shows how ``OptimizationBasedRepairOperator`` detects arm collisions (arm-arm,
arm-core) in a drone genome and repairs them via constrained optimisation
(SciPy minimize) while minimising the change to the original genome.

Four demos:
    1. Basic collision repair
    2. Repair with a custom configuration
    3. Before/after comparison (parameter by parameter)
    4. Functional API (``optimization_repair_individual``)

Run:
    python examples/d_drones/9_repair_demo.py
"""

from __future__ import annotations

import numpy as np

from airevolve.evolution_tools.genome_handlers.operators.optimization_repair_operator import (
    OptimizationBasedRepairOperator,
    OptimizationRepairConfig,
    optimization_repair_individual,
)
from airevolve.evolution_tools.genome_handlers.operators.repair_base import RepairConfig


# ---------------------------------------------------------------------------
# Test genome builders
# ---------------------------------------------------------------------------

def create_arm_collision_genome() -> np.ndarray:
    """Two arms pointing almost the same direction — they will collide."""
    return np.array([
        [0.12,  0.10, 0.0,        0.5, 0.0, 1],
        [0.12,  0.15, 0.0,        0.5, 0.0, 0],
        [0.12,  np.pi,     0.0,  -0.5, 0.0, 1],
        [0.12, -np.pi / 2, 0.0,  -0.5, 0.0, 0],
    ], dtype=float)


def create_core_collision_genome() -> np.ndarray:
    """Arms angled steeply inward — will intersect the central core sphere."""
    return np.array([
        [0.3,  0.0,          np.pi / 6, 0.5, 0.0, 1],
        [0.3,  np.pi / 2,    np.pi / 6, 0.5, 0.0, 1],
        [0.3,  np.pi,        np.pi / 6, 0.5, 0.0, 1],
        [0.3, -np.pi / 2,   np.pi / 6, 0.5, 0.0, 1],
    ], dtype=float)


def print_genome(genome: np.ndarray, label: str = "") -> None:
    if label:
        print(f"\n{label}:")
    cols = ["r", "theta", "phi", "motor_pitch", "motor_yaw", "dir"]
    header = "  " + "  ".join(f"{c:>12}" for c in cols)
    print(header)
    for i, row in enumerate(genome):
        vals = "  ".join(f"{v:>12.4f}" for v in row)
        print(f"  arm {i}: {vals}")


# ---------------------------------------------------------------------------
# Demo 1: Basic repair
# ---------------------------------------------------------------------------

def demo_basic_repair() -> None:
    print("=" * 70)
    print("DEMO 1: Basic Collision Repair")
    print("=" * 70)

    genome = create_arm_collision_genome()
    print_genome(genome, "Original genome")

    repair_op = OptimizationBasedRepairOperator(verbose=True)
    is_valid = repair_op.validate(genome)
    print(f"\nOriginal genome valid: {is_valid}")

    print("\nRepairing …")
    repaired = repair_op.repair(genome)
    print_genome(repaired, "Repaired genome")

    is_valid_after = repair_op.validate(repaired)
    print(f"\nRepaired genome valid: {is_valid_after}")
    changes = np.abs(repaired - genome)
    print(f"Total L2 change: {np.linalg.norm(changes):.6f}")


# ---------------------------------------------------------------------------
# Demo 2: Custom configuration
# ---------------------------------------------------------------------------

def demo_custom_config() -> None:
    print("\n\n" + "=" * 70)
    print("DEMO 2: Repair with Custom Configuration")
    print("=" * 70)

    genome = create_core_collision_genome()
    print_genome(genome, "Original genome")

    opt_config = OptimizationRepairConfig(
        disc_radius=0.12,
        disc_height=0.0,
        core_radius=0.06,
        propeller_radius=0.0254,
        optimization_method="SLSQP",
        max_iterations=500,
        constraint_tolerance=1e-6,
    )
    repair_op = OptimizationBasedRepairOperator(optimization_config=opt_config, verbose=False)

    print("\nRepairing with custom config …")
    repaired = repair_op.repair(genome)
    print_genome(repaired, "Repaired genome")

    is_valid = repair_op.validate(repaired)
    print(f"\nRepaired genome valid: {is_valid}")


# ---------------------------------------------------------------------------
# Demo 3: Before/after comparison
# ---------------------------------------------------------------------------

def demo_comparison() -> None:
    print("\n\n" + "=" * 70)
    print("DEMO 3: Before / After Comparison")
    print("=" * 70)

    genome = create_arm_collision_genome()
    repair_op = OptimizationBasedRepairOperator(verbose=False)
    repaired = repair_op.repair(genome)

    cols = ["r", "theta", "phi", "motor_pitch", "motor_yaw", "dir"]
    print(f"\n{'Arm':<6}  {'Param':<14}  {'Before':>10}  {'After':>10}  {'|Δ|':>10}")
    print("-" * 60)
    for arm_i, (orig_row, rep_row) in enumerate(zip(genome, repaired)):
        for col, (o, r) in zip(cols, zip(orig_row, rep_row)):
            delta = abs(r - o)
            marker = " ←" if delta > 1e-6 else ""
            print(f"arm {arm_i}  {col:<14}  {o:>10.4f}  {r:>10.4f}  {delta:>10.6f}{marker}")

    changes = np.abs(repaired - genome)
    print("\nSummary:")
    print(f"  Mean |Δ|: {np.mean(changes):.6f}")
    print(f"  Max  |Δ|: {np.max(changes):.6f}")
    print(f"  L2   |Δ|: {np.linalg.norm(changes):.6f}")


# ---------------------------------------------------------------------------
# Demo 4: Functional API
# ---------------------------------------------------------------------------

def demo_functional_api() -> None:
    print("\n\n" + "=" * 70)
    print("DEMO 4: Functional API")
    print("=" * 70)

    genome = create_core_collision_genome()
    print_genome(genome, "Original genome")

    config = OptimizationRepairConfig(
        disc_radius=0.15,
        core_radius=0.05,
        optimization_method="SLSQP",
    )

    print("\nRepairing via functional API …")
    repaired = optimization_repair_individual(genome, config=config, verbose=True)
    print_genome(repaired, "Repaired genome")

    changes = np.abs(repaired - genome)
    print(f"\nL2 change: {np.linalg.norm(changes):.6f}")


# ---------------------------------------------------------------------------
# Run all demos
# ---------------------------------------------------------------------------

print("\n")
print("*" * 70)
print("OPTIMIZATION-BASED REPAIR OPERATOR DEMONSTRATION")
print("*" * 70)

demo_basic_repair()
demo_custom_config()
demo_comparison()
demo_functional_api()

print("\n" + "*" * 70)
print("DEMONSTRATION COMPLETE")
print("*" * 70)
print("\nKey takeaways:")
print("  1. The repair minimises changes to the original genome")
print("  2. It respects disc-based attachment geometry")
print("  3. It eliminates cylinder-cylinder and cylinder-core collisions")
print("  4. Configuration (disc_radius, core_radius, propeller_radius) is flexible")
print("  5. Both class-based and functional APIs are available")
print()
