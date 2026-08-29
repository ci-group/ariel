"""Unified fitness wrapper for evolution.

Subsumes the prior `_RepairAndEvaluateFitness` (formerly in the deleted
`examples/evolution/run_evolution_with_lee_tuning.py`) and
`_CombinedHoverGateFitness` (`experimentation/run_combined_hover_gate_evolution.py`,
`ppsn_2026_submission` branch). Picklable so it works inside multiprocessing
Pool workers.

Modes:
  - 'gate':          run brain (rl/lee), return gates_passed.
                     With hover_gradient=True: skip brain on non-hoverable
                     drones and return continuous_hover_fitness ∈ [0, 3].
  - 'pure_hover':    return continuous_hover_fitness, no brain.
  - 'edit_distance': return -edit_distance to STANDARD_HEXACOPTER target.
  - 'zero':          return 0.0.
"""
import os
import pickle

import numpy as np

from ariel.ec.drone.evaluators.hover_fitness import continuous_hover_fitness
from ariel.ec.drone.genome_handlers.repair_workflow import (
    stage1_optimization_repair,
    stage2_hover_check,
    stage3_hover_repair,
)
from ariel.ec.drone.genome_handlers.operators.optimization_repair_operator import (
    OptimizationRepairConfig,
)


# Standard hexacopter target for edit-distance fitness.
# 6 arms equally spaced at 60°, magnitude 0.13, motors pointing up
# (motor_pitch=π → upward thrust in NED), alternating CCW/CW.
STANDARD_HEXACOPTER = np.array([
    [0.13,           0.0, 0.0, np.pi, 0.0, 0],  #   0°
    [0.13,    np.pi / 3.0, 0.0, np.pi, 0.0, 1],  #  60°
    [0.13,  2 * np.pi / 3.0, 0.0, np.pi, 0.0, 0],  # 120°
    [0.13,         np.pi, 0.0, np.pi, 0.0, 1],  # 180°
    [0.13, -2 * np.pi / 3.0, 0.0, np.pi, 0.0, 0],  # 240°
    [0.13,   -np.pi / 3.0, 0.0, np.pi, 0.0, 1],  # 300°
])

# Per-parameter [min, max] bounds matching shared_params in
# get_genome_handler_config (examples/evolution/run_evolution.py).
EDIT_DISTANCE_MIN = np.array([0.055, -np.pi, -np.pi / 2, -np.pi, -np.pi, 0])
EDIT_DISTANCE_MAX = np.array([0.17,   np.pi,  np.pi / 2,  np.pi,  np.pi, 1])


class UnifiedFitness:
    """Picklable fitness wrapper. Owns decode, repair, and fitness dispatch."""

    def __init__(
        self,
        *,
        brain,                       # 'rl' | 'lee' | None
        fitness_mode,                # 'gate' | 'pure_hover' | 'edit_distance' | 'zero'
        hover_gradient,              # bool
        per_individual_repair,       # bool
        is_indirect,                 # cppn / hybrid-cppn
        handler_class,               # for re-decoding indirect genomes
        handler_kwargs,
        coordinate_system,           # 'spherical' | 'cartesian' | 'cppn' | 'hybrid-cppn'
        brain_kwargs=None,
    ):
        self.brain = brain
        self.fitness_mode = fitness_mode
        self.hover_gradient = hover_gradient
        self.per_individual_repair = per_individual_repair
        self.is_indirect = is_indirect
        self.handler_class = handler_class
        self.handler_kwargs = handler_kwargs
        self.coordinate_system = coordinate_system
        self.brain_kwargs = brain_kwargs or {}

    def __call__(self, genome, ind_save_dir):
        # 1. Save genome.
        if ind_save_dir is not None:
            os.makedirs(ind_save_dir, exist_ok=True)
            if self.is_indirect:
                with open(os.path.join(ind_save_dir, "genotype.pkl"), "wb") as f:
                    pickle.dump(genome, f)
            else:
                np.save(os.path.join(ind_save_dir, "genome.npy"), genome)

        # 2. Decode indirect → phenotype.
        if self.is_indirect:
            handler = self.handler_class(genome=genome, **self.handler_kwargs)
            phenotype = handler.get_phenotype()
            repair_coord = "spherical"  # CPPN/hybrid-cppn decode to spherical
        else:
            # Mutation returns a SphericalNeatGenome wrapper (or similar) whose
            # arm matrix is in `.arms`; the EA hands these back to fitness
            # functions verbatim. Initial-pop genomes are already plain ndarrays.
            arms = genome.arms if hasattr(genome, "arms") else genome
            phenotype = np.asarray(arms, dtype=np.float64)
            repair_coord = self.coordinate_system

        # 3. Per-individual 3-stage repair. On failure:
        #    - hover_gradient on  → fall back to hover_fit on the raw phenotype
        #      so the gradient signal isn't lost.
        #    - hover_gradient off → return 0 (matches _RepairAndEvaluateFitness).
        if self.per_individual_repair:
            raw_phenotype = phenotype
            phenotype = self._repair(phenotype, repair_coord)
            if phenotype is None:
                if self.hover_gradient and self.fitness_mode == "gate":
                    return continuous_hover_fitness(raw_phenotype)
                return 0.0

        # 4. Dispatch by fitness_mode.
        if self.fitness_mode == "zero":
            return 0.0

        if self.fitness_mode == "pure_hover":
            return continuous_hover_fitness(phenotype)

        if self.fitness_mode == "edit_distance":
            from ariel.ec.drone.evaluators.edit_distance import compute_edit_distance
            return -compute_edit_distance(
                STANDARD_HEXACOPTER, phenotype,
                EDIT_DISTANCE_MIN, EDIT_DISTANCE_MAX,
            )

        if self.fitness_mode == "gate":
            if self.hover_gradient:
                hover_fit = continuous_hover_fitness(phenotype)
                can_hover, _ = stage2_hover_check(
                    phenotype, verbose=False, allow_spinning=False,
                )
                if not can_hover:
                    return hover_fit
                return hover_fit + self._run_brain(phenotype, ind_save_dir)
            return self._run_brain(phenotype, ind_save_dir)

        raise ValueError(f"Unknown fitness_mode: {self.fitness_mode}")

    def _repair(self, phenotype, repair_coord):
        """3-stage repair: hover_check → opt_repair → hover_repair. None on fail."""
        can_hover, _ = stage2_hover_check(
            phenotype, verbose=False, allow_spinning=False,
        )
        if not can_hover:
            return None

        repair_config = OptimizationRepairConfig(fixed_params=[3, 4])
        repaired, _ = stage1_optimization_repair(
            phenotype, coordinate_system=repair_coord,
            config=repair_config, verbose=False,
        )
        if repaired is None:
            return None

        final, _ = stage3_hover_repair(
            repaired, coordinate_system=repair_coord, verbose=False,
        )
        return final

    def _run_brain(self, phenotype, ind_save_dir):
        if self.brain == "rl":
            from ariel.ec.drone.evaluators import gate_train
            return gate_train.evaluate_individual(
                phenotype, ind_save_dir,
                self.brain_kwargs["training_ts"],
                self.brain_kwargs["num_envs"],
                self.brain_kwargs["gate_cfg"],
                self.brain_kwargs["device"],
                max_steps=self.brain_kwargs.get("max_steps", 1200),
            )
        if self.brain == "lee":
            from ariel.ec.drone.evaluators import lee_tune_evaluator
            return lee_tune_evaluator.evaluate_individual_with_tuning(
                phenotype, ind_save_dir,
                gate_cfg=self.brain_kwargs["gate_cfg"],
                max_evals=self.brain_kwargs["max_evals"],
                num_workers=self.brain_kwargs["cma_workers"],
                sim_time=self.brain_kwargs["sim_time"],
                dt=self.brain_kwargs["dt"],
                timeout=self.brain_kwargs["timeout"],
            )
        raise ValueError(
            f"fitness_mode='gate' requires brain in ('rl','lee'), got {self.brain!r}"
        )
