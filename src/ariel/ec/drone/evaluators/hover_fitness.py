"""Continuous hover fitness — gradient signal in [0, 3] for non-hoverable drones.

Ported from experimentation/run_combined_hover_gate_evolution.py
(ppsn_2026_submission branch, lines 279-325).
"""
import numpy as np
from numpy.linalg import norm, eig

G = 9.81
C_AUTHORITY = 300.0  # sqrt(lambda_min) half-saturation constant


def continuous_hover_fitness(phenotype) -> float:
    """Continuous hover fitness in [0, 3]. Higher = closer to hoverable.

    Sum of three [0, 1] terms:
      - rank feasibility:  (rank(Bf) + rank(Bm)) / 6
      - force capability:  min(||Bf @ 1|| / G, 2) / 2
      - torque balance:    (lambda_min / lambda_max) * sqrt(lambda_min) / (sqrt(lambda_min) + C)
    """
    from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim

    sim = get_sim(phenotype)
    if sim is None:
        return 0.0

    Bf = sim.Bf  # (3, n_props)
    Bm = sim.Bm  # (3, n_props)

    rank_f = np.linalg.matrix_rank(Bf)
    rank_m = np.linalg.matrix_rank(Bm)
    f_rank = (rank_f + rank_m) / 6.0

    n_props = Bf.shape[1]
    eta_max = np.ones(n_props)
    f_vec = Bf @ eta_max
    thrust_ratio = norm(f_vec) / G
    f_force = min(thrust_ratio, 2.0) / 2.0

    gram_m = Bm @ Bm.T
    eigs = np.real(eig(gram_m)[0])
    eigs = np.maximum(eigs, 0.0)

    lambda_min = np.min(eigs)
    lambda_max = np.max(eigs)

    condition = lambda_min / (lambda_max + 1e-12)
    authority = np.sqrt(lambda_min)
    authority_sat = authority / (authority + C_AUTHORITY)
    f_torque = condition * authority_sat

    return float(f_rank + f_force + f_torque)
