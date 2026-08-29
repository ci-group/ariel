"""NEAT-style compatibility distance for CPPN networks."""

from __future__ import annotations

import numpy as np

from .network import CPPNNetwork


def cppn_compatibility_distance(
    net1: CPPNNetwork,
    net2: CPPNNetwork,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 0.4,
) -> float:
    """Compute the NEAT compatibility distance between two CPPN networks.

    Parameters
    ----------
    net1, net2 : CPPNNetwork
        Networks to compare.
    c1 : float
        Excess gene coefficient.
    c2 : float
        Disjoint gene coefficient.
    c3 : float
        Mean weight difference coefficient (matching connection genes only).

    Returns
    -------
    float
        Compatibility distance.
    """
    keys1 = set(net1.connections.keys())
    keys2 = set(net2.connections.keys())

    if not keys1 and not keys2:
        return 0.0

    max_innov1 = max(keys1, default=-1)
    max_innov2 = max(keys2, default=-1)

    matching = keys1 & keys2
    excess = 0
    disjoint = 0

    for inn in keys1 - keys2:
        if inn > max_innov2:
            excess += 1
        else:
            disjoint += 1

    for inn in keys2 - keys1:
        if inn > max_innov1:
            excess += 1
        else:
            disjoint += 1

    # Mean absolute weight difference of matching connection genes only
    weight_diffs: list[float] = []
    for inn in matching:
        weight_diffs.append(
            abs(net1.connections[inn].weight - net2.connections[inn].weight)
        )

    avg_weight_diff = float(np.mean(weight_diffs)) if weight_diffs else 0.0

    N = max(len(net1.connections), len(net2.connections))
    if N < 20:
        N = 1

    return (c1 * excess / N) + (c2 * disjoint / N) + (c3 * avg_weight_diff)
