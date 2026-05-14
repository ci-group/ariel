"""Serialization between airevolve's SphericalNeatGenome and ARIEL's JSONIterable.

Genome format stored in Individual.genotype_ (SQLite JSON column):
    {
        "arms": [[mag, theta, phi, motor_theta, motor_phi, dir], ..., null, ...],
        "innovation_ids": [0, 1, -1, ...]
    }

Active arm slots are 6-element float lists; empty slots are JSON null (Python
None), avoiding the NaN-in-JSON problem while preserving exact slot positions
needed for NEAT crossover alignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
        SphericalNeatGenome,
    )


def serialize_genome(genome: "SphericalNeatGenome") -> dict[str, Any]:
    """Convert a SphericalNeatGenome to a JSON-serialisable dict.

    NaN arm slots become None (JSON null). Innovation IDs are stored
    inline alongside the arm array so crossover alignment survives the
    DB round-trip.
    """
    arms_list: list[list[float] | None] = []
    for i in range(genome.arms.shape[0]):
        if np.isnan(genome.arms[i, 0]):
            arms_list.append(None)
        else:
            arms_list.append(genome.arms[i].tolist())
    return {
        "arms": arms_list,
        "innovation_ids": genome.innovation_ids.tolist(),
    }


def deserialize_genome(data: dict[str, Any]) -> "SphericalNeatGenome":
    """Reconstruct a SphericalNeatGenome from a stored dict.

    None slots become NaN-padded arm rows. Innovation IDs are restored
    to their original integer array so NEAT crossover can align genes.
    """
    from airevolve.evolution_tools.genome_handlers.spherical_angular_genome_handler import (
        SphericalNeatGenome,
    )

    arms_list = data["arms"]
    max_narms = len(arms_list)
    arms = np.full((max_narms, 6), np.nan, dtype=np.float64)
    for i, arm in enumerate(arms_list):
        if arm is not None:
            arms[i] = arm
    innovation_ids = np.array(data["innovation_ids"], dtype=int)
    return SphericalNeatGenome(arms=arms, innovation_ids=innovation_ids)
