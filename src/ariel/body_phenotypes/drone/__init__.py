"""ARIEL drone body phenotype package.

Bridges airevolve's drone genome handlers and physics simulator into ARIEL's
EA engine. Genomes are stored as JSON in the SQLite database; airevolve's
custom ODE dynamics are used for fitness evaluation.
"""

from ariel.body_phenotypes.drone.genome import deserialize_genome, serialize_genome
from ariel.body_phenotypes.drone.operations import (
    crossover_drones,
    evaluate_drones,
    initialize_drones,
    mutate_drones,
    parent_tag,
    truncation_select,
)

__all__ = [
    "crossover_drones",
    "deserialize_genome",
    "evaluate_drones",
    "initialize_drones",
    "mutate_drones",
    "parent_tag",
    "serialize_genome",
    "truncation_select",
]
