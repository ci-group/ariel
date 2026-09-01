"""How different are two evolved controllers?

Clustering needs a distance, and for CPPN genomes there is more than one
defensible answer. Two genomes can share a topology but differ in every weight,
or reach the same behaviour by different structure. This module offers each
view separately and a weighted combination of the first two.

Notes
-----
    * Connections have no innovation number in this encoding, so a connection
      is identified by its ``(from_node, to_node)`` endpoints. Two genomes that
      grew the same connection independently are treated as sharing it, which
      is what makes a Jaccard comparison of topologies meaningful here.
    * Every metric is normalised to ``[0, 1]`` so the combined distance is a
      plain weighted sum and clustering thresholds transfer between metrics.

"""

# Standard library
from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Literal

# Third-party libraries
import numpy as np

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray
    from ariel.spatial_ea.hyperneat import CPPNGenome

# Type Aliases
type WeightMetric = Literal["euclidean", "manhattan", "cosine"]
VALID_WEIGHT_METRICS = ("euclidean", "manhattan", "cosine")
type DistanceType = Literal["structural", "weight", "combined", "behavioral"]

# Global constants
MAX_DISTANCE = 1.0
# Weights are clipped to ±3 by mutation, so the widest a single one can differ
# is 6. That bounds the per-connection error and normalises the vector norms.
WEIGHT_RANGE = 6.0
# Structural distance mixes four disagreements; these are the prototype's
# weights, kept so that clustering thresholds carry over.
NODE_COUNT_WEIGHT = 0.25
CONNECTION_COUNT_WEIGHT = 0.25
JACCARD_WEIGHT = 0.35
NODE_KIND_WEIGHT = 0.15
# Fitness is unbounded above; this is the span over which a difference is
# treated as total for behavioural comparison.
FITNESS_SCALE = 10.0
TRAJECTORY_SCALE = 50.0


def _connection_keys(genome: CPPNGenome) -> set[tuple[int, int]]:
    """Identify every connection in a genome by its endpoints.

    Parameters
    ----------
    genome
        The genome to read.

    Returns
    -------
        One ``(from_node, to_node)`` pair per connection.
    """
    return {
        (conn.from_node, conn.to_node) for conn in genome.get("connections", [])
    }


def _node_kinds(genome: CPPNGenome) -> dict[int, tuple[str, int]]:
    """Describe what kind of node each identifier refers to.

    Parameters
    ----------
    genome
        The genome to read.

    Returns
    -------
        Node identifier to its ``(activation, layer)``.
    """
    return {
        node.node_id: (node.activation, node.layer)
        for node in genome.get("nodes", [])
    }


def _enabled_weights(genome: CPPNGenome) -> dict[tuple[int, int], float]:
    """Read the weights of a genome's enabled connections.

    Parameters
    ----------
    genome
        The genome to read.

    Returns
    -------
        Connection endpoints to weight, skipping disabled connections.
    """
    return {
        (conn.from_node, conn.to_node): float(conn.weight)
        for conn in genome.get("connections", [])
        if conn.enabled
    }


def compute_structural_distance(
    genotype1: CPPNGenome,
    genotype2: CPPNGenome,
) -> float:
    """Compare two genomes by topology alone.

    Four disagreements are mixed: how many nodes each has, how many
    connections, how much their connection sets overlap, and whether the nodes
    they share are of the same kind.

    Parameters
    ----------
    genotype1
        First genome.
    genotype2
        Second genome.

    Returns
    -------
        A distance in ``[0, 1]``; zero means identical topology.
    """
    nodes1 = genotype1.get("nodes", [])
    nodes2 = genotype2.get("nodes", [])
    conns1 = genotype1.get("connections", [])
    conns2 = genotype2.get("connections", [])

    max_nodes = max(len(nodes1), len(nodes2), 1)
    node_diff = abs(len(nodes1) - len(nodes2)) / max_nodes

    max_conns = max(len(conns1), len(conns2), 1)
    conn_diff = abs(len(conns1) - len(conns2)) / max_conns

    keys1 = _connection_keys(genotype1)
    keys2 = _connection_keys(genotype2)
    union = keys1 | keys2
    jaccard = 1.0 - (len(keys1 & keys2) / len(union)) if union else 0.0

    kinds1 = _node_kinds(genotype1)
    kinds2 = _node_kinds(genotype2)
    shared = set(kinds1) & set(kinds2)
    if shared:
        mismatches = sum(1 for nid in shared if kinds1[nid] != kinds2[nid])
        kind_diff = mismatches / len(shared)
    else:
        # No shared identifiers at all is maximal disagreement.
        kind_diff = MAX_DISTANCE

    distance = (
        NODE_COUNT_WEIGHT * node_diff
        + CONNECTION_COUNT_WEIGHT * conn_diff
        + JACCARD_WEIGHT * jaccard
        + NODE_KIND_WEIGHT * kind_diff
    )
    return min(distance, MAX_DISTANCE)


def compute_weight_distance(
    genotype1: CPPNGenome,
    genotype2: CPPNGenome,
    metric: WeightMetric = "euclidean",
) -> float:
    """Compare two genomes by the weights of the connections they share.

    Only connections present and enabled in both genomes are compared; two
    genomes sharing nothing are maximally distant by this measure.

    Parameters
    ----------
    genotype1
        First genome.
    genotype2
        Second genome.
    metric
        ``euclidean``, ``manhattan`` or ``cosine``.

    Returns
    -------
        A distance in ``[0, 1]``.

    Raises
    ------
    ValueError
        If the metric is unknown.
    """
    if metric not in VALID_WEIGHT_METRICS:
        msg = f"Unknown weight metric {metric!r}"
        raise ValueError(msg)

    weights1 = _enabled_weights(genotype1)
    weights2 = _enabled_weights(genotype2)
    shared = set(weights1) & set(weights2)
    if not shared:
        return MAX_DISTANCE

    ordered = sorted(shared)
    first = np.array([weights1[key] for key in ordered])
    second = np.array([weights2[key] for key in ordered])

    if metric == "euclidean":
        largest = float(np.sqrt(len(ordered))) * WEIGHT_RANGE
        distance = float(np.linalg.norm(first - second))
    elif metric == "manhattan":
        largest = len(ordered) * WEIGHT_RANGE
        distance = float(np.sum(np.abs(first - second)))
    else:
        norm1 = float(np.linalg.norm(first))
        norm2 = float(np.linalg.norm(second))
        if norm1 == 0.0 or norm2 == 0.0:
            return MAX_DISTANCE
        similarity = float(np.dot(first, second) / (norm1 * norm2))
        return min(max(1.0 - similarity, 0.0), MAX_DISTANCE)

    if largest <= 0.0:
        return 0.0
    return min(distance / largest, MAX_DISTANCE)


def compute_combined_distance(
    genotype1: CPPNGenome,
    genotype2: CPPNGenome,
    structural_weight: float = 0.5,
    weight_weight: float = 0.5,
    weight_metric: WeightMetric = "euclidean",
) -> float:
    """Mix topological and weight distance.

    Parameters
    ----------
    genotype1
        First genome.
    genotype2
        Second genome.
    structural_weight
        Contribution of the topological distance.
    weight_weight
        Contribution of the weight distance.
    weight_metric
        Metric for the weight component.

    Returns
    -------
        A distance in ``[0, 1]``.
    """
    structural = compute_structural_distance(genotype1, genotype2)
    weights = compute_weight_distance(genotype1, genotype2, weight_metric)
    combined = structural_weight * structural + weight_weight * weights
    return min(combined, MAX_DISTANCE)


def compute_behavioral_distance(
    fitness1: float,
    fitness2: float,
    trajectory1: FloatArray | None = None,
    trajectory2: FloatArray | None = None,
) -> float:
    """Compare two individuals by what they did, not what they are.

    Useful for spotting convergent evolution: unrelated genomes that behave
    alike.

    Parameters
    ----------
    fitness1
        Fitness of the first individual.
    fitness2
        Fitness of the second individual.
    trajectory1
        Optional path walked by the first individual.
    trajectory2
        Optional path walked by the second individual.

    Returns
    -------
        A distance in ``[0, 1]``.
    """
    fitness_gap = min(abs(fitness1 - fitness2) / FITNESS_SCALE, MAX_DISTANCE)

    if trajectory1 is None or trajectory2 is None:
        return fitness_gap

    first = np.asarray(trajectory1, dtype=float)
    second = np.asarray(trajectory2, dtype=float)
    if first.size == 0 or second.size == 0:
        return min(0.5 * fitness_gap + 0.5 * MAX_DISTANCE, MAX_DISTANCE)

    shared = min(len(first), len(second))
    separation = float(
        np.mean(np.linalg.norm(first[:shared] - second[:shared], axis=1)),
    )
    trajectory_gap = min(separation / TRAJECTORY_SCALE, MAX_DISTANCE)

    return min(0.5 * fitness_gap + 0.5 * trajectory_gap, MAX_DISTANCE)


def compute_pairwise_distance_matrix(
    genotypes: list[CPPNGenome],
    distance_type: DistanceType = "combined",
    structural_weight: float = 0.5,
    weight_weight: float = 0.5,
    fitness_values: list[float] | None = None,
    trajectories: list[FloatArray] | None = None,
) -> FloatArray:
    """Build the full distance matrix for a population.

    Parameters
    ----------
    genotypes
        Genomes to compare.
    distance_type
        Which metric to use.
    structural_weight
        Structural contribution, for ``combined``.
    weight_weight
        Weight contribution, for ``combined``.
    fitness_values
        Fitness per individual; required for ``behavioral``.
    trajectories
        Optional paths per individual, for ``behavioral``.

    Returns
    -------
        A symmetric ``(n, n)`` matrix with a zero diagonal.

    Raises
    ------
    ValueError
        If the distance type is unknown, or ``behavioral`` was asked for
        without fitness values.
    """
    if distance_type == "behavioral" and fitness_values is None:
        msg = "fitness_values are required for behavioral distance"
        raise ValueError(msg)
    if distance_type not in {
        "structural",
        "weight",
        "combined",
        "behavioral",
    }:
        msg = f"Unknown distance_type {distance_type!r}"
        raise ValueError(msg)

    count = len(genotypes)
    matrix = np.zeros((count, count))

    for i in range(count):
        for j in range(i + 1, count):
            if distance_type == "structural":
                distance = compute_structural_distance(
                    genotypes[i],
                    genotypes[j],
                )
            elif distance_type == "weight":
                distance = compute_weight_distance(genotypes[i], genotypes[j])
            elif distance_type == "combined":
                distance = compute_combined_distance(
                    genotypes[i],
                    genotypes[j],
                    structural_weight=structural_weight,
                    weight_weight=weight_weight,
                )
            else:
                assert fitness_values is not None  # noqa: S101 - checked above
                distance = compute_behavioral_distance(
                    fitness_values[i],
                    fitness_values[j],
                    trajectories[i] if trajectories else None,
                    trajectories[j] if trajectories else None,
                )

            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def compute_genotype_diversity(
    genotypes: list[CPPNGenome],
    distance_type: DistanceType = "combined",
) -> dict[str, float]:
    """Summarise how spread out a population is in genome space.

    Parameters
    ----------
    genotypes
        Genomes to summarise.
    distance_type
        Which metric to use.

    Returns
    -------
        Mean, standard deviation, minimum and maximum pairwise distance, plus
        ``diversity_index`` (the mean). All zero for fewer than two genomes.
    """
    if len(genotypes) < 2:
        return {
            "mean_distance": 0.0,
            "std_distance": 0.0,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "diversity_index": 0.0,
        }

    matrix = compute_pairwise_distance_matrix(genotypes, distance_type)
    upper = matrix[np.triu_indices_from(matrix, k=1)]

    return {
        "mean_distance": float(np.mean(upper)),
        "std_distance": float(np.std(upper)),
        "min_distance": float(np.min(upper)),
        "max_distance": float(np.max(upper)),
        "diversity_index": float(np.mean(upper)),
    }


def _ranked_pairs(
    genotypes: list[CPPNGenome],
    individual_ids: list[int],
    distance_type: DistanceType,
    top_k: int,
    *,
    most_similar: bool,
) -> list[tuple[int, int, float]]:
    """Rank genome pairs by distance.

    Parameters
    ----------
    genotypes
        Genomes to compare.
    individual_ids
        Identifier per genome.
    distance_type
        Which metric to use.
    top_k
        How many pairs to return.
    most_similar
        Rank ascending when true, descending when false.

    Returns
    -------
        ``(id1, id2, distance)`` triples.
    """
    matrix = compute_pairwise_distance_matrix(genotypes, distance_type)
    pairs = [
        (individual_ids[i], individual_ids[j], float(matrix[i, j]))
        for i in range(len(genotypes))
        for j in range(i + 1, len(genotypes))
    ]
    pairs.sort(key=operator.itemgetter(2), reverse=not most_similar)
    return pairs[:top_k]


def find_most_similar_pairs(
    genotypes: list[CPPNGenome],
    individual_ids: list[int],
    distance_type: DistanceType = "combined",
    top_k: int = 5,
) -> list[tuple[int, int, float]]:
    """Find the closest pairs of genomes.

    Parameters
    ----------
    genotypes
        Genomes to compare.
    individual_ids
        Identifier per genome.
    distance_type
        Which metric to use.
    top_k
        How many pairs to return.

    Returns
    -------
        ``(id1, id2, distance)`` triples, closest first.
    """
    return _ranked_pairs(
        genotypes,
        individual_ids,
        distance_type,
        top_k,
        most_similar=True,
    )


def find_most_different_pairs(
    genotypes: list[CPPNGenome],
    individual_ids: list[int],
    distance_type: DistanceType = "combined",
    top_k: int = 5,
) -> list[tuple[int, int, float]]:
    """Find the furthest-apart pairs of genomes.

    Parameters
    ----------
    genotypes
        Genomes to compare.
    individual_ids
        Identifier per genome.
    distance_type
        Which metric to use.
    top_k
        How many pairs to return.

    Returns
    -------
        ``(id1, id2, distance)`` triples, furthest first.
    """
    return _ranked_pairs(
        genotypes,
        individual_ids,
        distance_type,
        top_k,
        most_similar=False,
    )
