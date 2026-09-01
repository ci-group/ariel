"""Test: distance metrics between CPPN genomes."""

# Standard library
import copy
import random

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.genetics import create_initial_hyperneat_genome
from ariel.spatial_ea.genotype_distance import (
    compute_behavioral_distance,
    compute_combined_distance,
    compute_genotype_diversity,
    compute_pairwise_distance_matrix,
    compute_structural_distance,
    compute_weight_distance,
    find_most_different_pairs,
    find_most_similar_pairs,
)
from ariel.spatial_ea.hyperneat import CPPNConnection, CPPNNode


def _genome(
    connections: list[tuple[int, int, float]],
    activations: dict[int, str] | None = None,
) -> dict:
    """Build a genome from explicit connections."""
    node_ids = sorted({n for conn in connections for n in conn[:2]})
    kinds = activations or {}
    return {
        "nodes": [
            CPPNNode(
                node_id=node_id,
                activation=kinds.get(node_id, "linear"),
                layer=0 if node_id < 4 else 1,
            )
            for node_id in node_ids
        ],
        "connections": [
            CPPNConnection(from_node=a, to_node=b, weight=w)
            for a, b, w in connections
        ],
    }


def test_a_genome_is_zero_distance_from_itself() -> None:
    """Every metric must put a genome at zero from its own copy."""
    genome = create_initial_hyperneat_genome()
    twin = copy.deepcopy(genome)

    assert compute_structural_distance(genome, twin) == 0.0
    assert compute_weight_distance(genome, twin) == 0.0
    assert compute_combined_distance(genome, twin) == 0.0


def test_structure_ignores_weights() -> None:
    """Rewiring nothing but the weights leaves topology identical."""
    first = _genome([(0, 4, 1.0), (1, 4, -1.0)])
    second = _genome([(0, 4, -2.5), (1, 4, 2.5)])

    assert compute_structural_distance(first, second) == 0.0
    assert compute_weight_distance(first, second) > 0.0


def test_weights_ignore_structure_it_does_not_share() -> None:
    """Only connections present in both genomes are compared."""
    first = _genome([(0, 4, 1.0), (1, 4, 1.0)])
    second = _genome([(0, 4, 1.0), (2, 4, 9.0)])

    # The shared connection (0, 4) is identical, so the weight distance is 0.
    assert compute_weight_distance(first, second) == 0.0
    # But the topologies genuinely differ.
    assert compute_structural_distance(first, second) > 0.0


def test_genomes_sharing_nothing_are_maximally_weight_distant() -> None:
    """With no shared connection there is nothing to compare."""
    first = _genome([(0, 4, 1.0)])
    second = _genome([(1, 5, 1.0)])

    assert compute_weight_distance(first, second) == 1.0


def test_node_kind_disagreement_counts() -> None:
    """The same node id doing a different job is a structural difference."""
    first = _genome([(0, 4, 1.0)], activations={4: "sine"})
    second = _genome([(0, 4, 1.0)], activations={4: "gaussian"})

    assert compute_structural_distance(first, second) > 0.0


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_weight_metrics_stay_in_range(metric: str) -> None:
    """Every metric is normalised, so thresholds transfer between them."""
    random.seed(4)
    np.random.seed(4)
    for _ in range(20):
        first = create_initial_hyperneat_genome()
        second = create_initial_hyperneat_genome()
        distance = compute_weight_distance(first, second, metric)
        assert 0.0 <= distance <= 1.0


def test_unknown_weight_metric_is_rejected() -> None:
    """A typo should fail rather than silently pick a default."""
    genome = _genome([(0, 4, 1.0)])

    with pytest.raises(ValueError, match="Unknown weight metric"):
        compute_weight_distance(genome, genome, "manhatten")


def test_combined_distance_respects_its_weights() -> None:
    """Weighting the components changes the mix."""
    first = _genome([(0, 4, 1.0), (1, 4, 1.0)])
    second = _genome([(0, 4, -3.0), (1, 4, 3.0)])

    structural_only = compute_combined_distance(first, second, 1.0, 0.0)
    weight_only = compute_combined_distance(first, second, 0.0, 1.0)

    assert structural_only == 0.0
    assert weight_only > 0.0


def test_behavioral_distance_uses_fitness_alone_without_paths() -> None:
    """Fitness difference is the fallback comparison."""
    assert compute_behavioral_distance(1.0, 1.0) == 0.0
    assert compute_behavioral_distance(0.0, 5.0) == pytest.approx(0.5)
    # Bounded above.
    assert compute_behavioral_distance(0.0, 1000.0) == 1.0


def test_behavioral_distance_uses_paths_when_given() -> None:
    """Two individuals that walked the same path are behaviourally close."""
    path = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    far = np.array([[40.0, 40.0], [41.0, 41.0], [42.0, 42.0]])

    same = compute_behavioral_distance(1.0, 1.0, path, path)
    apart = compute_behavioral_distance(1.0, 1.0, path, far)

    assert same == 0.0
    assert apart > same


def test_distance_matrix_is_symmetric_with_a_zero_diagonal() -> None:
    """A distance matrix must be well formed for clustering to accept it."""
    random.seed(2)
    np.random.seed(2)
    genomes = [create_initial_hyperneat_genome() for _ in range(6)]

    matrix = compute_pairwise_distance_matrix(genomes)

    assert matrix.shape == (6, 6)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_behavioral_matrix_needs_fitness() -> None:
    """Asking for behavioural distance without fitness is an error."""
    genomes = [create_initial_hyperneat_genome() for _ in range(2)]

    with pytest.raises(ValueError, match="fitness_values"):
        compute_pairwise_distance_matrix(genomes, "behavioral")


def test_unknown_distance_type_is_rejected() -> None:
    """A typo should fail rather than silently pick a default."""
    genomes = [create_initial_hyperneat_genome() for _ in range(2)]

    with pytest.raises(ValueError, match="Unknown distance_type"):
        compute_pairwise_distance_matrix(genomes, "structual")


def test_diversity_of_a_population() -> None:
    """Diversity summarises the pairwise distances."""
    random.seed(6)
    np.random.seed(6)
    genomes = [create_initial_hyperneat_genome() for _ in range(5)]

    diversity = compute_genotype_diversity(genomes)

    assert diversity["min_distance"] <= diversity["mean_distance"]
    assert diversity["mean_distance"] <= diversity["max_distance"]
    assert diversity["diversity_index"] == diversity["mean_distance"]


def test_diversity_of_a_lone_individual_is_zero() -> None:
    """One genome has nothing to be diverse from."""
    diversity = compute_genotype_diversity(
        [create_initial_hyperneat_genome()],
    )

    assert all(value == 0.0 for value in diversity.values())


def test_similar_and_different_pairs_are_opposite_ends() -> None:
    """The closest and furthest pairs bracket the population."""
    random.seed(8)
    np.random.seed(8)
    genomes = [create_initial_hyperneat_genome() for _ in range(6)]
    ids = list(range(100, 106))

    closest = find_most_similar_pairs(genomes, ids, top_k=3)
    furthest = find_most_different_pairs(genomes, ids, top_k=3)

    assert len(closest) == 3
    assert closest[0][2] <= closest[-1][2]
    assert furthest[0][2] >= furthest[-1][2]
    assert closest[0][2] <= furthest[0][2]
    # Identifiers are carried through, not indices.
    assert all(a in ids and b in ids for a, b, _ in closest)
