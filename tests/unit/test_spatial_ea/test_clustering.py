"""Test: clustering genomes and relating those clusters to space."""

# Standard library
import copy
import random

# Third-party libraries
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.clustering import (
    _circular_mean,
    analyze_spatial_clustering,
    cluster_dbscan,
    cluster_hierarchical,
    cluster_kmeans,
    find_optimal_clusters,
    reduce_dimensions_pca,
    reduce_dimensions_tsne,
    spatial_distance_matrix,
    spatial_silhouette,
)
from ariel.spatial_ea.genetics import (
    create_initial_hyperneat_genome,
    mutate_genome,
)
from ariel.spatial_ea.genotype_distance import compute_pairwise_distance_matrix

WORLD = (10.0, 10.0)


def _two_families(per_family: int = 6) -> list[dict]:
    """Build two genuinely distinct lineages of genomes."""
    random.seed(1)
    np.random.seed(1)
    population = []
    for _ in range(2):
        founder = create_initial_hyperneat_genome()
        for _ in range(per_family):
            child = copy.deepcopy(founder)
            mutate_genome(
                child,
                weight_mutation_rate=1.0,
                weight_mutation_power=0.2,
                add_connection_rate=0.0,
                add_node_rate=0.0,
            )
            population.append(child)
    return population


# -- Clustering ----------------------------------------------------------------
def test_hierarchical_recovers_known_families() -> None:
    """Two lineages should come back as two clusters."""
    distances = compute_pairwise_distance_matrix(_two_families())

    result = cluster_hierarchical(distances, n_clusters=2)

    assert result.n_clusters == 2
    assert sorted(result.cluster_sizes.values()) == [6, 6]
    assert result.silhouette is not None
    assert result.silhouette > 0.5
    # The first family and the second must not share a label.
    assert len(set(result.cluster_labels[:6])) == 1
    assert result.cluster_labels[0] != result.cluster_labels[-1]


def test_dbscan_finds_the_same_structure_without_being_told_k() -> None:
    """Density clustering chooses the cluster count itself."""
    distances = compute_pairwise_distance_matrix(_two_families())

    result = cluster_dbscan(distances, eps=0.25, min_samples=2)

    assert result.n_clusters == 2
    assert result.num_noise == 0


def test_dbscan_can_mark_noise() -> None:
    """A lone outlier should be left unassigned, not forced into a cluster."""
    # Two tight groups plus one point far from both.
    distances = np.array([
        [0.0, 0.05, 0.9, 0.9, 0.95],
        [0.05, 0.0, 0.9, 0.9, 0.95],
        [0.9, 0.9, 0.0, 0.05, 0.95],
        [0.9, 0.9, 0.05, 0.0, 0.95],
        [0.95, 0.95, 0.95, 0.95, 0.0],
    ])

    result = cluster_dbscan(distances, eps=0.1, min_samples=2)

    assert result.n_clusters == 2
    assert result.num_noise == 1


def test_kmeans_clusters_an_embedding() -> None:
    """K-means works on coordinates rather than distances."""
    distances = compute_pairwise_distance_matrix(_two_families())
    embedding = reduce_dimensions_pca(distances)

    result = cluster_kmeans(embedding, n_clusters=2)

    assert result.n_clusters == 2
    assert result.cluster_centers is not None
    assert result.cluster_centers.shape[0] == 2


def test_ward_linkage_is_rejected_for_distances() -> None:
    """Ward needs coordinates; saying so beats a cryptic sklearn error."""
    distances = compute_pairwise_distance_matrix(_two_families(3))

    with pytest.raises(ValueError, match="Ward linkage"):
        cluster_hierarchical(distances, 2, "ward")


def test_quality_scores_are_absent_for_a_single_cluster() -> None:
    """Silhouette is undefined when everything is in one group."""
    distances = np.zeros((4, 4))

    result = cluster_hierarchical(distances, n_clusters=1)

    assert result.n_clusters == 1
    assert result.silhouette is None


def test_optimal_cluster_count_finds_two() -> None:
    """Silhouette selection should recover the true structure."""
    distances = compute_pairwise_distance_matrix(_two_families())

    best, scores = find_optimal_clusters(distances, max_clusters=6)

    assert best == 2
    assert scores[2] == max(scores.values())


# -- Embeddings ----------------------------------------------------------------
def test_pca_embedding_shape() -> None:
    """PCA gives one coordinate row per individual."""
    distances = compute_pairwise_distance_matrix(_two_families(4))

    embedding = reduce_dimensions_pca(distances)

    assert embedding.shape == (8, 2)
    assert np.isfinite(embedding).all()


def test_tsne_clamps_perplexity_to_the_population() -> None:
    """t-SNE rejects a perplexity at or above the sample count."""
    distances = compute_pairwise_distance_matrix(_two_families(2))

    embedding = reduce_dimensions_tsne(distances, perplexity=50.0)

    assert embedding.shape == (4, 2)
    assert np.isfinite(embedding).all()


# -- Space ---------------------------------------------------------------------
def test_circular_mean_wraps() -> None:
    """The mean of two points either side of the seam is the seam."""
    assert _circular_mean(np.array([0.2, 9.8]), 10.0) == pytest.approx(
        0.0,
        abs=1e-6,
    )
    # Away from the seam it agrees with the plain mean.
    assert _circular_mean(np.array([4.0, 6.0]), 10.0) == pytest.approx(5.0)


def test_spatial_distance_matrix_respects_wrapping() -> None:
    """Across the seam, the toroidal distance is the short way round."""
    positions = np.array([[0.5, 5.0, 0.1], [9.5, 5.0, 0.1]])

    euclidean = spatial_distance_matrix(positions, WORLD)
    toroidal = spatial_distance_matrix(
        positions,
        WORLD,
        use_periodic_boundaries=True,
    )

    assert euclidean[0, 1] == pytest.approx(9.0)
    assert toroidal[0, 1] == pytest.approx(1.0)


def test_spatially_separated_clusters_score_high() -> None:
    """Genotype clusters in distinct regions give a positive silhouette."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [1.2, 0.9, 0.1],
        [0.9, 1.1, 0.1],
        [8.0, 8.0, 0.1],
        [8.2, 7.9, 0.1],
        [7.9, 8.1, 0.1],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])

    score = spatial_silhouette(positions, labels, WORLD)

    assert score is not None
    assert score > 0.8


def test_spatially_mixed_clusters_score_low() -> None:
    """Interleaved clusters must not look spatially structured."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [8.0, 8.0, 0.1],
        [1.2, 1.1, 0.1],
        [8.2, 8.1, 0.1],
        [1.1, 0.9, 0.1],
        [7.9, 7.9, 0.1],
    ])
    labels = np.array([0, 0, 1, 1, 0, 1])

    score = spatial_silhouette(positions, labels, WORLD)

    assert score is not None
    assert score < 0.3


def test_spatial_silhouette_needs_two_clusters() -> None:
    """One cluster has nothing to be separated from."""
    positions = np.array([[1.0, 1.0, 0.1], [2.0, 2.0, 0.1]])

    assert spatial_silhouette(positions, np.array([0, 0]), WORLD) is None


def test_a_seam_straddling_cluster_is_tight_on_a_torus() -> None:
    """Wrapping changes the centroid, the spread and the coherence.

    A cluster split across the world edge is genuinely tight; measuring it
    with Euclidean geometry puts its centroid where no member is and reports
    it as diffuse.
    """
    positions = np.array([
        [0.2, 5.0, 0.1],
        [9.8, 5.0, 0.1],
        [0.4, 5.2, 0.1],
        [5.0, 1.0, 0.1],
        [5.2, 1.2, 0.1],
        [4.8, 0.8, 0.1],
    ])
    labels = np.array([0, 0, 0, 1, 1, 1])

    flat = analyze_spatial_clustering(positions, labels, WORLD)
    torus = analyze_spatial_clustering(
        positions,
        labels,
        WORLD,
        use_periodic_boundaries=True,
    )

    # The wrapped centroid sits with its members, near x = 0.
    assert torus["cluster_centroids"][0][0] < 1.0
    assert flat["cluster_centroids"][0][0] > 3.0
    # And the cluster reads as tight rather than diffuse.
    assert torus["cluster_spreads"][0] < flat["cluster_spreads"][0]
    assert torus["spatial_silhouette"] > flat["spatial_silhouette"]


def test_spatial_analysis_reports_per_cluster_statistics() -> None:
    """Centroids, spreads and densities are given per cluster."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [1.5, 1.5, 0.1],
        [8.0, 8.0, 0.1],
        [8.5, 8.5, 0.1],
    ])
    labels = np.array([0, 0, 1, 1])

    analysis = analyze_spatial_clustering(positions, labels, WORLD)

    assert analysis["n_clusters"] == 2
    assert set(analysis["cluster_centroids"]) == {0, 1}
    assert set(analysis["cluster_spreads"]) == {0, 1}
    assert all(v > 0 for v in analysis["within_cluster_density"].values())
    assert analysis["spatial_segregation"] > 0
    assert 0.0 <= analysis["spatial_segregation_normalized"] <= 1.0


def test_noise_is_excluded_from_spatial_statistics() -> None:
    """Unassigned individuals should not form a phantom cluster."""
    positions = np.array([
        [1.0, 1.0, 0.1],
        [1.5, 1.5, 0.1],
        [8.0, 8.0, 0.1],
        [8.5, 8.5, 0.1],
        [5.0, 5.0, 0.1],
    ])
    labels = np.array([0, 0, 1, 1, -1])

    analysis = analyze_spatial_clustering(positions, labels, WORLD)

    assert analysis["n_clusters"] == 2
    assert -1 not in analysis["cluster_centroids"]


def test_a_single_member_cluster_has_no_density() -> None:
    """Density needs at least a pair to measure."""
    positions = np.array([[1.0, 1.0, 0.1], [8.0, 8.0, 0.1]])
    labels = np.array([0, 1])

    analysis = analyze_spatial_clustering(positions, labels, WORLD)

    assert analysis["within_cluster_density"][0] == 0.0
    assert analysis["cluster_spreads"][0] == 0.0
