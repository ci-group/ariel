"""Do genotype clusters correspond to places in the world?

This is the question the spatial EA exists to ask. Mating by proximity should,
if it does anything, let genetically similar individuals accumulate in the same
region — emergent speciation, with no explicit species mechanism anywhere in
the algorithm.

The work splits in two. Clustering groups individuals by genome distance,
knowing nothing about where they are. :func:`analyze_spatial_clustering` then
asks whether those groups are spatially coherent.

Notes
-----
    * Distances between genomes are precomputed (see
      :mod:`ariel.spatial_ea.genotype_distance`), so every clustering algorithm
      here is given a distance matrix rather than feature vectors.
    * Spatial statistics respect periodic boundaries. On a torus a plain mean
      is not a centroid and a plain norm is not a distance, so with wrapping
      enabled the centroid is a circular mean and separations use the toroidal
      metric. The research prototype used Euclidean geometry throughout, which
      overstates how far apart clusters are whenever one straddles a seam.

"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

# Third-party libraries
import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Local libraries
from ariel import log
from ariel.spatial_ea.interaction import calculate_periodic_distance

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from ariel.parameters.ariel_types import FloatArray

# Type Aliases
type ReductionMethod = Literal["pca", "tsne"]
type LinkageMethod = Literal["average", "complete", "single", "ward"]

# Global constants
NOISE_LABEL = -1
MIN_SAMPLES_FOR_QUALITY = 2
DENSITY_EPSILON = 1e-6


@dataclass
class ClusteringResult:
    """The outcome of clustering one population.

    Parameters
    ----------
    cluster_labels
        Cluster index per individual. ``-1`` marks noise, which only DBSCAN
        produces.
    algorithm
        Which algorithm produced the labels.
    cluster_centers
        Cluster centres, when the algorithm computes them.
    silhouette
        Cohesion versus separation in ``[-1, 1]``; higher is better.
    calinski_harabasz
        Variance ratio; higher is better.
    davies_bouldin
        Average cluster similarity; lower is better.
    reduced_coords
        Two-dimensional embedding for plotting.
    reduction_method
        How that embedding was produced.
    """

    cluster_labels: FloatArray
    algorithm: str
    cluster_centers: FloatArray | None = None
    silhouette: float | None = None
    calinski_harabasz: float | None = None
    davies_bouldin: float | None = None
    reduced_coords: FloatArray | None = None
    reduction_method: str | None = None
    n_clusters: int = field(init=False, default=0)
    cluster_sizes: dict[int, int] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Derive the cluster count and sizes from the labels."""
        labels = np.asarray(self.cluster_labels)
        unique = np.unique(labels)
        # Noise is not a cluster.
        self.n_clusters = len(unique[unique >= 0])
        self.cluster_sizes = {
            int(label): int(np.sum(labels == label)) for label in unique
        }

    @property
    def num_noise(self) -> int:
        """How many individuals were left unassigned.

        Returns
        -------
            Size of the noise group, zero when there is none.
        """
        return self.cluster_sizes.get(NOISE_LABEL, 0)


def _quality_scores(
    distance_matrix: FloatArray,
    labels: FloatArray,
) -> tuple[float | None, float | None, float | None]:
    """Score a clustering, when it is meaningful to do so.

    Quality indices are undefined for a single cluster, and noise points would
    distort them, so both are excluded.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    labels
        Cluster label per individual.

    Returns
    -------
    silhouette
        Silhouette score, or ``None``.
    calinski_harabasz
        Calinski-Harabasz index, or ``None``.
    davies_bouldin
        Davies-Bouldin index, or ``None``.
    """
    assigned = labels >= 0
    kept_labels = labels[assigned]
    if len(np.unique(kept_labels)) < MIN_SAMPLES_FOR_QUALITY:
        return None, None, None

    kept = distance_matrix[np.ix_(assigned, assigned)]

    try:
        silhouette = float(
            silhouette_score(kept, kept_labels, metric="precomputed"),
        )
    except ValueError:
        silhouette = None

    # These two want coordinates, not distances; the distance rows are a
    # serviceable embedding for a relative comparison.
    try:
        calinski = float(calinski_harabasz_score(kept, kept_labels))
        davies = float(davies_bouldin_score(kept, kept_labels))
    except ValueError:
        calinski, davies = None, None

    return silhouette, calinski, davies


def cluster_dbscan(
    distance_matrix: FloatArray,
    eps: float = 0.3,
    min_samples: int = 2,
) -> ClusteringResult:
    """Group individuals by density, leaving outliers unassigned.

    The only algorithm here that decides the number of clusters itself, and the
    only one that can label an individual as noise.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    eps
        Neighbourhood radius in genome distance.
    min_samples
        Neighbours required to form a dense region.

    Returns
    -------
        The clustering, scored.
    """
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="precomputed",
    ).fit_predict(distance_matrix)

    silhouette, calinski, davies = _quality_scores(distance_matrix, labels)
    return ClusteringResult(
        cluster_labels=labels,
        algorithm="dbscan",
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
    )


def cluster_hierarchical(
    distance_matrix: FloatArray,
    n_clusters: int = 3,
    linkage: LinkageMethod = "average",
) -> ClusteringResult:
    """Group individuals by successively merging the closest pairs.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    n_clusters
        How many clusters to cut the tree into.
    linkage
        Merge criterion. ``ward`` is unavailable for precomputed distances.

    Returns
    -------
        The clustering, scored.

    Raises
    ------
    ValueError
        If ``ward`` linkage is requested.
    """
    if linkage == "ward":
        msg = "Ward linkage needs coordinates, not a distance matrix"
        raise ValueError(msg)

    labels = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(distance_matrix)),
        metric="precomputed",
        linkage=linkage,
    ).fit_predict(distance_matrix)

    silhouette, calinski, davies = _quality_scores(distance_matrix, labels)
    return ClusteringResult(
        cluster_labels=labels,
        algorithm=f"hierarchical_{linkage}",
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
    )


def cluster_kmeans(
    feature_vectors: FloatArray,
    n_clusters: int = 3,
    random_state: int = 42,
) -> ClusteringResult:
    """Group individuals around cluster centres.

    Unlike the other two, this one needs coordinates rather than distances —
    pass an embedding from :func:`reduce_dimensions_pca`.

    Parameters
    ----------
    feature_vectors
        Coordinates per individual.
    n_clusters
        How many clusters to find.
    random_state
        Seed for the centroid initialisation.

    Returns
    -------
        The clustering, scored against Euclidean distances in the embedding.
    """
    features = np.asarray(feature_vectors, dtype=float)
    model = KMeans(
        n_clusters=min(n_clusters, len(features)),
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(features)

    distances = np.linalg.norm(
        features[:, None, :] - features[None, :, :],
        axis=-1,
    )
    silhouette, calinski, davies = _quality_scores(distances, labels)

    return ClusteringResult(
        cluster_labels=labels,
        algorithm="kmeans",
        cluster_centers=model.cluster_centers_,
        silhouette=silhouette,
        calinski_harabasz=calinski,
        davies_bouldin=davies,
    )


def reduce_dimensions_pca(
    distance_matrix: FloatArray,
    n_components: int = 2,
) -> FloatArray:
    """Embed a distance matrix in a few dimensions, linearly.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    n_components
        Target dimensionality.

    Returns
    -------
        One coordinate row per individual.
    """
    components = min(n_components, *np.shape(distance_matrix))
    return np.asarray(
        PCA(n_components=components).fit_transform(distance_matrix),
        dtype=float,
    )


def reduce_dimensions_tsne(
    distance_matrix: FloatArray,
    n_components: int = 2,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> FloatArray:
    """Embed a distance matrix in a few dimensions, preserving neighbourhoods.

    Better than PCA for seeing clusters, worse for reading distances off the
    plot: t-SNE preserves who is near whom, not how far apart things are.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    n_components
        Target dimensionality.
    perplexity
        Neighbourhood size. Clamped below the population size, which t-SNE
        requires.
    random_state
        Seed for the embedding.

    Returns
    -------
        One coordinate row per individual.
    """
    count = len(distance_matrix)
    safe_perplexity = max(1.0, min(perplexity, count - 1))

    return np.asarray(
        TSNE(
            n_components=n_components,
            perplexity=safe_perplexity,
            metric="precomputed",
            init="random",
            random_state=random_state,
        ).fit_transform(distance_matrix),
        dtype=float,
    )


def _circular_mean(values: FloatArray, span: float) -> float:
    """Average positions on a wrapped axis.

    The mean of ``0.1`` and ``9.9`` on a ten-metre torus is ``0``, not ``5``.

    Parameters
    ----------
    values
        Positions along one axis.
    span
        Length of the axis before it wraps.

    Returns
    -------
        The circular mean, in ``[0, span)``.
    """
    angles = 2.0 * np.pi * np.asarray(values, dtype=float) / span
    mean_angle = np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
    return float((mean_angle % (2.0 * np.pi)) * span / (2.0 * np.pi))


def spatial_distance_matrix(
    positions: FloatArray,
    world_size: tuple[float, float],
    *,
    use_periodic_boundaries: bool = False,
) -> FloatArray:
    """Build the matrix of distances between individuals in the world.

    Parameters
    ----------
    positions
        One position row per individual.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether distances wrap around the world edges.

    Returns
    -------
        A symmetric matrix of spatial distances.
    """
    points = np.asarray(positions, dtype=float)
    count = len(points)
    matrix = np.zeros((count, count))

    for i in range(count):
        for j in range(i + 1, count):
            if use_periodic_boundaries:
                distance = calculate_periodic_distance(
                    points[i],
                    points[j],
                    world_size,
                )
            else:
                distance = float(
                    np.linalg.norm(points[i][:2] - points[j][:2]),
                )
            matrix[i, j] = distance
            matrix[j, i] = distance

    return matrix


def analyze_spatial_clustering(
    positions: FloatArray,
    cluster_labels: FloatArray,
    world_size: tuple[float, float] = (25.0, 25.0),
    *,
    use_periodic_boundaries: bool = False,
) -> dict[str, Any]:
    """Ask whether genotype clusters occupy distinct places.

    The headline number is ``spatial_silhouette``: the silhouette of the
    *genotype* clusters measured in *space*. Above zero means individuals are
    spatially closer to their own genetic cluster than to any other — which is
    the emergent-speciation claim. Around zero means the clusters are spatially
    mixed, whatever the genome distances say.

    Parameters
    ----------
    positions
        One position row per individual.
    cluster_labels
        Genotype cluster label per individual.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether the world wraps. This changes both the centroids and the
        separations.

    Returns
    -------
        Per-cluster centroids, spreads and densities, plus
        ``spatial_segregation`` (mean centroid separation),
        ``spatial_segregation_normalized`` (as a fraction of the world
        diagonal) and ``spatial_silhouette``.
    """
    points = np.asarray(positions, dtype=float)
    labels = np.asarray(cluster_labels)
    unique = np.unique(labels[labels >= 0])

    centroids: dict[int, FloatArray] = {}
    spreads: dict[int, float] = {}
    densities: dict[int, float] = {}

    for label in unique:
        member_positions = points[labels == label]
        if not len(member_positions):
            continue

        if use_periodic_boundaries:
            centroid = np.array([
                _circular_mean(member_positions[:, 0], world_size[0]),
                _circular_mean(member_positions[:, 1], world_size[1]),
            ])
            offsets = np.array([
                calculate_periodic_distance(position, centroid, world_size)
                for position in member_positions
            ])
        else:
            centroid = member_positions[:, :2].mean(axis=0)
            offsets = np.linalg.norm(
                member_positions[:, :2] - centroid,
                axis=1,
            )

        centroids[int(label)] = centroid
        spreads[int(label)] = float(np.std(offsets))

        if len(member_positions) > 1:
            within = spatial_distance_matrix(
                member_positions,
                world_size,
                use_periodic_boundaries=use_periodic_boundaries,
            )
            mean_separation = float(
                np.mean(within[np.triu_indices_from(within, k=1)]),
            )
            densities[int(label)] = 1.0 / (mean_separation + DENSITY_EPSILON)
        else:
            densities[int(label)] = 0.0

    segregation = 0.0
    if len(centroids) > 1:
        centroid_points = np.array(list(centroids.values()))
        separations = spatial_distance_matrix(
            centroid_points,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
        )
        segregation = float(
            np.mean(separations[np.triu_indices_from(separations, k=1)]),
        )

    diagonal = float(np.hypot(world_size[0], world_size[1]))

    return {
        "cluster_centroids": centroids,
        "cluster_spreads": spreads,
        "within_cluster_density": densities,
        "spatial_segregation": segregation,
        "spatial_segregation_normalized": (
            segregation / diagonal if diagonal > 0 else 0.0
        ),
        "spatial_silhouette": spatial_silhouette(
            points,
            labels,
            world_size,
            use_periodic_boundaries=use_periodic_boundaries,
        ),
        "n_clusters": len(centroids),
    }


def spatial_silhouette(
    positions: FloatArray,
    cluster_labels: FloatArray,
    world_size: tuple[float, float],
    *,
    use_periodic_boundaries: bool = False,
) -> float | None:
    """Score how spatially coherent the genotype clusters are.

    Parameters
    ----------
    positions
        One position row per individual.
    cluster_labels
        Genotype cluster label per individual.
    world_size
        World dimensions ``(width, height)``.
    use_periodic_boundaries
        Whether distances wrap around the world edges.

    Returns
    -------
        A value in ``[-1, 1]``, or ``None`` when there are fewer than two
        clusters to compare.
    """
    labels = np.asarray(cluster_labels)
    assigned = labels >= 0
    kept_labels = labels[assigned]
    if len(np.unique(kept_labels)) < MIN_SAMPLES_FOR_QUALITY:
        return None

    distances = spatial_distance_matrix(
        np.asarray(positions, dtype=float)[assigned],
        world_size,
        use_periodic_boundaries=use_periodic_boundaries,
    )
    try:
        return float(
            silhouette_score(distances, kept_labels, metric="precomputed"),
        )
    except ValueError:
        return None


def find_optimal_clusters(
    distance_matrix: FloatArray,
    max_clusters: int = 10,
    linkage: LinkageMethod = "average",
) -> tuple[int, dict[int, float]]:
    """Choose a cluster count by silhouette score.

    Parameters
    ----------
    distance_matrix
        Pairwise genome distances.
    max_clusters
        Largest count to try.
    linkage
        Merge criterion for the trial clusterings.

    Returns
    -------
    best
        The count with the highest silhouette, or ``1`` if none scored.
    scores
        Silhouette per count tried.
    """
    count = len(distance_matrix)
    scores: dict[int, float] = {}

    for k in range(2, min(max_clusters, count - 1) + 1):
        result = cluster_hierarchical(distance_matrix, k, linkage)
        if result.silhouette is not None:
            scores[k] = result.silhouette

    if not scores:
        return 1, scores

    best = max(scores, key=lambda k: scores[k])
    msg = f"Best silhouette {scores[best]:.3f} at {best} clusters"
    log.info(msg)
    return best, scores
