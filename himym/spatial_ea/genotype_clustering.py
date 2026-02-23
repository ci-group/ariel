"""
Clustering analysis for spatial evolutionary algorithm genotypes.

This module provides tools for:
1. Clustering genotypes using various algorithms (DBSCAN, hierarchical, k-means)
2. Dimensionality reduction for visualization (PCA, t-SNE, UMAP)
3. Spatial-genotype correlation analysis
4. Cluster quality metrics
"""

import numpy as np
from typing import Any
from dataclasses import dataclass
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import warnings

# Try to import UMAP (optional dependency)
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not available. Install with: pip install umap-learn")


@dataclass
class ClusteringResult:
    """Results from clustering analysis."""
    cluster_labels: np.ndarray  # Cluster label for each individual
    n_clusters: int  # Number of clusters found
    cluster_sizes: dict[int, int]  # Number of individuals per cluster
    cluster_centers: np.ndarray | None  # Cluster centers (if applicable)
    algorithm: str  # Algorithm used
    
    # Quality metrics
    silhouette: float | None = None  # Silhouette score [-1, 1], higher is better
    calinski_harabasz: float | None = None  # CH index, higher is better
    davies_bouldin: float | None = None  # DB index, lower is better
    
    # Dimensionality reduction
    reduced_coords: np.ndarray | None = None  # 2D coordinates for visualization
    reduction_method: str | None = None  # Method used for reduction
    
    def __post_init__(self):
        """Compute cluster sizes from labels."""
        unique_labels = np.unique(self.cluster_labels)
        self.n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        self.cluster_sizes = {
            int(label): int(np.sum(self.cluster_labels == label))
            for label in unique_labels
        }


def cluster_dbscan(
    distance_matrix: np.ndarray,
    eps: float = 0.3,
    min_samples: int = 2
) -> ClusteringResult:
    """
    Cluster genotypes using DBSCAN (density-based clustering).
    
    DBSCAN is good for:
    - Finding clusters of arbitrary shape
    - Identifying noise/outliers
    - Not requiring pre-specified number of clusters
    
    Args:
        distance_matrix: Pairwise distance matrix (n x n)
        eps: Maximum distance for neighborhood (lower = tighter clusters)
        min_samples: Minimum samples in neighborhood to form cluster
        
    Returns:
        ClusteringResult with cluster labels and statistics
    """
    # DBSCAN needs distance matrix in condensed form
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clusterer.fit_predict(distance_matrix)
    
    result = ClusteringResult(
        cluster_labels=labels,
        n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
        cluster_sizes={},
        cluster_centers=None,
        algorithm='DBSCAN'
    )
    
    # Compute quality metrics (excluding noise points)
    if result.n_clusters > 1:
        non_noise = labels >= 0
        if np.sum(non_noise) > result.n_clusters:
            try:
                result.silhouette = silhouette_score(
                    distance_matrix[np.ix_(non_noise, non_noise)],
                    labels[non_noise],
                    metric='precomputed'
                )
            except:
                result.silhouette = None
    
    return result


def cluster_hierarchical(
    distance_matrix: np.ndarray,
    n_clusters: int = 3,
    linkage: str = 'average'
) -> ClusteringResult:
    """
    Cluster genotypes using hierarchical/agglomerative clustering.
    
    Hierarchical clustering is good for:
    - Understanding nested cluster structure
    - Visualizing dendrograms
    - Not sensitive to initialization
    
    Args:
        distance_matrix: Pairwise distance matrix (n x n)
        n_clusters: Number of clusters to form
        linkage: Linkage criterion ('ward', 'complete', 'average', 'single')
        
    Returns:
        ClusteringResult with cluster labels and statistics
    """
    # Use distance matrix directly
    if linkage == 'ward':
        # Ward requires affinity='euclidean', convert from precomputed
        # This is approximate - better to use with feature vectors
        warnings.warn("Ward linkage with distance matrix may not be optimal. "
                     "Consider using feature vectors instead.")
        linkage = 'average'
    
    clusterer = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage=linkage
    )
    labels = clusterer.fit_predict(distance_matrix)
    
    result = ClusteringResult(
        cluster_labels=labels,
        n_clusters=n_clusters,
        cluster_sizes={},
        cluster_centers=None,
        algorithm=f'Hierarchical-{linkage}'
    )
    
    # Compute quality metrics
    if n_clusters > 1 and n_clusters < len(labels):
        try:
            result.silhouette = silhouette_score(
                distance_matrix, labels, metric='precomputed'
            )
        except:
            result.silhouette = None
    
    return result


def cluster_kmeans(
    feature_vectors: np.ndarray,
    n_clusters: int = 3,
    n_init: int = 10,
    random_state: int | None = None
) -> ClusteringResult:
    """
    Cluster genotypes using k-means.
    
    K-means requires feature vectors (not distance matrix).
    Good for:
    - Fast clustering of large datasets
    - Well-separated, spherical clusters
    - When number of clusters is known
    
    Args:
        feature_vectors: Feature matrix (n x d)
        n_clusters: Number of clusters to form
        n_init: Number of initializations
        random_state: Random seed for reproducibility
        
    Returns:
        ClusteringResult with cluster labels and statistics
    """
    clusterer = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state
    )
    labels = clusterer.fit_predict(feature_vectors)
    
    result = ClusteringResult(
        cluster_labels=labels,
        n_clusters=n_clusters,
        cluster_sizes={},
        cluster_centers=clusterer.cluster_centers_,
        algorithm='K-Means'
    )
    
    # Compute quality metrics
    if n_clusters > 1 and n_clusters < len(labels):
        try:
            result.silhouette = silhouette_score(feature_vectors, labels)
            result.calinski_harabasz = calinski_harabasz_score(feature_vectors, labels)
            result.davies_bouldin = davies_bouldin_score(feature_vectors, labels)
        except:
            pass
    
    return result


def reduce_dimensions_pca(
    distance_matrix: np.ndarray | None = None,
    feature_vectors: np.ndarray | None = None,
    n_components: int = 2
) -> np.ndarray:
    """
    Reduce dimensionality using PCA.
    
    PCA is good for:
    - Linear dimensionality reduction
    - Fast computation
    - Preserving global structure
    
    Args:
        distance_matrix: Pairwise distance matrix (n x n) OR
        feature_vectors: Feature matrix (n x d)
        n_components: Number of dimensions (usually 2 for visualization)
        
    Returns:
        Reduced coordinates (n x n_components)
    """
    if feature_vectors is not None:
        data = feature_vectors
    elif distance_matrix is not None:
        # Convert distance matrix to feature space using MDS-like approach
        # This is approximate - better to use feature vectors if available
        n = distance_matrix.shape[0]
        # Center the squared distance matrix
        D_sq = distance_matrix ** 2
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ D_sq @ H
        # Get eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        # Take top components
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
        # Construct embedding
        data = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    else:
        raise ValueError("Either distance_matrix or feature_vectors must be provided")
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(data)
    
    return reduced


def reduce_dimensions_tsne(
    distance_matrix: np.ndarray | None = None,
    feature_vectors: np.ndarray | None = None,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int | None = None
) -> np.ndarray:
    """
    Reduce dimensionality using t-SNE.
    
    t-SNE is good for:
    - Non-linear dimensionality reduction
    - Preserving local structure
    - Visualizing clusters
    
    Args:
        distance_matrix: Pairwise distance matrix (n x n) OR
        feature_vectors: Feature matrix (n x d)
        n_components: Number of dimensions (usually 2)
        perplexity: Perplexity parameter (5-50 typical)
        random_state: Random seed
        
    Returns:
        Reduced coordinates (n x n_components)
    """
    # Adjust perplexity if needed
    n_samples = (distance_matrix.shape[0] if distance_matrix is not None 
                 else feature_vectors.shape[0])
    perplexity = min(perplexity, (n_samples - 1) / 3.0)
    
    if feature_vectors is not None:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            init='pca'
        )
        reduced = tsne.fit_transform(feature_vectors)
    elif distance_matrix is not None:
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            metric='precomputed'
        )
        reduced = tsne.fit_transform(distance_matrix)
    else:
        raise ValueError("Either distance_matrix or feature_vectors must be provided")
    
    return reduced


def reduce_dimensions_umap(
    distance_matrix: np.ndarray | None = None,
    feature_vectors: np.ndarray | None = None,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int | None = None
) -> np.ndarray:
    """
    Reduce dimensionality using UMAP.
    
    UMAP is good for:
    - Non-linear dimensionality reduction
    - Preserving both local and global structure
    - Faster than t-SNE for large datasets
    
    Args:
        distance_matrix: Pairwise distance matrix (n x n) OR
        feature_vectors: Feature matrix (n x d)
        n_components: Number of dimensions (usually 2)
        n_neighbors: Number of neighbors (5-50 typical)
        min_dist: Minimum distance in embedding
        random_state: Random seed
        
    Returns:
        Reduced coordinates (n x n_components)
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not installed. Install with: pip install umap-learn")
    
    if feature_vectors is not None:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        reduced = reducer.fit_transform(feature_vectors)
    elif distance_matrix is not None:
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='precomputed'
        )
        reduced = reducer.fit_transform(distance_matrix)
    else:
        raise ValueError("Either distance_matrix or feature_vectors must be provided")
    
    return reduced


def analyze_spatial_clustering(
    positions: np.ndarray,
    cluster_labels: np.ndarray,
    world_size: tuple[float, float] = (50.0, 50.0)
) -> dict:
    """
    Analyze how genotype clusters are spatially distributed.
    
    Args:
        positions: Spatial positions (n x 2) of individuals
        cluster_labels: Cluster labels for each individual
        world_size: Size of the world (width, height)
        
    Returns:
        Dictionary with spatial clustering metrics:
        - cluster_centroids: Mean position for each cluster
        - cluster_spreads: Standard deviation of positions per cluster
        - spatial_segregation: Degree of spatial separation between clusters
        - within_cluster_density: How tightly packed each cluster is spatially
    """
    unique_labels = np.unique(cluster_labels[cluster_labels >= 0])
    
    cluster_centroids = {}
    cluster_spreads = {}
    within_cluster_density = {}
    
    for label in unique_labels:
        mask = cluster_labels == label
        cluster_pos = positions[mask]
        
        if len(cluster_pos) > 0:
            # Centroid
            centroid = np.mean(cluster_pos, axis=0)
            cluster_centroids[int(label)] = centroid
            
            # Spread (std of distances from centroid)
            distances = np.linalg.norm(cluster_pos - centroid, axis=1)
            cluster_spreads[int(label)] = float(np.std(distances))
            
            # Density (inverse of average pairwise distance)
            if len(cluster_pos) > 1:
                pairwise_dists = []
                for i in range(len(cluster_pos)):
                    for j in range(i + 1, len(cluster_pos)):
                        pairwise_dists.append(np.linalg.norm(cluster_pos[i] - cluster_pos[j]))
                avg_pairwise = np.mean(pairwise_dists)
                within_cluster_density[int(label)] = 1.0 / (avg_pairwise + 1e-6)
            else:
                within_cluster_density[int(label)] = 0.0
    
    # Spatial segregation: average distance between cluster centroids
    if len(cluster_centroids) > 1:
        centroid_dists = []
        centroids_list = list(cluster_centroids.values())
        for i in range(len(centroids_list)):
            for j in range(i + 1, len(centroids_list)):
                centroid_dists.append(np.linalg.norm(centroids_list[i] - centroids_list[j]))
        spatial_segregation = float(np.mean(centroid_dists))
        
        # Normalize by world diagonal
        world_diagonal = np.sqrt(world_size[0]**2 + world_size[1]**2)
        spatial_segregation_normalized = spatial_segregation / world_diagonal
    else:
        spatial_segregation = 0.0
        spatial_segregation_normalized = 0.0
    
    return {
        'cluster_centroids': cluster_centroids,
        'cluster_spreads': cluster_spreads,
        'within_cluster_density': within_cluster_density,
        'spatial_segregation': spatial_segregation,
        'spatial_segregation_normalized': spatial_segregation_normalized,
        'n_clusters': len(cluster_centroids)
    }


def find_optimal_clusters(
    distance_matrix: np.ndarray,
    max_clusters: int = 10,
    method: str = 'hierarchical'
) -> tuple[int, dict]:
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        distance_matrix: Pairwise distance matrix
        max_clusters: Maximum number of clusters to try
        method: Clustering method ('hierarchical' or 'kmeans')
        
    Returns:
        Tuple of (optimal_k, scores_dict)
    """
    n_samples = distance_matrix.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    
    scores = {}
    
    for k in range(2, max_clusters + 1):
        if method == 'hierarchical':
            result = cluster_hierarchical(distance_matrix, n_clusters=k)
            if result.silhouette is not None:
                scores[k] = result.silhouette
        elif method == 'kmeans':
            # K-means needs feature vectors - use MDS embedding
            coords = reduce_dimensions_pca(distance_matrix=distance_matrix, n_components=min(10, n_samples))
            result = cluster_kmeans(coords, n_clusters=k)
            if result.silhouette is not None:
                scores[k] = result.silhouette
    
    if scores:
        optimal_k = max(scores, key=scores.get)
        return optimal_k, scores
    else:
        return 2, {}
