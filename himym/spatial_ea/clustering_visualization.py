"""
Visualization tools for genotype clustering analysis.

This module provides comprehensive visualizations for:
1. Spatial distribution of genotype clusters
2. Cluster structure in reduced dimensions
3. Dendrograms for hierarchical clustering
4. Cluster quality metrics
5. Temporal evolution of clustering
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import numpy as np
from pathlib import Path
from typing import Any
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage


def plot_spatial_clusters(
    positions: np.ndarray,
    cluster_labels: np.ndarray,
    fitness_values: np.ndarray | None = None,
    world_size: tuple[float, float] = (50.0, 50.0),
    cluster_centroids: dict | None = None,
    title: str = "Spatial Distribution of Genotype Clusters",
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 10)
) -> None:
    """
    Plot spatial positions colored by cluster membership.
    
    Args:
        positions: Spatial positions (n x 2)
        cluster_labels: Cluster label for each individual
        fitness_values: Optional fitness values for sizing points
        world_size: Size of the world
        cluster_centroids: Optional cluster centroids to plot
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique clusters (excluding noise -1)
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    # Create colormap
    if n_clusters > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    
    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        if label == -1:
            # Noise points (DBSCAN)
            color = 'gray'
            marker = 'x'
            label_name = 'Noise'
        else:
            color = colors[label % len(colors)]
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = cluster_labels == label
        cluster_pos = positions[mask]
        
        # Size by fitness if available
        if fitness_values is not None:
            sizes = 50 + 200 * (fitness_values[mask] / np.max(fitness_values))
        else:
            sizes = 100
        
        ax.scatter(
            cluster_pos[:, 0], cluster_pos[:, 1],
            c=[color], marker=marker, s=sizes,
            label=label_name, alpha=0.7, edgecolors='black', linewidths=0.5
        )
    
    # Plot cluster centroids if provided
    if cluster_centroids:
        for label, centroid in cluster_centroids.items():
            ax.scatter(
                centroid[0], centroid[1],
                c='red', marker='*', s=500,
                edgecolors='black', linewidths=2,
                zorder=10
            )
            ax.annotate(
                f'C{label}',
                xy=centroid,
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12,
                fontweight='bold',
                color='red'
            )
    
    ax.set_xlim(0, world_size[0])
    ax.set_ylim(0, world_size[1])
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Legend
    if n_clusters <= 10:
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spatial cluster plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cluster_embedding(
    reduced_coords: np.ndarray,
    cluster_labels: np.ndarray,
    fitness_values: np.ndarray | None = None,
    reduction_method: str = "PCA",
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 8)
) -> None:
    """
    Plot clusters in 2D reduced coordinate space.
    
    Args:
        reduced_coords: 2D coordinates from dimensionality reduction
        cluster_labels: Cluster labels
        fitness_values: Optional fitness for coloring
        reduction_method: Name of reduction method (for title)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels[unique_labels >= 0])
    
    # Colormap
    if n_clusters > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    else:
        colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
    
    # Plot each cluster
    for idx, label in enumerate(unique_labels):
        if label == -1:
            color = 'gray'
            marker = 'x'
            label_name = 'Noise'
        else:
            color = colors[label % len(colors)]
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = cluster_labels == label
        cluster_coords = reduced_coords[mask]
        
        # Size by fitness if available
        if fitness_values is not None:
            sizes = 50 + 200 * (fitness_values[mask] / np.max(fitness_values))
        else:
            sizes = 100
        
        ax.scatter(
            cluster_coords[:, 0], cluster_coords[:, 1],
            c=[color], marker=marker, s=sizes,
            label=label_name, alpha=0.7, edgecolors='black', linewidths=0.5
        )
    
    if title is None:
        title = f"Genotype Clusters in {reduction_method} Space"
    
    ax.set_xlabel(f'{reduction_method} Component 1', fontsize=12)
    ax.set_ylabel(f'{reduction_method} Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if n_clusters <= 10:
        ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster embedding plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_dendrogram(
    distance_matrix: np.ndarray,
    individual_ids: list[int] | None = None,
    linkage_method: str = 'average',
    title: str = "Hierarchical Clustering Dendrogram",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 8)
) -> None:
    """
    Plot hierarchical clustering dendrogram.
    
    Args:
        distance_matrix: Pairwise distance matrix
        individual_ids: Labels for individuals
        linkage_method: Linkage method for dendrogram
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    from scipy.spatial.distance import squareform
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convert distance matrix to condensed form
    condensed_dist = squareform(distance_matrix, checks=False)
    
    # Perform linkage
    Z = scipy_linkage(condensed_dist, method=linkage_method)
    
    # Plot dendrogram
    if individual_ids is not None:
        labels = [str(id_) for id_ in individual_ids]
    else:
        labels = None
    
    dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=0.7 * max(Z[:, 2])
    )
    
    ax.set_xlabel('Individual ID', fontsize=12)
    ax.set_ylabel('Distance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dendrogram to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_cluster_quality_metrics(
    silhouette_scores: dict[int, float],
    title: str = "Cluster Quality by Number of Clusters",
    save_path: str | None = None,
    figsize: tuple[int, int] = (10, 6)
) -> None:
    """
    Plot cluster quality metrics vs number of clusters.
    
    Args:
        silhouette_scores: Dict mapping n_clusters -> silhouette score
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    k_values = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in k_values]
    
    ax.plot(k_values, scores, marker='o', linewidth=2, markersize=8, color='blue')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Random clustering')
    
    # Highlight best
    best_k = max(silhouette_scores, key=silhouette_scores.get)
    best_score = silhouette_scores[best_k]
    ax.scatter([best_k], [best_score], s=300, c='red', marker='*', 
              edgecolors='black', linewidths=2, zorder=10,
              label=f'Best: k={best_k}')
    
    ax.set_xlabel('Number of Clusters', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster quality plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    individual_ids: list[int] | None = None,
    cluster_labels: np.ndarray | None = None,
    title: str = "Genotype Distance Heatmap",
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 10)
) -> None:
    """
    Plot heatmap of pairwise genotype distances.
    
    Args:
        distance_matrix: Pairwise distance matrix
        individual_ids: Labels for individuals
        cluster_labels: Optional cluster labels to order by
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Reorder by cluster if provided
    if cluster_labels is not None:
        order = np.argsort(cluster_labels)
        distance_matrix = distance_matrix[order][:, order]
        if individual_ids is not None:
            individual_ids = [individual_ids[i] for i in order]
    
    # Plot heatmap
    im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Distance', fontsize=12)
    
    # Labels
    if individual_ids is not None and len(individual_ids) <= 50:
        ax.set_xticks(range(len(individual_ids)))
        ax.set_yticks(range(len(individual_ids)))
        ax.set_xticklabels(individual_ids, rotation=90, fontsize=8)
        ax.set_yticklabels(individual_ids, fontsize=8)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Individual', fontsize=12)
    ax.set_ylabel('Individual', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved distance heatmap to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_temporal_clustering(
    clustering_results_by_generation: dict[int, Any],
    metric: str = 'n_clusters',
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (12, 6)
) -> None:
    """
    Plot how clustering evolves over generations.
    
    Args:
        clustering_results_by_generation: Dict mapping generation -> ClusteringResult
        metric: Metric to plot ('n_clusters', 'silhouette', 'diversity')
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    generations = sorted(clustering_results_by_generation.keys())
    values = []
    
    for gen in generations:
        result = clustering_results_by_generation[gen]
        if metric == 'n_clusters':
            values.append(result.n_clusters)
        elif metric == 'silhouette':
            values.append(result.silhouette if result.silhouette is not None else 0)
        elif metric == 'diversity':
            # Would need diversity metric from distance matrix
            values.append(0)  # Placeholder
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    ax.plot(generations, values, marker='o', linewidth=2, markersize=6, color='blue')
    ax.fill_between(generations, values, alpha=0.2, color='blue')
    
    if title is None:
        title = f"{metric.replace('_', ' ').title()} Over Generations"
    
    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved temporal clustering plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_clustering_report(
    positions: np.ndarray,
    cluster_labels: np.ndarray,
    reduced_coords: np.ndarray,
    distance_matrix: np.ndarray,
    fitness_values: np.ndarray | None = None,
    individual_ids: list[int] | None = None,
    cluster_centroids: dict | None = None,
    spatial_analysis: dict | None = None,
    output_dir: str | Path = "clustering_analysis",
    generation: int | None = None
) -> None:
    """
    Create a comprehensive clustering report with multiple visualizations.
    
    Args:
        positions: Spatial positions (n x 2)
        cluster_labels: Cluster labels for each individual
        reduced_coords: 2D coordinates from dimensionality reduction
        distance_matrix: Pairwise distance matrix
        fitness_values: Optional fitness values
        individual_ids: Individual IDs
        cluster_centroids: Cluster centroids
        spatial_analysis: Spatial clustering analysis results
        output_dir: Directory to save plots
        generation: Generation number (for filenames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gen_suffix = f"_gen{generation:03d}" if generation is not None else ""
    
    # 1. Spatial cluster distribution
    plot_spatial_clusters(
        positions=positions,
        cluster_labels=cluster_labels,
        fitness_values=fitness_values,
        cluster_centroids=cluster_centroids,
        save_path=output_dir / f"spatial_clusters{gen_suffix}.png"
    )
    
    # 2. Cluster embedding
    plot_cluster_embedding(
        reduced_coords=reduced_coords,
        cluster_labels=cluster_labels,
        fitness_values=fitness_values,
        reduction_method="Dimension Reduction",
        save_path=output_dir / f"cluster_embedding{gen_suffix}.png"
    )
    
    # 3. Distance heatmap
    plot_distance_heatmap(
        distance_matrix=distance_matrix,
        individual_ids=individual_ids,
        cluster_labels=cluster_labels,
        save_path=output_dir / f"distance_heatmap{gen_suffix}.png"
    )
    
    # 4. Dendrogram (if not too many individuals)
    if len(positions) <= 100:
        plot_dendrogram(
            distance_matrix=distance_matrix,
            individual_ids=individual_ids,
            save_path=output_dir / f"dendrogram{gen_suffix}.png"
        )
    
    print(f"\nClustering report saved to {output_dir}/")
