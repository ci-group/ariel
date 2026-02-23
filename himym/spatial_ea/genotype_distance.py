"""
Genotype distance metrics for clustering analysis.

This module provides various distance metrics for comparing HyperNEAT genotypes:
1. Structural distance: Based on network topology (nodes, connections)
2. Weight distance: Based on connection weights
3. Combined distance: Weighted combination of structural and weight distances
4. Behavioral distance: Based on phenotypic behavior (if available)
"""

import numpy as np
from typing import Any
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr


def compute_structural_distance(genotype1: dict, genotype2: dict) -> float:
    """
    Compute structural distance between two HyperNEAT genotypes.
    
    This measures how different the network topologies are based on:
    - Number of nodes
    - Number of connections
    - Innovation numbers (connection IDs)
    - Node types and layers
    
    Args:
        genotype1: First HyperNEAT genotype dictionary
        genotype2: Second HyperNEAT genotype dictionary
        
    Returns:
        Normalized structural distance [0, 1], where 0 = identical structure
    """
    if not isinstance(genotype1, dict) or not isinstance(genotype2, dict):
        return 1.0  # Maximum distance for incompatible types
    
    # Extract nodes and connections
    nodes1 = genotype1.get('nodes', [])
    nodes2 = genotype2.get('nodes', [])
    conns1 = genotype1.get('connections', [])
    conns2 = genotype2.get('connections', [])
    
    # Node difference (normalized by max nodes)
    num_nodes1 = len(nodes1)
    num_nodes2 = len(nodes2)
    max_nodes = max(num_nodes1, num_nodes2, 1)
    node_diff = abs(num_nodes1 - num_nodes2) / max_nodes
    
    # Connection difference (normalized by max connections)
    num_conns1 = len(conns1)
    num_conns2 = len(conns2)
    max_conns = max(num_conns1, num_conns2, 1)
    conn_diff = abs(num_conns1 - num_conns2) / max_conns
    
    # Innovation set difference (Jaccard distance)
    innov1 = set(conn.innovation for conn in conns1)
    innov2 = set(conn.innovation for conn in conns2)
    
    if len(innov1) == 0 and len(innov2) == 0:
        jaccard_dist = 0.0
    else:
        intersection = len(innov1 & innov2)
        union = len(innov1 | innov2)
        jaccard_dist = 1.0 - (intersection / union if union > 0 else 0.0)
    
    # Node type difference (if nodes have same IDs but different types)
    node_dict1 = {node.id: (node.type, node.layer) for node in nodes1}
    node_dict2 = {node.id: (node.type, node.layer) for node in nodes2}
    common_nodes = set(node_dict1.keys()) & set(node_dict2.keys())
    
    if common_nodes:
        type_mismatches = sum(1 for nid in common_nodes if node_dict1[nid] != node_dict2[nid])
        type_diff = type_mismatches / len(common_nodes)
    else:
        type_diff = 1.0
    
    # Weighted combination
    structural_distance = (
        0.25 * node_diff +
        0.25 * conn_diff +
        0.35 * jaccard_dist +
        0.15 * type_diff
    )
    
    return min(structural_distance, 1.0)


def compute_weight_distance(genotype1: dict, genotype2: dict, metric: str = 'euclidean') -> float:
    """
    Compute weight distance between two HyperNEAT genotypes.
    
    This measures how different the connection weights are.
    Only considers connections that exist in both genotypes.
    
    Args:
        genotype1: First HyperNEAT genotype dictionary
        genotype2: Second HyperNEAT genotype dictionary
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        
    Returns:
        Normalized weight distance [0, 1]
    """
    if not isinstance(genotype1, dict) or not isinstance(genotype2, dict):
        return 1.0
    
    conns1 = genotype1.get('connections', [])
    conns2 = genotype2.get('connections', [])
    
    # Build weight dictionaries keyed by (in_node, out_node)
    weights1 = {(conn.in_node, conn.out_node): conn.weight for conn in conns1 if conn.enabled}
    weights2 = {(conn.in_node, conn.out_node): conn.weight for conn in conns2 if conn.enabled}
    
    # Find common connections
    common_keys = set(weights1.keys()) & set(weights2.keys())
    
    if not common_keys:
        # No common connections -> maximum distance
        return 1.0
    
    # Extract weight vectors for common connections
    w1 = np.array([weights1[key] for key in common_keys])
    w2 = np.array([weights2[key] for key in common_keys])
    
    # Compute distance based on metric
    if metric == 'euclidean':
        distance = euclidean(w1, w2)
        # Normalize by max possible distance (assuming weights in [-3, 3])
        max_dist = np.sqrt(len(common_keys)) * 6.0  # Conservative estimate
        normalized_distance = min(distance / max_dist, 1.0) if max_dist > 0 else 0.0
    elif metric == 'manhattan':
        distance = np.sum(np.abs(w1 - w2))
        max_dist = len(common_keys) * 6.0
        normalized_distance = min(distance / max_dist, 1.0) if max_dist > 0 else 0.0
    elif metric == 'cosine':
        # Cosine distance (1 - similarity)
        if np.linalg.norm(w1) > 0 and np.linalg.norm(w2) > 0:
            normalized_distance = 1.0 - np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
            normalized_distance = max(0.0, min(normalized_distance, 1.0))
        else:
            normalized_distance = 1.0
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return normalized_distance


def compute_combined_distance(
    genotype1: dict,
    genotype2: dict,
    structural_weight: float = 0.5,
    weight_weight: float = 0.5,
    weight_metric: str = 'euclidean'
) -> float:
    """
    Compute combined structural and weight distance.
    
    Args:
        genotype1: First HyperNEAT genotype dictionary
        genotype2: Second HyperNEAT genotype dictionary
        structural_weight: Weight for structural distance (default: 0.5)
        weight_weight: Weight for weight distance (default: 0.5)
        weight_metric: Metric for weight distance
        
    Returns:
        Normalized combined distance [0, 1]
    """
    struct_dist = compute_structural_distance(genotype1, genotype2)
    weight_dist = compute_weight_distance(genotype1, genotype2, metric=weight_metric)
    
    combined = structural_weight * struct_dist + weight_weight * weight_dist
    return min(combined, 1.0)


def compute_behavioral_distance(
    fitness1: float,
    fitness2: float,
    trajectory1: np.ndarray | None = None,
    trajectory2: np.ndarray | None = None
) -> float:
    """
    Compute behavioral distance based on fitness and trajectories.
    
    This measures how different two individuals behave, regardless of genotype.
    Useful for identifying convergent evolution (different genotypes, similar behavior).
    
    Args:
        fitness1: Fitness of first individual
        fitness2: Fitness of second individual
        trajectory1: Movement trajectory of first individual (optional)
        trajectory2: Movement trajectory of second individual (optional)
        
    Returns:
        Normalized behavioral distance [0, 1]
    """
    # Fitness difference (normalized)
    fitness_diff = abs(fitness1 - fitness2)
    # Assume fitness is typically in [0, 10] range
    normalized_fitness_diff = min(fitness_diff / 10.0, 1.0)
    
    # If trajectories are provided, compute trajectory similarity
    if trajectory1 is not None and trajectory2 is not None:
        if len(trajectory1) == 0 or len(trajectory2) == 0:
            trajectory_dist = 1.0
        else:
            # Resample to same length
            min_len = min(len(trajectory1), len(trajectory2))
            traj1_resampled = trajectory1[:min_len]
            traj2_resampled = trajectory2[:min_len]
            
            # Compute average position distance
            distances = np.linalg.norm(traj1_resampled - traj2_resampled, axis=1)
            avg_dist = np.mean(distances)
            
            # Normalize by world size (assume ~50 unit world)
            trajectory_dist = min(avg_dist / 50.0, 1.0)
        
        # Weighted combination
        behavioral_dist = 0.5 * normalized_fitness_diff + 0.5 * trajectory_dist
    else:
        # Only fitness available
        behavioral_dist = normalized_fitness_diff
    
    return behavioral_dist


def compute_pairwise_distance_matrix(
    genotypes: list[dict],
    distance_type: str = 'combined',
    structural_weight: float = 0.5,
    weight_weight: float = 0.5,
    fitness_values: list[float] | None = None,
    trajectories: list[np.ndarray] | None = None
) -> np.ndarray:
    """
    Compute pairwise distance matrix for a list of genotypes.
    
    Args:
        genotypes: List of HyperNEAT genotype dictionaries
        distance_type: Type of distance ('structural', 'weight', 'combined', 'behavioral')
        structural_weight: Weight for structural component in 'combined' mode
        weight_weight: Weight for weight component in 'combined' mode
        fitness_values: List of fitness values (required for 'behavioral')
        trajectories: List of trajectories (optional for 'behavioral')
        
    Returns:
        Symmetric distance matrix (n x n) where n = len(genotypes)
    """
    n = len(genotypes)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            if distance_type == 'structural':
                dist = compute_structural_distance(genotypes[i], genotypes[j])
            elif distance_type == 'weight':
                dist = compute_weight_distance(genotypes[i], genotypes[j])
            elif distance_type == 'combined':
                dist = compute_combined_distance(
                    genotypes[i], genotypes[j],
                    structural_weight=structural_weight,
                    weight_weight=weight_weight
                )
            elif distance_type == 'behavioral':
                if fitness_values is None:
                    raise ValueError("fitness_values required for behavioral distance")
                traj1 = trajectories[i] if trajectories else None
                traj2 = trajectories[j] if trajectories else None
                dist = compute_behavioral_distance(
                    fitness_values[i], fitness_values[j],
                    trajectory1=traj1, trajectory2=traj2
                )
            else:
                raise ValueError(f"Unknown distance_type: {distance_type}")
            
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix


def compute_genotype_diversity(genotypes: list[dict], distance_type: str = 'combined') -> dict:
    """
    Compute diversity metrics for a population of genotypes.
    
    Args:
        genotypes: List of HyperNEAT genotype dictionaries
        distance_type: Type of distance metric to use
        
    Returns:
        Dictionary with diversity metrics:
        - mean_distance: Average pairwise distance
        - std_distance: Standard deviation of distances
        - min_distance: Minimum pairwise distance
        - max_distance: Maximum pairwise distance
        - diversity_index: Overall diversity index [0, 1]
    """
    if len(genotypes) < 2:
        return {
            'mean_distance': 0.0,
            'std_distance': 0.0,
            'min_distance': 0.0,
            'max_distance': 0.0,
            'diversity_index': 0.0
        }
    
    # Compute distance matrix
    dist_matrix = compute_pairwise_distance_matrix(genotypes, distance_type=distance_type)
    
    # Extract upper triangle (excluding diagonal)
    upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    
    return {
        'mean_distance': float(np.mean(upper_triangle)),
        'std_distance': float(np.std(upper_triangle)),
        'min_distance': float(np.min(upper_triangle)),
        'max_distance': float(np.max(upper_triangle)),
        'diversity_index': float(np.mean(upper_triangle))  # Simple diversity measure
    }


def find_most_similar_pairs(
    genotypes: list[dict],
    individual_ids: list[int],
    distance_type: str = 'combined',
    top_k: int = 5
) -> list[tuple[int, int, float]]:
    """
    Find the most similar pairs of genotypes.
    
    Args:
        genotypes: List of HyperNEAT genotype dictionaries
        individual_ids: List of individual IDs corresponding to genotypes
        distance_type: Type of distance metric
        top_k: Number of most similar pairs to return
        
    Returns:
        List of tuples (id1, id2, distance) sorted by distance (ascending)
    """
    dist_matrix = compute_pairwise_distance_matrix(genotypes, distance_type=distance_type)
    n = len(genotypes)
    
    # Extract all pairs with distances
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((individual_ids[i], individual_ids[j], dist_matrix[i, j]))
    
    # Sort by distance
    pairs.sort(key=lambda x: x[2])
    
    return pairs[:top_k]


def find_most_different_pairs(
    genotypes: list[dict],
    individual_ids: list[int],
    distance_type: str = 'combined',
    top_k: int = 5
) -> list[tuple[int, int, float]]:
    """
    Find the most different pairs of genotypes.
    
    Args:
        genotypes: List of HyperNEAT genotype dictionaries
        individual_ids: List of individual IDs corresponding to genotypes
        distance_type: Type of distance metric
        top_k: Number of most different pairs to return
        
    Returns:
        List of tuples (id1, id2, distance) sorted by distance (descending)
    """
    dist_matrix = compute_pairwise_distance_matrix(genotypes, distance_type=distance_type)
    n = len(genotypes)
    
    # Extract all pairs with distances
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((individual_ids[i], individual_ids[j], dist_matrix[i, j]))
    
    # Sort by distance (descending)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    return pairs[:top_k]
