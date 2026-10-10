"""
Population selection and filtering utilities for evolutionary visualization.

This module provides functions for selecting, filtering, and ranking individuals
from evolutionary populations based on fitness and similarity metrics.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import numpy.typing as npt

# Import edit distance functions
from ariel.ec.drone.evaluators.edit_distance import (
    compute_edit_distance,
    compute_individual_population_edit_distance
)


def filter_by_unique_individuals(
    population: npt.NDArray,
    fitnesses: npt.NDArray,
    similarity_threshold: float,
    parameter_limits: Optional[npt.NDArray] = None,
    max_num_arms: int = 6,
    fitness_threshold: Optional[float] = None,
    reverse: bool = False
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Filter population to keep only unique individuals based on similarity threshold.
    
    Args:
        population: Array of individuals
        fitnesses: Array of fitness values
        similarity_threshold: Maximum similarity allowed between selected individuals
        parameter_limits: Parameter bounds for edit distance calculation
        max_num_arms: Maximum number of arms for distance normalization
        fitness_threshold: Minimum fitness threshold for inclusion
        reverse: If True, sort by ascending fitness (worst first)
        
    Returns:
        Tuple of (selected_population, selected_fitnesses)
    """
    # Apply fitness threshold filter
    if fitness_threshold is not None:
        fitnesses = np.array(fitnesses)
        mask = fitnesses > fitness_threshold
        population = population[mask]
        fitnesses = fitnesses[mask]
    
    # Default parameter limits if not provided
    if parameter_limits is None:
        parameter_limits = np.array([
            [0.09, 0.4],      # magnitude
            [0, 2*np.pi],     # azimuth rotation
            [0, 2*np.pi],     # azimuth pitch
            [0, 2*np.pi],     # motor rotation
            [0, 2*np.pi],     # motor pitch
            [0, 1]            # direction
        ])
    
    # Extract min/max values from parameter_limits
    if parameter_limits.shape[1] == 2:
        min_vals = parameter_limits[:, 0]
        max_vals = parameter_limits[:, 1]
    else:
        raise ValueError("parameter_limits should have shape (n_params, 2)")
    
    selected_population = []
    selected_fitnesses = []
    
    # Calculate diversity-adjusted fitness
    updated_fitnesses = []
    for ind, fitness in zip(population, fitnesses):
        distance = compute_individual_population_edit_distance(ind, population, min_vals, max_vals)
        distance = min(distance, max_num_arms - 0.001)  # Clamp to valid range
        
        # Add diversity bonus (normalized distance)
        updated_fitness = fitness + distance / max_num_arms
        updated_fitnesses.append(updated_fitness)
    
    updated_fitnesses = np.array(updated_fitnesses)
    
    # Sort by updated fitness
    if reverse:
        sorted_indices = np.argsort(updated_fitnesses)
    else:
        sorted_indices = np.argsort(updated_fitnesses)[::-1]
    
    # Select diverse individuals
    for idx in sorted_indices:
        ind1 = population[idx]
        is_similar = False
        
        # Check similarity with already selected individuals
        for selected_ind in selected_population:
            if compute_edit_distance(ind1, selected_ind, min_vals, max_vals) < similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            selected_population.append(ind1)
            selected_fitnesses.append(updated_fitnesses[idx])
    
    return np.array(selected_population), np.array(selected_fitnesses)


def get_top_k_individuals(
    population: Union[npt.NDArray, List[npt.NDArray]],
    fitnesses: Union[npt.NDArray, List[npt.NDArray]],
    k: int,
    reverse: bool = False,
    similarity_threshold: Optional[float] = None,
    parameter_limits: Optional[npt.NDArray] = None,
    max_num_arms: int = 8
) -> Union[Tuple[List, List], Tuple[List, List, List]]:
    """
    Select the top k individuals from a population or across generations.
    
    Args:
        population: Individual population array or list of generation arrays
        fitnesses: Individual fitness array or list of generation fitness arrays
        k: Number of individuals to select
        reverse: If True, select worst individuals instead of best
        similarity_threshold: If provided, ensure selected individuals are diverse
        parameter_limits: Parameter bounds for similarity calculation
        max_num_arms: Maximum number of arms for distance normalization
        
    Returns:
        If similarity_threshold is None: (selected_population, selected_fitnesses)
        If similarity_threshold is not None: (selected_population, selected_fitnesses, selected_indices)
    """
    # Check if we have multiple generations
    is_multi_generation = isinstance(population, list) or (
        isinstance(population, np.ndarray) and population.ndim > 2
    )
    
    if similarity_threshold is not None:
        return _get_top_k_diverse_individuals(
            population, fitnesses, k, reverse, similarity_threshold,
            parameter_limits, max_num_arms, is_multi_generation
        )
    else:
        return _get_top_k_simple(
            population, fitnesses, k, reverse, is_multi_generation
        )


def _get_top_k_diverse_individuals(
    population: Union[npt.NDArray, List[npt.NDArray]],
    fitnesses: Union[npt.NDArray, List[npt.NDArray]],
    k: int,
    reverse: bool,
    similarity_threshold: float,
    parameter_limits: Optional[npt.NDArray],
    max_num_arms: int,
    is_multi_generation: bool
) -> Tuple[List, List, List]:
    """Helper function for diverse individual selection."""
    # Default parameter limits
    if parameter_limits is None:
        parameter_limits = np.array([
            [0.09, 0.4], [0, 2*np.pi], [0, 2*np.pi],
            [0, 2*np.pi], [0, 2*np.pi], [0, 1]
        ])
    
    # Extract min/max values from parameter_limits
    if parameter_limits.shape[1] == 2:
        min_vals = parameter_limits[:, 0]
        max_vals = parameter_limits[:, 1]
    else:
        raise ValueError("parameter_limits should have shape (n_params, 2)")
    
    selected_population = []
    selected_fitnesses = []
    selected_indices = []
    
    # Create list of all individuals with their fitness and indices
    all_individuals = []
    
    if is_multi_generation:
        for gen_idx, (generation, gen_fitnesses) in enumerate(zip(population, fitnesses)):
            for ind_idx, (ind, fitness) in enumerate(zip(generation, gen_fitnesses)):
                # Calculate diversity-adjusted fitness
                distance = compute_individual_population_edit_distance(ind, generation, min_vals, max_vals)
                distance = min(distance, max_num_arms - 0.001)  # Clamp to valid range
                updated_fitness = fitness + distance / max_num_arms
                all_individuals.append((updated_fitness, gen_idx, ind_idx, ind, fitness))
    else:
        for ind_idx, (ind, fitness) in enumerate(zip(population, fitnesses)):
            distance = compute_individual_population_edit_distance(ind, population, min_vals, max_vals)
            distance = min(distance, max_num_arms - 0.001)  # Clamp to valid range
            updated_fitness = fitness + distance / max_num_arms
            all_individuals.append((updated_fitness, 0, ind_idx, ind, fitness))
    
    # Sort by updated fitness
    all_individuals.sort(key=lambda x: x[0], reverse=not reverse)
    
    # Select diverse individuals
    for updated_fitness, gen_idx, ind_idx, ind, original_fitness in all_individuals:
        if len(selected_population) >= k:
            break
        
        is_similar = False
        for selected_ind in selected_population:
            if compute_edit_distance(ind, selected_ind, min_vals, max_vals) < similarity_threshold:
                is_similar = True
                break
        
        if not is_similar:
            selected_population.append(ind)
            selected_fitnesses.append(original_fitness)
            selected_indices.append((gen_idx, ind_idx))
    
    return selected_population, selected_fitnesses, selected_indices


def _get_top_k_simple(
    population: Union[npt.NDArray, List[npt.NDArray]],
    fitnesses: Union[npt.NDArray, List[npt.NDArray]],
    k: int,
    reverse: bool,
    is_multi_generation: bool
) -> Tuple[List, List, List]:
    """Helper function for simple top-k selection without diversity constraint."""
    all_individuals = []
    
    if is_multi_generation:
        for gen_idx, gen_fitnesses in enumerate(fitnesses):
            for ind_idx, fitness in enumerate(gen_fitnesses):
                all_individuals.append((fitness, gen_idx, ind_idx))
    else:
        for ind_idx, fitness in enumerate(fitnesses):
            all_individuals.append((fitness, 0, ind_idx))
    
    # Sort by fitness
    all_individuals.sort(key=lambda x: x[0], reverse=not reverse)
    
    # Select top k
    top_k = all_individuals[:k]
    
    if is_multi_generation:
        top_k_population = [population[gen_idx][ind_idx] for _, gen_idx, ind_idx in top_k]
    else:
        top_k_population = [population[ind_idx] for _, _, ind_idx in top_k]
    
    top_k_fitnesses = [fitness for fitness, _, _ in top_k]
    top_k_indices = [(gen_idx, ind_idx) for _, gen_idx, ind_idx in top_k]
    
    return top_k_population, top_k_fitnesses, top_k_indices


def get_generation_indices_for_limit(num_generations: int, limit: int) -> npt.NDArray:
    """
    Get evenly spaced generation indices for visualization with a specific limit.
    
    Args:
        num_generations: Total number of generations
        limit: Maximum number of generations to include
        
    Returns:
        Array of generation indices including first and last generation
    """
    if limit >= num_generations:
        return np.arange(num_generations)
    
    if limit < 2:
        return np.array([0])
    
    step = np.round(num_generations / (limit - 2))
    indices = np.arange(step, num_generations, step=step)
    
    if len(indices) == 0:
        indices = np.array([0])
    
    # Adjust to get exactly limit-2 middle indices
    while len(indices) != limit - 2:
        if len(indices) < limit - 2:
            indices = np.append(indices, indices[-1] + 1)
        elif len(indices) > limit - 2:
            indices = np.delete(indices, -1)
    
    # Add first and last generation
    indices = np.insert(indices, 0, 0)
    indices = np.append(indices, num_generations - 1)
    
    return indices.astype(int)


def select_best_per_generation(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    worst: bool = False
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Select the best (or worst) individual from each generation.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        worst: If True, select worst individuals instead of best
        
    Returns:
        Tuple of (best_individuals, best_fitnesses)
    """
    if worst:
        arg_indices = np.argmin(fitness_data, axis=1)
    else:
        arg_indices = np.argmax(fitness_data, axis=1)
    
    best_individuals = population_data[np.arange(population_data.shape[0]), arg_indices]
    best_fitnesses = fitness_data[np.arange(fitness_data.shape[0]), arg_indices]
    
    return best_individuals, best_fitnesses


def flatten_population_across_generations(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    remove_nan: bool = True
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Flatten population data across all generations into a single population.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        remove_nan: If True, remove individuals with NaN fitness
        
    Returns:
        Tuple of (flattened_population, flattened_fitnesses)
    """
    n_gens, pop_size, n_arms, n_params = population_data.shape
    
    # Reshape to combine all generations
    flattened_population = population_data.reshape((n_gens * pop_size, n_arms, n_params))
    flattened_fitnesses = fitness_data.flatten()
    
    if remove_nan:
        # Remove individuals with NaN fitness
        mask = ~np.isnan(flattened_fitnesses)
        flattened_population = flattened_population[mask]
        flattened_fitnesses = flattened_fitnesses[mask]
    
    return flattened_population, flattened_fitnesses


def get_fitness_statistics(fitness_data: npt.NDArray) -> dict:
    """
    Calculate statistics for fitness data across generations.
    
    Args:
        fitness_data: Array of shape (n_generations, pop_size)
        
    Returns:
        Dictionary with fitness statistics
    """
    # Remove NaN values for statistics
    valid_fitness = fitness_data[~np.isnan(fitness_data)]
    
    if len(valid_fitness) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q25': np.nan,
            'q75': np.nan
        }
    
    return {
        'mean': np.mean(valid_fitness),
        'std': np.std(valid_fitness),
        'min': np.min(valid_fitness),
        'max': np.max(valid_fitness),
        'median': np.median(valid_fitness),
        'q25': np.percentile(valid_fitness, 25),
        'q75': np.percentile(valid_fitness, 75)
    }


def get_generation_fitness_statistics(fitness_data: npt.NDArray) -> dict:
    """
    Calculate fitness statistics for each generation.
    
    Args:
        fitness_data: Array of shape (n_generations, pop_size)
        
    Returns:
        Dictionary with per-generation statistics
    """
    n_generations = fitness_data.shape[0]
    
    stats = {
        'mean': np.zeros(n_generations),
        'std': np.zeros(n_generations),
        'min': np.zeros(n_generations),
        'max': np.zeros(n_generations),
        'median': np.zeros(n_generations)
    }
    
    for gen in range(n_generations):
        gen_fitness = fitness_data[gen]
        valid_fitness = gen_fitness[~np.isnan(gen_fitness)]
        
        if len(valid_fitness) > 0:
            stats['mean'][gen] = np.mean(valid_fitness)
            stats['std'][gen] = np.std(valid_fitness)
            stats['min'][gen] = np.min(valid_fitness)
            stats['max'][gen] = np.max(valid_fitness)
            stats['median'][gen] = np.median(valid_fitness)
        else:
            stats['mean'][gen] = np.nan
            stats['std'][gen] = np.nan
            stats['min'][gen] = np.nan
            stats['max'][gen] = np.nan
            stats['median'][gen] = np.nan
    
    return stats


# Backward compatibility aliases
get_top_k_individuals_legacy = get_top_k_individuals
get_idxs_between = get_generation_indices_for_limit