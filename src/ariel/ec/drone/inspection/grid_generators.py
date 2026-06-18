"""
Grid generation utilities for creating visualization layouts.

This module provides functions for generating different types of grid layouts
for visualizing evolutionary populations, including fitness grids, comparison grids,
and population overview grids.

# TODO: [MEDIUM] Add comprehensive input validation to all grid generation functions
# TODO: [MEDIUM] Standardize parameter naming across functions (e.g., twod vs use_2d)
# TODO: [LOW] Add more grid layout options (circular, hierarchical, etc.)
# TODO: [LOW] Consider adding caching for expensive grid computations
"""

import functools
import numpy as np
from typing import List, Callable, Tuple, Optional, Union
import numpy.typing as npt

from .population_filtering import (
    get_top_k_individuals,
    get_generation_indices_for_limit,
    select_best_per_generation,
    flatten_population_across_generations
)

# Import visualization components
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer
from ariel.ec.drone.inspection.create_subplot import remove_ticks
from ariel.ec.drone.inspection.plot_fitness import plot_fitness


def generate_best_individuals_fitness_grid(
    population_data: List[npt.NDArray],
    fitness_data: List[npt.NDArray],
    twod: bool = True,
    small_grid: bool = False,
    include_motor_orientation: Union[bool, int] = True,
    worst: bool = False,
    title: Optional[str] = None
) -> List[npt.NDArray]:
    """
    Generate grids showing best individuals from each generation with fitness plot.
    
    Args:
        population_data: List of population arrays for each generation
        fitness_data: List of fitness arrays for each generation
        twod: If True, use 2D visualization; if False, use 3D
        small_grid: If True, use 3x3 grid; if False, use 4x4 grid
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals instead of best
        title: Title for the grid
        
    Returns:
        List of grid arrays, one for each generation
    """
    grids = []
    k = 8 if small_grid else 15  # Number of individuals to show
    grid_size = 3 if small_grid else 4
    
    for gen, (population, fitness) in enumerate(zip(population_data, fitness_data)):
        # Get top individuals for this generation
        if worst:
            top_individuals, top_fitnesses, _ = get_top_k_individuals(
                population, fitness, k, reverse=True
            )
        else:
            top_individuals, top_fitnesses, _ = get_top_k_individuals(
                population, fitness, k, reverse=False
            )
        
        # Create grid of visualization functions
        grid = []
        for i, (individual, fit_score) in enumerate(zip(top_individuals, top_fitnesses)):
            # Determine label positioning
            labelleft = (i % grid_size == 0)
            labelbottom = ((grid_size * (grid_size - 1)) <= i < (grid_size * grid_size))
            
            visualizer = DroneVisualizer()
            if twod:
                viz_func = functools.partial(
                    visualizer.plot_2d,
                    genome_data=individual,
                    title=f"Fitness: {np.round(fit_score, 2)}"
                )
            else:
                viz_func = functools.partial(
                    visualizer.plot_3d,
                    genome_data=individual,
                    title=f"Fitness: {np.round(fit_score, 2)}"
                )
            
            grid.append(viz_func)
        
        # Add fitness plot in the corner
        fitness_func = functools.partial(
            plot_fitness,
            population_data=population_data,
            fitness_data=fitness_data,
            gen_line=gen
        )
        grid.append(fitness_func)
        
        # Reshape to square grid
        grid = np.array(grid).reshape(grid_size, grid_size)
        grids.append(grid)
    
    return grids


def generate_population_grid(
    population: npt.NDArray,
    fitnesses: Optional[npt.NDArray] = None,
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0
) -> List[List[Callable]]:
    """
    Generate a grid layout for visualizing an entire population.
    
    Args:
        population: Array of individuals
        fitnesses: Optional array of fitness values
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        
    Returns:
        2D list of visualization functions arranged in a grid
    """
    pop_size = len(population)
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(pop_size)))
    cols = int(np.ceil(pop_size / rows))
    
    grid = []
    for r in range(rows):
        row = []
        for c in range(cols):
            idx = r * cols + c
            
            # Determine label positioning
            labelleft = (c == 0)
            labelbottom = (r == rows - 1)
            
            if idx < pop_size:
                individual = population[idx]
                fitness = fitnesses[idx] if fitnesses is not None else None
                
                visualizer = DroneVisualizer()
                fitness_title = f"Fitness: {np.round(fitness, 2)}" if fitness is not None else "No fitness"
                if twod:
                    viz_func = functools.partial(
                        visualizer.plot_2d,
                        genome_data=individual,
                        title=fitness_title
                    )
                else:
                    viz_func = functools.partial(
                        visualizer.plot_3d,
                        genome_data=individual,
                        title=fitness_title
                    )
                row.append(viz_func)
            else:
                # Empty cell
                row.append(remove_ticks)
        
        grid.append(row)
    
    return grid


def generate_top_individuals_across_generations_grid(
    population_data: List[npt.NDArray],
    fitness_data: List[npt.NDArray],
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    show_ticks: bool = True
) -> List[List[Callable]]:
    """
    Generate a grid showing top individuals across selected generations.
    
    Args:
        population_data: List of population arrays
        fitness_data: List of fitness arrays
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals
        show_ticks: Whether to show axis ticks and labels
        
    Returns:
        2D list of visualization functions
    """
    rows, cols = grid_size
    num_gens = len(population_data)
    
    # Get evenly spaced generation indices
    gen_indices = get_generation_indices_for_limit(num_gens, cols)
    
    # Get top individuals from selected generations
    top_individuals_per_gen = []
    for idx in gen_indices:
        if worst:
            top_inds, top_fits, _ = get_top_k_individuals(
                population_data[idx], fitness_data[idx], rows, reverse=True
            )
        else:
            top_inds, top_fits, _ = get_top_k_individuals(
                population_data[idx], fitness_data[idx], rows, reverse=False
            )
        top_individuals_per_gen.append((top_inds, top_fits))
    
    # Create grid
    grid = []
    for row in range(rows):
        grid_row = []
        for col in range(cols):
            if col < len(top_individuals_per_gen):
                individuals, fitnesses = top_individuals_per_gen[col]
                
                if row < len(individuals):
                    individual = individuals[row]
                    fitness = fitnesses[row]
                    generation = gen_indices[col]
                    
                    visualizer = DroneVisualizer()
                    if twod:
                        viz_func = functools.partial(
                            visualizer.plot_2d,
                            genome_data=individual,
                            title=f"Gen {generation}: {np.round(fitness, 2)}"
                        )
                    else:
                        viz_func = functools.partial(
                            visualizer.plot_3d,
                            genome_data=individual,
                            title=f"Gen {generation}: {np.round(fitness, 2)}"
                        )
                    
                    grid_row.append(viz_func)
                else:
                    grid_row.append(remove_ticks)
            else:
                grid_row.append(remove_ticks)
        
        grid.append(grid_row)
    
    return grid


def generate_best_per_generation_grid(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    title: Optional[str] = None
) -> List[List[Callable]]:
    """
    Generate a grid showing the best individual from selected generations.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals
        title: Title for the grid
        
    Returns:
        2D list of visualization functions
    """
    rows, cols = grid_size
    num_gens = len(population_data)
    num_slots = rows * cols
    
    # Get evenly spaced generation indices
    gen_indices = get_generation_indices_for_limit(num_gens, num_slots)
    
    # Get best individual from each selected generation
    best_individuals, best_fitnesses = select_best_per_generation(
        population_data, fitness_data, worst=worst
    )
    
    # Create visualization functions
    viz_functions = []
    for i, gen_idx in enumerate(gen_indices):
        individual = best_individuals[gen_idx]
        fitness = best_fitnesses[gen_idx]
        
        visualizer = DroneVisualizer()
        if twod:
            viz_func = functools.partial(
                visualizer.plot_2d,
                genome_data=individual,
                title=f"Gen {gen_idx}: {np.round(fitness, 2)}"
            )
        else:
            viz_func = functools.partial(
                visualizer.plot_3d,
                genome_data=individual,
                title=f"Gen {gen_idx}: {np.round(fitness, 2)}"
            )
        
        viz_functions.append(viz_func)
    
    # Arrange in grid
    grid = np.array(viz_functions).reshape(grid_size)
    return grid.tolist()


def generate_best_ever_grid(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    title: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    round_to: int = 2
) -> Tuple[List[List[Callable]], npt.NDArray, npt.NDArray]:
    """
    Generate a grid showing the best individuals across all generations.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals
        title: Title for the grid
        similarity_threshold: If provided, ensure diversity in selection
        round_to: Number of decimal places for fitness display
        
    Returns:
        Tuple of (grid, top_individuals, top_fitnesses)
    """
    rows, cols = grid_size
    k = rows * cols
    
    # Flatten across all generations
    flat_population, flat_fitnesses = flatten_population_across_generations(
        population_data, fitness_data, remove_nan=True
    )
    
    # Get top individuals
    if worst:
        top_individuals, top_fitnesses, _ = get_top_k_individuals(
            flat_population, flat_fitnesses, k, reverse=True,
            similarity_threshold=similarity_threshold
        )
    else:
        top_individuals, top_fitnesses, _ = get_top_k_individuals(
            flat_population, flat_fitnesses, k, reverse=False,
            similarity_threshold=similarity_threshold
        )
    
    # Create grid
    grid = []
    for i, (individual, fitness) in enumerate(zip(top_individuals, top_fitnesses)):
        # Determine label positioning
        labelleft = (i % cols == 0)
        labelbottom = ((cols * (rows - 1)) <= i < (cols * rows))
        
        visualizer = DroneVisualizer()
        if twod:
            viz_func = functools.partial(
                visualizer.plot_2d,
                genome_data=individual,
                title=f"Fitness: {np.round(fitness, round_to)}"
            )
        else:
            viz_func = functools.partial(
                visualizer.plot_3d,
                genome_data=individual,
                title=f"Fitness: {np.round(fitness, round_to)}"
            )
        
        grid.append(viz_func)
    
    # Arrange in grid
    grid_2d = np.array(grid).reshape(grid_size)
    
    return grid_2d.tolist(), np.array(top_individuals), np.array(top_fitnesses)


def generate_comparison_grid(
    population_lists: List[List[npt.NDArray]],
    fitness_lists: List[List[npt.NDArray]],
    labels: List[str],
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0
) -> List[List[Callable]]:
    """
    Generate a grid comparing best individuals from multiple evolution runs.
    
    Args:
        population_lists: List of population data lists (one per run)
        fitness_lists: List of fitness data lists (one per run)
        labels: Labels for each evolution run
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        
    Returns:
        2D list of visualization functions
    """
    rows, cols = grid_size
    num_runs = len(population_lists)
    
    if len(fitness_lists) != num_runs or len(labels) != num_runs:
        raise ValueError("Number of runs must match across all inputs")
    
    # Get best individuals from each run
    best_individuals = []
    best_fitnesses = []
    
    for pop_data, fit_data in zip(population_lists, fitness_lists):
        # Flatten across generations for this run
        flat_pop, flat_fit = flatten_population_across_generations(
            np.array(pop_data), np.array(fit_data), remove_nan=True
        )
        
        # Get single best individual
        best_idx = np.argmax(flat_fit)
        best_individuals.append(flat_pop[best_idx])
        best_fitnesses.append(flat_fit[best_idx])
    
    # Create grid with run comparisons
    grid = []
    items_per_row = cols // num_runs if cols >= num_runs else 1
    
    for row in range(rows):
        grid_row = []
        for col in range(cols):
            run_idx = col // items_per_row
            
            if run_idx < num_runs:
                individual = best_individuals[run_idx]
                fitness = best_fitnesses[run_idx]
                label = labels[run_idx]
                
                visualizer = DroneVisualizer()
                if twod:
                    viz_func = functools.partial(
                        visualizer.plot_2d,
                        genome_data=individual,
                        title=f"{label}: {np.round(fitness, 2)}"
                    )
                else:
                    viz_func = functools.partial(
                        visualizer.plot_3d,
                        genome_data=individual,
                        title=f"{label}: {np.round(fitness, 2)}"
                    )
                
                grid_row.append(viz_func)
            else:
                grid_row.append(remove_ticks)
        
        grid.append(grid_row)
    
    return grid


def calculate_optimal_grid_size(num_items: int, max_cols: int = 8) -> Tuple[int, int]:
    """
    Calculate optimal grid dimensions for a given number of items.
    
    Args:
        num_items: Number of items to arrange
        max_cols: Maximum number of columns allowed
        
    Returns:
        Tuple of (rows, cols) for optimal layout
        
    Raises:
        ValueError: If num_items or max_cols are invalid
    """
    if not isinstance(num_items, int):
        raise TypeError(f"num_items must be an integer, got {type(num_items)}")
    if not isinstance(max_cols, int):
        raise TypeError(f"max_cols must be an integer, got {type(max_cols)}")
    
    if num_items < 0:
        raise ValueError(f"num_items must be non-negative, got {num_items}")
    if max_cols <= 0:
        raise ValueError(f"max_cols must be positive, got {max_cols}")
        
    if num_items == 0:
        return (1, 1)
    
    # Try to make it as square as possible
    sqrt_items = int(np.ceil(np.sqrt(num_items)))
    
    if sqrt_items <= max_cols:
        cols = sqrt_items
        rows = int(np.ceil(num_items / cols))
    else:
        cols = max_cols
        rows = int(np.ceil(num_items / cols))
    
    return (rows, cols)


def add_grid_labels(
    grid: List[List[Callable]],
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None
) -> List[List[Callable]]:
    """
    Add row and column labels to a grid layout.
    
    Args:
        grid: 2D list of visualization functions
        row_labels: Optional labels for rows
        col_labels: Optional labels for columns
        
    Returns:
        Modified grid with label functions added
    """
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Create new grid with space for labels
    new_grid = []
    
    # Add column labels if provided
    if col_labels:
        label_row = []
        if row_labels:
            label_row.append(remove_ticks)  # Empty corner
        
        for col, label in enumerate(col_labels[:cols]):
            label_func = functools.partial(
                lambda ax, text=label: ax.text(
                    0.5, 0.5, text, ha='center', va='center',
                    fontsize=12, fontweight='bold'
                )
            )
            label_row.append(label_func)
        
        new_grid.append(label_row)
    
    # Add rows with optional row labels
    for row_idx, row in enumerate(grid):
        new_row = []
        
        if row_labels and row_idx < len(row_labels):
            label_func = functools.partial(
                lambda ax, text=row_labels[row_idx]: ax.text(
                    0.5, 0.5, text, ha='center', va='center',
                    fontsize=12, fontweight='bold', rotation=90
                )
            )
            new_row.append(label_func)
        
        new_row.extend(row)
        new_grid.append(new_row)
    
    return new_grid


# Backward compatibility functions
get_best_individuals_fitness_grid = generate_best_individuals_fitness_grid
get_grid_for_population = generate_population_grid