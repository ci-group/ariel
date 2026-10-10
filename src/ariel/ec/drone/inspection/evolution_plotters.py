"""
High-level plotting functions for evolutionary visualization.

This module provides the main plotting interfaces for visualizing evolutionary
populations, combining grid generation with subplot creation and display.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import functools
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer

from .grid_generators import (
    generate_best_individuals_fitness_grid,
    generate_population_grid,
    generate_top_individuals_across_generations_grid,
    generate_best_per_generation_grid,
    generate_best_ever_grid,
    generate_comparison_grid
)
from .create_subplot import create_subplot


def plot_top_individuals_across_generations(
    population_data: List[npt.NDArray],
    fitness_data: List[npt.NDArray],
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    title: Optional[str] = None,
    show_ticks: bool = True
) -> Tuple[plt.Figure, npt.NDArray, List, List]:
    """
    Plot top individuals from selected generations across evolution.
    
    Args:
        population_data: List of population arrays for each generation
        fitness_data: List of fitness arrays for each generation
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization; if False, use 3D
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals instead of best
        title: Title for the plot
        show_ticks: Whether to show axis ticks and labels
        
    Returns:
        Tuple of (figure, axes, top_individuals_per_generation, selected_generation_indices)
    """
    # Generate grid layout
    grid = generate_top_individuals_across_generations_grid(
        population_data=population_data,
        fitness_data=fitness_data,
        grid_size=grid_size,
        twod=twod,
        include_motor_orientation=include_motor_orientation,
        worst=worst,
        show_ticks=show_ticks
    )
    
    # Create subplot from grid
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, title=title)
    
    # Return additional data for analysis
    from .population_filtering import get_generation_indices_for_limit, get_top_k_individuals
    
    num_gens = len(population_data)
    gen_indices = get_generation_indices_for_limit(num_gens, grid_size[1])
    
    top_individuals_per_gen = []
    for idx in gen_indices:
        if worst:
            top_inds, top_fits, _ = get_top_k_individuals(
                population_data[idx], fitness_data[idx], grid_size[0], reverse=True
            )
        else:
            top_inds, top_fits, _ = get_top_k_individuals(
                population_data[idx], fitness_data[idx], grid_size[0], reverse=False
            )
        top_individuals_per_gen.append((top_inds, top_fits))
    
    return fig, axs, top_individuals_per_gen, gen_indices


def plot_best_individuals_per_generation(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    title: Optional[str] = None
) -> Tuple[plt.Figure, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Plot the best individual from selected generations.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes, best_individuals, best_fitnesses)
    """
    # Generate grid layout
    grid = generate_best_per_generation_grid(
        population_data=population_data,
        fitness_data=fitness_data,
        grid_size=grid_size,
        twod=twod,
        include_motor_orientation=include_motor_orientation,
        worst=worst,
        title=title
    )
    
    # Create subplot from grid
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, title=title)
    
    # Get the actual best individuals data
    from .population_filtering import select_best_per_generation
    best_individuals, best_fitnesses = select_best_per_generation(
        population_data, fitness_data, worst=worst
    )
    
    return fig, axs, best_individuals, best_fitnesses


def plot_best_individuals_ever(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    grid_size: Tuple[int, int] = (4, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    worst: bool = False,
    title: Optional[str] = None,
    similarity_threshold: Optional[float] = None,
    round_to: int = 2
) -> Tuple[plt.Figure, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Plot the best individuals across all generations.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        worst: If True, show worst individuals
        title: Title for the plot
        similarity_threshold: If provided, ensure diversity in selection
        round_to: Number of decimal places for fitness display
        
    Returns:
        Tuple of (figure, axes, top_individuals, top_fitnesses)
    """
    # Generate grid layout
    grid, top_individuals, top_fitnesses = generate_best_ever_grid(
        population_data=population_data,
        fitness_data=fitness_data,
        grid_size=grid_size,
        twod=twod,
        include_motor_orientation=include_motor_orientation,
        worst=worst,
        title=title,
        similarity_threshold=similarity_threshold,
        round_to=round_to
    )
    
    # Create subplot from grid
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, title=title)
    
    return fig, axs, top_individuals, top_fitnesses


def plot_population(
    population: npt.NDArray,
    fitnesses: Optional[npt.NDArray] = None,
    twod: bool = True,
    title: Optional[str] = None,
    include_motor_orientation: Union[bool, int] = 0,
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, npt.NDArray]:
    """
    Plot an entire population in a grid layout.
    
    Args:
        population: Array of individuals
        fitnesses: Optional array of fitness values
        twod: If True, use 2D visualization
        title: Title for the plot
        include_motor_orientation: Whether to show motor orientations
        figsize: Optional figure size
        
    Returns:
        Tuple of (figure, axes)
    """
    # Generate grid layout
    grid = generate_population_grid(
        population=population,
        fitnesses=fitnesses,
        twod=twod,
        include_motor_orientation=include_motor_orientation
    )
    
    # Create subplot from grid
    adjust = twod
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, adjust=adjust)
    
    if title:
        fig.suptitle(title)
    
    if figsize:
        fig.set_size_inches(figsize)
    
    return fig, axs


def plot_evolution_comparison(
    population_lists: List[List[npt.NDArray]],
    fitness_lists: List[List[npt.NDArray]],
    labels: List[str],
    grid_size: Tuple[int, int] = (2, 4),
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    title: Optional[str] = None
) -> Tuple[plt.Figure, npt.NDArray]:
    """
    Plot a comparison of best individuals from multiple evolution runs.
    
    Args:
        population_lists: List of population data lists (one per run)
        fitness_lists: List of fitness data lists (one per run)
        labels: Labels for each evolution run
        grid_size: Tuple of (rows, cols) for grid layout
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    # Generate grid layout
    grid = generate_comparison_grid(
        population_lists=population_lists,
        fitness_lists=fitness_lists,
        labels=labels,
        grid_size=grid_size,
        twod=twod,
        include_motor_orientation=include_motor_orientation
    )
    
    # Create subplot from grid
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, title=title)
    
    return fig, axs


def plot_fitness_progression_with_individuals(
    population_data: List[npt.NDArray],
    fitness_data: List[npt.NDArray],
    num_individuals: int = 5,
    twod: bool = True,
    include_motor_orientation: Union[bool, int] = 0,
    title: Optional[str] = None
) -> Tuple[plt.Figure, npt.NDArray]:
    """
    Plot fitness progression alongside visualizations of top individuals.
    
    Args:
        population_data: List of population arrays for each generation
        fitness_data: List of fitness arrays for each generation
        num_individuals: Number of top individuals to show
        twod: If True, use 2D visualization
        include_motor_orientation: Whether to show motor orientations
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes)
    """
    from ariel.ec.drone.inspection.plot_fitness import plot_fitness
    import functools
    
    # Get the overall best individuals
    from .population_filtering import flatten_population_across_generations, get_top_k_individuals
    
    flat_pop, flat_fit = flatten_population_across_generations(
        np.array(population_data), np.array(fitness_data), remove_nan=True
    )
    
    top_individuals, top_fitnesses, _ = get_top_k_individuals(
        flat_pop, flat_fit, num_individuals, reverse=False
    )
    
    # Create a custom grid: fitness plot on left, individuals on right
    grid = []
    
    # First row: fitness plot spanning multiple cells
    fitness_func = functools.partial(
        plot_fitness,
        population_data=population_data,
        fitness_data=fitness_data
    )
    
    # Create layout: fitness plot + individual visualizations
    if num_individuals <= 4:
        # Single row layout
        grid_layout = [[fitness_func]]
        
        for i, (individual, fitness) in enumerate(zip(top_individuals, top_fitnesses)):
            visualizer = DroneVisualizer()
            if twod:
                viz_func = functools.partial(
                    lambda ax, ind=individual, fit=fitness: 
                    visualizer.plot_2d(
                        genome_data=ind, title=f"Fitness: {np.round(fit, 2)}"
                    )
                )
            else:
                viz_func = functools.partial(
                    lambda ax, ind=individual, fit=fitness:
                    visualizer.plot_3d(
                        genome_data=ind, title=f"Fitness: {np.round(fit, 2)}"
                    )
                )
            grid_layout[0].append(viz_func)
        
        grid = grid_layout
    else:
        # Multi-row layout
        grid.append([fitness_func])
        
        # Add individuals in subsequent rows
        individuals_per_row = min(4, num_individuals)
        for i in range(0, num_individuals, individuals_per_row):
            row = []
            for j in range(individuals_per_row):
                idx = i + j
                if idx < len(top_individuals):
                    individual = top_individuals[idx]
                    fitness = top_fitnesses[idx]
                    
                    visualizer = DroneVisualizer()
                    if twod:
                        viz_func = functools.partial(
                            visualizer.plot_2d,
                            genome_data=individual,
                            title=f"Fitness: {np.round(fitness, 2)}"
                        )
                    else:
                        viz_func = functools.partial(
                            visualizer.plot_3d,
                            genome_data=individual,
                            title=f"Fitness: {np.round(fitness, 2)}"
                        )
                    
                    row.append(viz_func)
            
            if row:  # Only add non-empty rows
                grid.append(row)
    
    # Create subplot from grid
    fig, axs, img_plots, texts = create_subplot(grid, twod=twod, title=title)
    
    return fig, axs


def create_evolution_summary_plot(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (16, 12),
    parameter_limits: Optional[Tuple[npt.NDArray, npt.NDArray]] = None
) -> plt.Figure:
    """
    Create a comprehensive summary plot of the evolutionary run.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        title: Title for the overall plot
        figsize: Figure size
        parameter_limits: Optional tuple of (min_vals, max_vals) for diversity calculation
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create a custom layout with multiple subplots
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Fitness progression plot
    ax_fitness = fig.add_subplot(gs[0, :2])
    from ariel.ec.drone.inspection.plot_fitness import plot_fitness
    plot_fitness(ax_fitness, population_data, fitness_data)
    ax_fitness.set_title("Fitness Progression")
    
    # 2. Diversity plot
    ax_diversity = fig.add_subplot(gs[0, 2:])
    from .plot_diversity import plot_diversity_from_amalgamated
    
    # Convert population data to format expected by diversity function
    population_list = [population_data[gen] for gen in range(len(population_data))]
    
    # Set default parameter limits if not provided
    if parameter_limits is None:
        min_vals = np.array([0.09, 0, 0, 0, 0, 0])
        max_vals = np.array([0.4, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1])
        param_bounds = np.column_stack([min_vals, max_vals])
    else:
        min_vals, max_vals = parameter_limits
        param_bounds = np.column_stack([min_vals, max_vals])
    
    # Plot diversity for single run (wrapped as list for consistency)
    plot_diversity_from_amalgamated(ax_diversity, [population_list], 
                                   title='Population Diversity', 
                                   min_max_params=param_bounds)
    
    # 3. Top individuals ever
    ax_top = fig.add_subplot(gs[1:, :2])
    top_grid, top_inds, top_fits = generate_best_ever_grid(
        population_data, fitness_data, grid_size=(3, 2), twod=True
    )
    ax_top.set_title("Top Individuals Overall")
    ax_top.axis('off')
    
    # 4. Fitness statistics
    ax_stats = fig.add_subplot(gs[1:, 2:])
    from .population_filtering import get_generation_fitness_statistics
    stats = get_generation_fitness_statistics(fitness_data)
    
    generations = np.arange(len(fitness_data))
    ax_stats.plot(generations, stats['mean'], label='Mean', linewidth=2)
    ax_stats.fill_between(
        generations, 
        stats['mean'] - stats['std'], 
        stats['mean'] + stats['std'], 
        alpha=0.3, label='±1 Std'
    )
    ax_stats.plot(generations, stats['max'], label='Max', linestyle='--')
    ax_stats.plot(generations, stats['min'], label='Min', linestyle='--')
    
    ax_stats.set_xlabel("Generation")
    ax_stats.set_ylabel("Fitness")
    ax_stats.set_title("Fitness Statistics")
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    return fig


def save_evolution_plots(
    population_data: npt.NDArray,
    fitness_data: npt.NDArray,
    output_dir: str,
    prefix: str = "evolution",
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300
) -> List[str]:
    """
    Save multiple evolution visualization plots to files.
    
    Args:
        population_data: Array of shape (n_generations, pop_size, n_arms, n_params)
        fitness_data: Array of shape (n_generations, pop_size)
        output_dir: Directory to save plots
        prefix: Prefix for filenames
        formats: List of file formats to save
        dpi: Resolution for raster formats
        
    Returns:
        List of saved file paths
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    
    # 1. Best individuals ever
    fig1, _, top_inds, top_fits = plot_best_individuals_ever(
        population_data, fitness_data, title="Best Individuals Ever"
    )
    
    for fmt in formats:
        filename = f"{prefix}_best_ever.{fmt}"
        filepath = os.path.join(output_dir, filename)
        fig1.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved_files.append(filepath)
    
    plt.close(fig1)
    
    # 2. Best per generation
    fig2, _, best_inds, best_fits = plot_best_individuals_per_generation(
        population_data, fitness_data, title="Best Per Generation"
    )
    
    for fmt in formats:
        filename = f"{prefix}_best_per_gen.{fmt}"
        filepath = os.path.join(output_dir, filename)
        fig2.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved_files.append(filepath)
    
    plt.close(fig2)
    
    # 3. Evolution summary
    fig3 = create_evolution_summary_plot(
        population_data, fitness_data, title="Evolution Summary"
    )
    
    for fmt in formats:
        filename = f"{prefix}_summary.{fmt}"
        filepath = os.path.join(output_dir, filename)
        fig3.savefig(filepath, dpi=dpi, bbox_inches='tight')
        saved_files.append(filepath)
    
    plt.close(fig3)
    
    return saved_files


# Backward compatibility aliases
plot_top_individuals = plot_top_individuals_across_generations
plot_best_individuals = plot_best_individuals_per_generation