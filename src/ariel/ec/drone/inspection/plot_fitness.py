
"""
Fitness plotting utilities for evolutionary runs.

This module provides basic fitness visualization functionality for evolutionary
algorithms. It creates statistical plots showing mean, max, min, and standard
deviation of fitness over generations.

# TODO: [HIGH] Add comprehensive docstrings with parameter descriptions
# TODO: [HIGH] Add proper type hints for all functions  
# TODO: [MEDIUM] Standardize parameter naming (fitness_data vs population_data)
# TODO: [MEDIUM] Add input validation and error handling
# TODO: [MEDIUM] Expand functionality to match other visualization modules
# TODO: [LOW] Add more plot styling options and customization
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union
import numpy.typing as npt

def plot_fitness(
    ax, 
    fitness_data: npt.NDArray, 
    gen_line: Optional[int] = None, 
    log: bool = False, 
    pltmax: bool = True, 
    label: Optional[str] = None, 
    color: Optional[str] = None, 
    legend: bool = True
) -> None:
    """
    Plot fitness evolution statistics over generations.
    
    Args:
        ax: Matplotlib axes to plot on
        fitness_data: Array of shape (n_generations, pop_size) with fitness values
        gen_line: Optional generation to highlight with vertical line
        log: Whether to apply log transformation to fitness values
        pltmax: Whether to plot max and min fitness lines
        label: Optional label prefix for legend entries
        color: Optional color for plot elements
        legend: Whether to show legend
        
    # TODO: [HIGH] Add proper error handling for invalid inputs
    # TODO: [MEDIUM] Add parameter validation (check array shapes, etc.)
    # TODO: [MEDIUM] Consider returning plot elements for further customization
    """
    # TODO: [HIGH] Replace in-place modification with copy to avoid side effects
    # ignore infinities and nans
    fitness_data[fitness_data == -np.inf] = np.nan
    fitness_data[fitness_data == np.inf] = np.nan
    if log:
        # TODO: [MEDIUM] Add validation that fitness values are negative for log transform
        fitness_data = -np.log(-fitness_data)

    # Calculate statistics across population for each generation
    means = np.nanmean(fitness_data, axis=1)
    stds = np.nanstd(fitness_data, axis=1)
    maxs = np.nanmax(fitness_data, axis=1)  # TODO: [LOW] Rename to max_values for clarity
    mins = np.nanmin(fitness_data, axis=1)  # TODO: [LOW] Rename to min_values for clarity
    
    generations = np.arange(len(fitness_data))
    
    # TODO: [MEDIUM] Use f-strings for cleaner string formatting
    label1 = label+" Mean Fitness" if label != None else 'Mean Fitness'
    label2 = label+" Max Fitness" if label != None else 'Max Fitness'
    label3 = label+" Min Fitness" if label != None else 'Min Fitness'
    ax.plot(generations, means, label=label1, color=color)
    if pltmax:
        ax.plot(generations, maxs, linestyle='--', label=label2, color=color)
        ax.plot(generations, mins, linestyle='--', label=label3, color=color)
    ax.fill_between(generations, means-stds, means+stds, alpha=0.2, color=color)

    # Add vertical line to highlight specific generation if requested
    if gen_line != None:  # TODO: [LOW] Use 'is not None' instead of '!= None'
        the_max = np.max(maxs)
        the_min = np.min(mins)  # Fixed: was np.max(mins) which was incorrect
        ax.plot([gen_line, gen_line], [the_min, the_max], color="black")

    # Set up axes labels and styling
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    if legend:
        ax.legend()
    ax.grid()
    
    # TODO: [MEDIUM] Make axis limits configurable or computed more intelligently
    # ax.set_xlim([np.min(means-stds), np.max(maxs)])  # TODO: [LOW] Remove commented code
    mx = np.nanmax(maxs) * 1.1 if np.nanmax(maxs) > 0 else np.nanmax(maxs) * 0.9
    ax.set_ylim([np.nanmin(means), mx])

