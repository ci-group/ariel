"""
Subplot creation utilities for grid-based visualizations.

This module creates subplot layouts from 2D grids of plotting functions,
providing the foundation for evolutionary population visualizations.

# TODO: [HIGH] Add comprehensive error handling and input validation
# TODO: [HIGH] Improve function documentation and add type hints
# TODO: [MEDIUM] Refactor create_subplot function - it's overly complex
# TODO: [MEDIUM] Standardize parameter naming and conventions
# TODO: [MEDIUM] Add proper logging for debugging subplot creation issues
# TODO: [LOW] Consider splitting large functions into smaller utilities
# TODO: [LOW] Add more flexible subplot configuration options
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List, Tuple, Callable, Union, Any
import numpy.typing as npt

def create_subplot(
    grid: List[List[Callable]], 
    axs: Optional[Any] = None, 
    fig: Optional[plt.Figure] = None, 
    twod: bool = True, 
    adjust: bool = True, 
    figaspect: float = 0.5, 
    title: Optional[str] = None, 
    color: str = 'blue'  # color parameter appears unused
) -> Tuple[plt.Figure, Any, List, List]:
    """
    Create a subplot using a 2D grid of functions.

    Each function in the grid should take a single argument: the axes object to plot on.
    
    Args:
        grid: 2D list of plotting functions
        axs: Optional existing axes objects to use
        fig: Optional existing figure to use
        twod: Whether to create 2D or 3D subplots
        adjust: Whether to adjust subplot spacing
        figaspect: Aspect ratio for 3D plots
        title: Optional title for the entire figure
        color: Color parameter (currently unused)
        
    Returns:
        Tuple of (figure, axes, images_list, texts_list)
        
    Raises:
        ValueError: If grid is empty, not rectangular, or contains invalid functions
        TypeError: If grid contains non-callable elements
    """
    # Input validation
    if not grid:
        raise ValueError("Grid cannot be empty")
    
    # Validate grid structure
    rows = len(grid)
    if rows == 0:
        raise ValueError("Grid must have at least one row")
    
    if not grid[0]:
        raise ValueError("Grid rows cannot be empty")
        
    cols = len(grid[0])
    
    # Check that all rows have the same length (rectangular grid)
    for i, row in enumerate(grid):
        if len(row) != cols:
            raise ValueError(f"Grid must be rectangular. Row {i} has {len(row)} columns, expected {cols}")
        
        # Check that all elements are callable
        for j, func in enumerate(row):
            if not callable(func):
                raise TypeError(f"Grid element at [{i}][{j}] is not callable: {type(func)}")
    
    # Validate parameters
    if figaspect <= 0:
        raise ValueError("figaspect must be positive")
    
    # Get the dimensions of the grid
    rows = len(grid)
    cols = len(grid[0])

    # Create figure and axes objects if not provided
    # TODO: [MEDIUM] The axes handling logic is overly complex and error-prone
    if twod and axs is None:
        fig, axs = plt.subplots(rows, cols, figsize=(16,12))  # TODO: [MEDIUM] Make figsize configurable

        # TODO: [HIGH] This axes reshaping logic is confusing and fragile
        if type(axs) == np.ndarray:  # TODO: [LOW] Use isinstance() instead of type()
            if len(axs.shape) == 1:
                axs = np.array([axs])
    elif not twod and axs is None:
        fig = plt.figure(figsize=plt.figaspect(figaspect))

    
    # Adjust subplot spacing if requested
    if adjust and twod:
        plt.subplots_adjust(wspace=0, hspace=0)  # TODO: [MEDIUM] Make spacing configurable
    elif adjust and not twod:
        fig.subplots_adjust(wspace=0, hspace=0)

    # TODO: [MEDIUM] These lists are never populated - should either remove or implement
    imgs = []
    txts = []
    
    # Iterate over the grid and call each plotting function
    for i in range(rows):
        for j in range(cols):
            try:
                if twod:
                    if isinstance(axs, np.ndarray):
                        if len(axs.shape) == 1:
                            if j >= len(axs):
                                raise IndexError(f"Column index {j} out of range for axes array of length {len(axs)}")
                            grid[i][j](axs[j])
                        else:
                            if i >= axs.shape[0] or j >= axs.shape[1]:
                                raise IndexError(f"Grid position [{i}][{j}] out of range for axes shape {axs.shape}")
                            grid[i][j](axs[i, j])
                    else:
                        grid[i][j](axs)  # Single axes case
                else:
                    # Create 3D subplot for each grid position
                    axs = fig.add_subplot(rows, cols, i*cols+j+1, projection='3d')
                    grid[i][j](axs)
            except Exception as e:
                print(f"Warning: Failed to execute plotting function at grid position [{i}][{j}]: {e}")
                # Continue with next function rather than failing entirely
    
    # TODO: [MEDIUM] Handle case where title is None more gracefully
    fig.suptitle(title)
    
    # TODO: [HIGH] The imgs and txts lists are always empty - fix or document why
    return fig, axs, imgs, txts

def plot_img(
    ax: plt.Axes, 
    img: npt.NDArray, 
    fitness: Optional[float] = None, 
    generation: Optional[int] = None
) -> Tuple[Any, Any]:
    """
    Plot an image with optional fitness and generation text overlay.

    Args:
        ax: The axes object to plot on
        img: The image array to plot
        fitness: Optional fitness value to display
        generation: Optional generation number to display

    Returns:
        Tuple of (image_object, text_object)
        
    Raises:
        ValueError: If img is not a valid image array
        TypeError: If ax is not a matplotlib Axes object
    """
    # Input validation
    if ax is None:
        raise TypeError("Axes object cannot be None")
    
    if img is None:
        raise ValueError("Image array cannot be None")
        
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Image must be numpy array, got {type(img)}")
        
    if img.size == 0:
        raise ValueError("Image array cannot be empty")
    
    # Validate image dimensions (should be 2D or 3D)
    if img.ndim not in [2, 3]:
        raise ValueError(f"Image must be 2D or 3D array, got {img.ndim}D")
    
    try:
        im = ax.imshow(img, interpolation='nearest', animated=True)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Create text overlay with improved logic
        txt = None
        if generation is not None and fitness is not None:
            txt = ax.text(0.5, 0.1, f"Gen: {generation}, f: {fitness:.3f}", 
                         horizontalalignment='center', verticalalignment='center', 
                         transform=ax.transAxes, color='white')
        elif generation is not None:
            txt = ax.text(0.5, 0.1, f"Gen: {generation}", horizontalalignment='center', 
                         verticalalignment='center', transform=ax.transAxes, color='white')
        elif fitness is not None:
            txt = ax.text(0.5, 0.1, f"Fitness: {fitness:.3f}", horizontalalignment='center', 
                         verticalalignment='center', transform=ax.transAxes, color='white')
        
        return im, txt
    except Exception as e:
        raise RuntimeError(f"Failed to plot image: {e}") from e

def remove_ticks(ax: plt.Axes) -> Tuple[None, None]:
    """
    Remove ticks and spines from an axes object for clean visualization.

    Args:
        ax: The axes object to clean up
        
    Returns:
        Tuple of (None, None) for compatibility with other plotting functions
        
    # TODO: [LOW] Consider if return value is needed - seems like compatibility hack
    # TODO: [MEDIUM] Add option to selectively remove only certain spines/ticks
    """
    # Remove all spines (borders)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Remove tick marks and labels
    ax.set_xticks([])
    ax.set_yticks([])

    # TODO: [LOW] This return value seems arbitrary - document or remove
    return None, None