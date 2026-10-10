"""
Base classes and interfaces for repair operators.

This module provides the abstract base class and common utilities for implementing
repair operations across different genome representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List, Tuple
from enum import Enum

import numpy as np
import numpy.typing as npt


class RepairStrategy(Enum):
    """Enumeration of repair strategies."""
    CLIP = "clip"  # Clip values to bounds
    WRAP = "wrap"  # Wrap values around bounds
    RANDOM = "random"  # Replace with random values

class RepairConfig:
    """Configuration for repair operations."""
    
    def __init__(
        self,
        apply_symmetry: bool = True,
        enable_collision_repair: bool = True,
        propeller_radius: float = 0.0254,  # 2-inch propeller radius in meters
        inner_boundary_radius: float = 0.09,
        outer_boundary_radius: float = 0.4,
        max_repair_iterations: int = 50,
        repair_step_size: float = 1.0,
        propeller_tolerance: float = 0.1,
    ):
        """
        Initialize repair configuration.

        Args:
            apply_symmetry: Whether to apply symmetry after repair
            enable_collision_repair: Whether to enable collision detection and repair
            propeller_radius: Radius of propellers for collision detection (default: 0.0254m = 2-inch props)
            inner_boundary_radius: Minimum distance from origin
            outer_boundary_radius: Maximum distance from origin
            max_repair_iterations: Maximum iterations for collision repair
            repair_step_size: Step size multiplier for collision resolution
            propeller_tolerance: Additional clearance tolerance for propellers
            reflection_axis: Optional fixed axis direction for symmetric collision resolution
        """
        self.apply_symmetry = apply_symmetry
        self.enable_collision_repair = enable_collision_repair
        self.propeller_radius = propeller_radius
        self.inner_boundary_radius = inner_boundary_radius
        self.outer_boundary_radius = outer_boundary_radius
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size
        self.propeller_tolerance = propeller_tolerance

        assert self.inner_boundary_radius < self.outer_boundary_radius, \
            "Inner boundary radius must be less than outer boundary radius"
        assert self.inner_boundary_radius >= 0, \
            "Inner boundary radius must be non-negative"
        assert self.outer_boundary_radius > 0, \
            "Outer boundary radius must be positive"
        assert self.propeller_radius > 0, \
            "Propeller radius must be positive"
    
    def __repr__(self) -> str:
        collision_str = f", collision={self.enable_collision_repair}" if self.enable_collision_repair else ""
        return (f"RepairConfig(symmetry={self.apply_symmetry}{collision_str})")

class RepairOperator(ABC):
    """
    Abstract base class for repair operations on genomes.
    
    This class defines the interface that all repair operators must implement,
    providing a consistent API across different genome representations.
    """
    
    def __init__(self, config: Optional[RepairConfig] = None):
        """
        Initialize the repair operator.
        
        Args:
            config: Repair configuration. If None, creates default config.
        """
        self.config = config or RepairConfig()
    
    @abstractmethod
    def repair(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a genome to make it valid.
        
        Args:
            genome: Input genome array
            
        Returns:
            Repaired genome
        """
        pass
    
    @abstractmethod
    def validate(self, genome: npt.NDArray[Any]) -> bool:
        """
        Check if a genome is valid.
        
        Args:
            genome: Genome array to validate
            
        Returns:
            True if genome is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_bounds(self) -> npt.NDArray[Any]:
        """
        Get the parameter bounds for this genome type.
        
        Returns:
            Array of shape (n_params, 2) with [min, max] for each parameter
        """
        pass
    
    def repair_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, ...)
            
        Returns:
            Repaired population
        """
        result = np.empty_like(population)
        for i in range(population.shape[0]):
            result[i] = self.repair(population[i])
        return result
    
    def validate_population(self, population: npt.NDArray[Any]) -> npt.NDArray[bool]:
        """
        Validate a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, ...)
            
        Returns:
            Boolean array indicating which individuals are valid
        """
        result = np.empty(population.shape[0], dtype=bool)
        for i in range(population.shape[0]):
            result[i] = self.validate(population[i])
        return result
    
    def repair_collisions(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair collisions in a genome (optional, implemented by subclasses).
        
        This method provides a default implementation that returns the genome unchanged.
        Subclasses should override this method to implement collision repair specific
        to their coordinate system.
        
        Args:
            genome: Input genome array
            
        Returns:
            Genome with collisions repaired (default: unchanged)
        """
        return genome
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"


class RepairUtilities:
    """Common utilities for repair operations."""
    
    @staticmethod
    def clip_to_bounds(
        values: Union[npt.NDArray[Any], float], 
        bounds: Union[npt.NDArray[Any], Tuple[float, float]]
    ) -> Union[npt.NDArray[Any], float]:
        """
        Clip values to specified bounds.
        
        Args:
            values: Array of values to clip or single value
            bounds: Array of shape (n_params, 2) with [min, max] for each parameter,
                   or tuple (min, max) for single value
            
        Returns:
            Clipped values
        """
        # Handle scalar case
        if np.isscalar(values):
            if isinstance(bounds, tuple):
                return np.clip(values, bounds[0], bounds[1])
            else:
                raise ValueError("For scalar values, bounds must be a tuple (min, max)")
        
        # Handle array case
        values = np.asarray(values)
        result = values.copy()
        
        if isinstance(bounds, tuple):
            # Single bounds tuple - apply to all values
            return np.clip(values, bounds[0], bounds[1])
        else:
            # Array of bounds
            bounds = np.asarray(bounds)
            for i in range(bounds.shape[0]):
                if i < values.shape[-1]:  # Handle cases where values has fewer parameters
                    result[..., i] = np.clip(values[..., i], bounds[i, 0], bounds[i, 1])
            return result
    
    @staticmethod
    def wrap_to_bounds(
        values: Union[npt.NDArray[Any], float], 
        bounds: Union[npt.NDArray[Any], Tuple[float, float]]
    ) -> Union[npt.NDArray[Any], float]:
        """
        Wrap values to specified bounds (useful for angular parameters).
        
        Args:
            values: Array of values to wrap or single value
            bounds: Array of shape (n_params, 2) with [min, max] for each parameter,
                   or tuple (min, max) for single value
            
        Returns:
            Wrapped values
        """
        # Handle scalar case
        if np.isscalar(values):
            if isinstance(bounds, tuple):
                min_val, max_val = bounds
                range_val = max_val - min_val
                return ((values - min_val) % range_val) + min_val
            else:
                raise ValueError("For scalar values, bounds must be a tuple (min, max)")
        
        # Handle array case
        values = np.asarray(values)
        result = values.copy()
        
        if isinstance(bounds, tuple):
            # Single bounds tuple - apply to all values
            min_val, max_val = bounds
            range_val = max_val - min_val
            return ((values - min_val) % range_val) + min_val
        else:
            # Array of bounds
            bounds = np.asarray(bounds)
            for i in range(bounds.shape[0]):
                if i < values.shape[-1]:
                    min_val, max_val = bounds[i, 0], bounds[i, 1]
                    range_val = max_val - min_val
                    result[..., i] = ((values[..., i] - min_val) % range_val) + min_val
            return result
    
    @staticmethod
    def detect_invalid_values(values: npt.NDArray[Any]) -> npt.NDArray[bool]:
        """
        Detect NaN or infinite values.
        
        Args:
            values: Array to check
            
        Returns:
            Boolean mask indicating invalid values
        """
        return ~np.isfinite(values)
    
    @staticmethod
    def replace_invalid_values(
        values: npt.NDArray[Any], 
        replacement: Union[npt.NDArray[Any], float], 
        mask: Optional[npt.NDArray[bool]] = None
    ) -> npt.NDArray[Any]:
        """
        Replace invalid values with replacement values.
        
        Args:
            values: Array with potentially invalid values
            replacement: Replacement values (array or scalar)
            mask: Optional mask indicating which values to replace
            
        Returns:
            Array with invalid values replaced
        """
        if mask is None:
            mask = RepairUtilities.detect_invalid_values(values)
        
        result = values.copy()
        if np.isscalar(replacement):
            result[mask] = replacement
        else:
            replacement = np.asarray(replacement)
            result[mask] = replacement[mask]
        return result
    
    @staticmethod
    def generate_random_replacement(
        shape: tuple, 
        bounds: npt.NDArray[Any], 
        rng: Optional[np.random.Generator] = None
    ) -> npt.NDArray[Any]:
        """
        Generate random values within specified bounds.
        
        Args:
            shape: Shape of array to generate
            bounds: Array of shape (n_params, 2) with [min, max] for each parameter
            rng: Random number generator
            
        Returns:
            Random values within bounds
        """
        if rng is None:
            rng = np.random.default_rng()
        
        result = np.empty(shape)
        for i in range(bounds.shape[0]):
            if i < shape[-1]:
                result[..., i] = rng.uniform(bounds[i, 0], bounds[i, 1], size=shape[:-1])
        return result
    
    @staticmethod
    def enforce_discrete_values(
        values: npt.NDArray[Any], 
        discrete_indices: List[int], 
        discrete_values: List[List[Any]]
    ) -> npt.NDArray[Any]:
        """
        Enforce discrete values for specified parameters.
        
        Args:
            values: Array of values
            discrete_indices: Indices of parameters that should be discrete
            discrete_values: List of allowed values for each discrete parameter
            
        Returns:
            Array with discrete values enforced
        """
        result = values.copy()
        for i, allowed_values in zip(discrete_indices, discrete_values):
            if i < values.shape[-1]:
                # Find closest allowed value for each element
                for j, allowed in enumerate(allowed_values):
                    if j == 0:
                        distances = np.abs(values[..., i] - allowed)
                        closest = np.full_like(values[..., i], allowed)
                    else:
                        new_distances = np.abs(values[..., i] - allowed)
                        closer_mask = new_distances < distances
                        distances[closer_mask] = new_distances[closer_mask]
                        closest[closer_mask] = allowed
                result[..., i] = closest
        return result
    
    @staticmethod
    def enforce_binary_values(
        values: npt.NDArray[Any], 
        binary_indices: List[int], 
        threshold: float = 0.5
    ) -> npt.NDArray[Any]:
        """
        Enforce binary (0/1) values for specified parameters.
        
        Args:
            values: Array of values
            binary_indices: Indices of parameters that should be binary
            threshold: Threshold for binary conversion
            
        Returns:
            Array with binary values enforced
        """
        result = values.copy()
        for i in binary_indices:
            if i < values.shape[-1]:
                result[..., i] = np.where(values[..., i] >= threshold, 1, 0)
        return result
    
    @staticmethod
    def validate_bounds(
        values: npt.NDArray[Any], 
        bounds: npt.NDArray[Any], 
        tolerance: float = 1e-9
    ) -> npt.NDArray[bool]:
        """
        Validate that values are within specified bounds.
        
        Args:
            values: Array of values to validate
            bounds: Array of shape (n_params, 2) with [min, max] for each parameter
            tolerance: Tolerance for bound checking
            
        Returns:
            Boolean array indicating which values are within bounds
        """
        valid = np.ones(values.shape, dtype=bool)
        for i in range(bounds.shape[0]):
            if i < values.shape[-1]:
                valid[..., i] &= (values[..., i] >= bounds[i, 0] - tolerance)
                valid[..., i] &= (values[..., i] <= bounds[i, 1] + tolerance)
        return valid
    
    @staticmethod
    def validate_finite(values: npt.NDArray[Any]) -> npt.NDArray[bool]:
        """
        Validate that values are finite (not NaN or infinite).
        
        Args:
            values: Array of values to validate
            
        Returns:
            Boolean array indicating which values are finite
        """
        return np.isfinite(values)