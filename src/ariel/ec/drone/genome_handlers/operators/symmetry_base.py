"""
Base classes and interfaces for symmetry operators.

This module provides the abstract base class and common utilities for implementing
symmetry operations across different genome representations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
from enum import Enum

import numpy as np
import numpy.typing as npt


class SymmetryPlane(Enum):
    """Enumeration of supported symmetry planes."""
    XY = "xy"  # Mirror across z=0 plane
    XZ = "xz"  # Mirror across y=0 plane
    YZ = "yz"  # Mirror across x=0 plane

class SymmetryConfig:
    """Configuration for symmetry operations."""
    
    def __init__(
        self,
        plane: Union[SymmetryPlane, str, None] = None,
        enabled: bool = True,
        tolerance: float = 1e-6
    ):
        """
        Initialize symmetry configuration.
        
        Args:
            plane: Symmetry plane (SymmetryPlane enum, string, or None)
            enabled: Whether symmetry is enabled
            tolerance: Tolerance for symmetry validation
        """
        if isinstance(plane, str):
            try:
                self.plane = SymmetryPlane(plane)
            except ValueError:
                raise ValueError(f"Invalid symmetry plane: {plane}")
        elif plane is None:
            self.plane = None
        else:
            self.plane = plane
            
        self.enabled = enabled and (self.plane != None)
        self.tolerance = tolerance
    
    def __repr__(self) -> str:
        return f"SymmetryConfig(plane={self.plane}, enabled={self.enabled}, tolerance={self.tolerance})"


class SymmetryOperator(ABC):
    """
    Abstract base class for symmetry operations on genomes.
    
    This class defines the interface that all symmetry operators must implement,
    providing a consistent API across different genome representations.
    """
    
    def __init__(self, config: Optional[SymmetryConfig] = None):
        """
        Initialize the symmetry operator.
        
        Args:
            config: Symmetry configuration. If None, creates default config.
        """
        self.config = config or SymmetryConfig()
    
    @abstractmethod
    def apply_symmetry(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply symmetry constraints to a genome.
        
        Args:
            genome: Input genome array
            
        Returns:
            Genome with symmetry applied
        """
        pass
    
    @abstractmethod
    def validate_symmetry(self, genome: npt.NDArray[Any]) -> bool:
        """
        Check if a genome satisfies symmetry constraints.
        
        Args:
            genome: Genome array to validate
            
        Returns:
            True if genome satisfies symmetry constraints, False otherwise
        """
        pass
    
    @abstractmethod
    def get_symmetry_pairs(self, genome: npt.NDArray[Any]) -> List[tuple]:
        """
        Get pairs of indices that should be symmetric.
        
        Args:
            genome: Genome array
            
        Returns:
            List of (source_index, target_index) tuples
        """
        pass
    
    def apply_symmetry_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply symmetry to a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, ...)
            
        Returns:
            Population with symmetry applied
        """
        if not self.config.enabled:
            return population
            
        result = np.empty_like(population)
        for i in range(population.shape[0]):
            result[i] = self.apply_symmetry(population[i])
        return result
    
    def validate_symmetry_population(self, population: npt.NDArray[Any]) -> npt.NDArray[bool]:
        """
        Validate symmetry for a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, ...)
            
        Returns:
            Boolean array indicating which individuals satisfy symmetry
        """
        if not self.config.enabled:
            return np.ones(population.shape[0], dtype=bool)
            
        result = np.empty(population.shape[0], dtype=bool)
        for i in range(population.shape[0]):
            result[i] = self.validate_symmetry(population[i])
        return result
    
    def is_enabled(self) -> bool:
        """Check if symmetry is enabled."""
        return self.config.enabled
    
    def get_plane(self) -> SymmetryPlane:
        """Get the symmetry plane."""
        return self.config.plane
    
    def get_tolerance(self) -> float:
        """Get the symmetry tolerance."""
        return self.config.tolerance
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.config})"


class SymmetryUtilities:
    """Common utilities for symmetry operations."""
    
    @staticmethod
    def flip_directions(directions: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Flip propeller directions for symmetry.
        
        Args:
            directions: Array of directions (0 or 1)
            
        Returns:
            Flipped directions (1 - original)
        """
        return 1 - directions
    
    @staticmethod
    def mirror_coordinate(
        coordinate: npt.NDArray[Any], 
        axis: int
    ) -> npt.NDArray[Any]:
        """
        Mirror coordinates across a specific axis.
        
        Args:
            coordinate: Coordinate array
            axis: Axis to mirror across (0=x, 1=y, 2=z)
            
        Returns:
            Mirrored coordinates
        """
        result = coordinate.copy()
        result[..., axis] = -result[..., axis]
        return result
    
    @staticmethod
    def mirror_angles(
        angles: npt.NDArray[Any], 
        axes: List[int]
    ) -> npt.NDArray[Any]:
        """
        Mirror angles for symmetry transformation.
        
        Args:
            angles: Array of angles
            axes: List of axes to mirror
            
        Returns:
            Mirrored angles
        """
        result = angles.copy()
        for axis in axes:
            result[..., axis] = -result[..., axis]
        return result
    
    @staticmethod
    def validate_even_arm_count(narms: int) -> None:
        """
        Validate that arm count is even (required for bilateral symmetry).
        
        Args:
            narms: Number of arms
            
        Raises:
            ValueError: If arm count is odd
        """
        if narms % 2 != 0:
            raise ValueError("Bilateral symmetry requires an even number of arms")
    
    @staticmethod
    def get_symmetric_pairs(narms: int) -> List[tuple]:
        """
        Get symmetric pairs for bilateral symmetry.
        
        Args:
            narms: Number of arms (must be even)
            
        Returns:
            List of (source, target) index pairs
        """
        SymmetryUtilities.validate_even_arm_count(narms)
        half_arms = narms // 2
        return [(i, half_arms + i) for i in range(half_arms)]
    
    @staticmethod
    def check_value_symmetry(
        value1: float, 
        value2: float, 
        tolerance: float = 1e-6, 
        should_be_equal: bool = True
    ) -> bool:
        """
        Check if two values satisfy symmetry constraints.
        
        Args:
            value1: First value
            value2: Second value
            tolerance: Tolerance for comparison
            should_be_equal: If True, values should be equal; if False, they should be negatives
            
        Returns:
            True if values satisfy symmetry constraint
        """
        if should_be_equal:
            return abs(value1 - value2) <= tolerance
        else:
            return abs(value1 + value2) <= tolerance