"""
Cartesian symmetry operator for drone genome handlers.

This module implements bilateral symmetry operations for Cartesian coordinate
representations with Euler angles.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt

from .symmetry_base import SymmetryOperator, SymmetryConfig, SymmetryPlane, SymmetryUtilities


class CartesianSymmetryOperator(SymmetryOperator):
    """
    Symmetry operator for Cartesian coordinate genomes with Euler angles.
    
    Genome format expected: (max_narms, 7) with columns:
    [x, y, z, roll, pitch, yaw, direction]
    
    Uses NaN masking for variable arm counts.
    """
    
    def __init__(
        self,
        config: Optional[SymmetryConfig] = None
    ):
        """
        Initialize the Cartesian symmetry operator.
        
        Args:
            config: Symmetry configuration
        """
        super().__init__(config)
    
    def apply_symmetry(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply bilateral symmetry to a Cartesian genome.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            Genome with bilateral symmetry applied
        """
        if not self.config.enabled:
            return genome.copy()
        
        # Validate input shape
        if genome.ndim != 2 or genome.shape[1] != 7:
            raise ValueError(f"Expected genome shape (max_narms, 7), got {genome.shape}")
        
        # Find valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return genome.copy()
        max_narms = genome.shape[0]


        valid_arms = genome[valid_arms_mask]
        num_valid_arms = len(valid_arms)
        
        # Create result genome
        result = np.full_like(genome, np.nan)
        half_arms = max_narms // 2
        
        # Place all valid arms in the result first
        result[:num_valid_arms] = valid_arms
        # Mirror the first half to the second half
        for i in range(half_arms):
            source_idx = i
            target_idx = half_arms + i
            
            # Copy all parameters from source to target
            result[target_idx] = result[source_idx].copy()
            
            # Apply symmetry transformations based on the plane
            if self.config.plane == SymmetryPlane.XY:
                # Mirror across z=0 plane
                result[target_idx, 2] = -result[source_idx, 2]  # z position
                result[target_idx, 3] = -result[source_idx, 3]  #  (around y-axis)
                result[target_idx, 4] = -result[source_idx, 4]  #  (around z-axis)
                # yaw remains the same
                
            elif self.config.plane == SymmetryPlane.XZ:
                # Mirror across y=0 plane
                result[target_idx, 1] = -result[source_idx, 1]  # y position
                result[target_idx, 3] = -result[source_idx, 3]  #  (around x-axis)
                result[target_idx, 5] = -result[source_idx, 5]  #  (around z-axis)
                # pitch remains the same
                
            elif self.config.plane == SymmetryPlane.YZ:
                # Mirror across x=0 plane, 34, 35, 45
                result[target_idx, 0] = -result[source_idx, 0]  # x position
                result[target_idx, 4] = -result[source_idx, 4]  #  (around x-axis)
                result[target_idx, 5] = -result[source_idx, 5]  #  (around y-axis)
                # roll remains the same
            
            # For bilateral symmetry, propeller directions should be mirrored
            # to maintain rotational balance
            result[target_idx, 6] = 1 - result[source_idx, 6]
        
        return result
    
    def validate_symmetry(self, genome: npt.NDArray[Any]) -> bool:
        """
        Validate that a genome satisfies bilateral symmetry constraints.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            True if genome satisfies symmetry constraints
        """
        if not self.config.enabled:
            return True
        
        # Validate input
        if genome.shape[1] != 7:
            return False
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return True  # Empty genome is trivially symmetric
        
        valid_arms = genome[valid_arms_mask]
        num_valid_arms = len(valid_arms)
        
        # Check even number of arms
        if num_valid_arms % 2 != 0:
            return False
        
        half_arms = num_valid_arms // 2
        tolerance = self.config.tolerance
        
        for i in range(half_arms):
            source_arm = valid_arms[i]
            target_arm = valid_arms[half_arms + i]
            
            # Check symmetry based on the plane
            if self.config.plane == SymmetryPlane.XY:
                # Check z position mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[2], source_arm[2], tolerance, should_be_equal=False
                ):
                    return False
                # Check roll and pitch mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[3], source_arm[3], tolerance, should_be_equal=False
                ):
                    return False
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[4], source_arm[4], tolerance, should_be_equal=False
                ):
                    return False
                # Check yaw remains same
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[5], source_arm[5], tolerance, should_be_equal=True
                ):
                    return False
                    
            elif self.config.plane == SymmetryPlane.XZ:
                # Check y position mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[1], source_arm[1], tolerance, should_be_equal=False
                ):
                    return False
                # Check roll and yaw mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[3], source_arm[3], tolerance, should_be_equal=False
                ):
                    return False
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[5], source_arm[5], tolerance, should_be_equal=False
                ):
                    return False
                # Check pitch remains same
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[4], source_arm[4], tolerance, should_be_equal=True
                ):
                    return False
                    
            elif self.config.plane == SymmetryPlane.YZ:
                # Check x position mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[0], source_arm[0], tolerance, should_be_equal=False
                ):
                    return False
                # Check pitch and yaw mirroring
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[4], source_arm[4], tolerance, should_be_equal=False
                ):
                    return False
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[5], source_arm[5], tolerance, should_be_equal=False
                ):
                    return False
                # Check roll remains same
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[3], source_arm[3], tolerance, should_be_equal=True
                ):
                    return False
            
            # Check coordinates that should be same for all planes
            for coord_idx in [0, 1, 2]:
                if coord_idx == 0 and self.config.plane == SymmetryPlane.YZ:
                    continue  # x is mirrored for yz plane
                if coord_idx == 1 and self.config.plane == SymmetryPlane.XZ:
                    continue  # y is mirrored for xz plane
                if coord_idx == 2 and self.config.plane == SymmetryPlane.XY:
                    continue  # z is mirrored for xy plane
                
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[coord_idx], source_arm[coord_idx], 
                    tolerance, should_be_equal=True
                ):
                    return False
            
            # Check propeller direction mirroring
            if target_arm[6] != (1 - source_arm[6]):
                return False
        
        return True
    
    def get_symmetry_pairs(self, genome: npt.NDArray[Any]) -> List[tuple]:
        """
        Get pairs of indices that should be symmetric.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            List of (source_index, target_index) tuples
        """
        if not self.config.enabled:
            return []
        
        # Find valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return []  # No valid arms
        
        valid_indices = np.where(valid_arms_mask)[0]
        num_valid_arms = len(valid_indices)
        
        # Check even number of valid arms
        if num_valid_arms % 2 != 0:
            return []  # Can't have pairs with odd number of arms
        
        half_arms = num_valid_arms // 2
        pairs = []
        
        for i in range(half_arms):
            source_idx = valid_indices[i]
            target_idx = valid_indices[half_arms + i]
            pairs.append((source_idx, target_idx))
        
        return pairs
    
    def get_mirrored_coordinates(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Get coordinates with mirroring applied (without full symmetry).
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            Coordinates with mirroring applied
        """
        if not self.config.enabled:
            return genome.copy()
        
        result = genome.copy()
        
        # Only mirror valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        
        if self.config.plane == SymmetryPlane.XY:
            result[valid_arms_mask, 2] = -result[valid_arms_mask, 2]  # Mirror z
        elif self.config.plane == SymmetryPlane.XZ:
            result[valid_arms_mask, 1] = -result[valid_arms_mask, 1]  # Mirror y
        elif self.config.plane == SymmetryPlane.YZ:
            result[valid_arms_mask, 0] = -result[valid_arms_mask, 0]  # Mirror x
        
        return result
    
    def get_mirrored_orientations(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Get orientations with mirroring applied (without full symmetry).
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            Orientations with mirroring applied
        """
        if not self.config.enabled:
            return genome.copy()
        
        result = genome.copy()
        
        # Only mirror valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        
        if self.config.plane == SymmetryPlane.XY:
            result[valid_arms_mask, 3] = -result[valid_arms_mask, 3]  # Mirror roll
            result[valid_arms_mask, 4] = -result[valid_arms_mask, 4]  # Mirror pitch
        elif self.config.plane == SymmetryPlane.XZ:
            result[valid_arms_mask, 3] = -result[valid_arms_mask, 3]  # Mirror roll
            result[valid_arms_mask, 5] = -result[valid_arms_mask, 5]  # Mirror yaw
        elif self.config.plane == SymmetryPlane.YZ:
            result[valid_arms_mask, 4] = -result[valid_arms_mask, 4]  # Mirror pitch
            result[valid_arms_mask, 5] = -result[valid_arms_mask, 5]  # Mirror yaw
        
        return result
    
    
    def unapply_symmetry(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Remove symmetry from a genome (keep only first half).
        
        Args:
            genome: Symmetric genome array
            
        Returns:
            Genome with only first half of symmetric arms
        """
        if not self.config.enabled:
            raise ValueError("Symmetry removal is not enabled in the configuration")
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return genome.copy()
        
        valid_indices = np.where(valid_arms_mask)[0]
        num_valid_arms = len(valid_indices)
        
        half_arms = num_valid_arms // 2
        
        # Create result with only first half
        result = genome.copy()
        result[half_arms:] = np.nan  # Set second half to NaN
        
        return result
    
    def unapply_symmetry_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Remove symmetry from a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 7)
            
        Returns:
            Population with symmetry removed
        """
        if not self.config.enabled:
            return population.copy()
        
        result = np.empty_like(population)
        for i in range(population.shape[0]):
            result[i] = self.unapply_symmetry(population[i])
        
        return result
    
    def __repr__(self) -> str:
        return f"CartesianSymmetryOperator(config={self.config})"