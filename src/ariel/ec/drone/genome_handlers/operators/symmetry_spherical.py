"""
Spherical symmetry operator for drone genome handlers.

This module implements bilateral symmetry operations for spherical coordinate
representations with angular orientations.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt

from .symmetry_base import SymmetryOperator, SymmetryConfig, SymmetryPlane, SymmetryUtilities
from scipy.spatial.transform import Rotation as R

class SphericalSymmetryOperator(SymmetryOperator):
    """
    Symmetry operator for spherical coordinate genomes with angular orientations.
    
    Genome format expected: (max_narms, 6) with columns:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    
    Uses NaN masking for variable arm counts.
    """
    
    def __init__(
        self,
        config: Optional[SymmetryConfig] = None,
        coordinate_conversion_functions: Optional[dict] = None
    ):
        """
        Initialize the Spherical symmetry operator.
        
        Args:
            config: Symmetry configuration
            coordinate_conversion_functions: Optional dict with conversion functions
        """
        super().__init__(config)

        # Import coordinate conversion functions
        if coordinate_conversion_functions is None:
            from ..conversions.arm_conversions import (
                spherical_angular_arms_to_cartesian_positions_and_quaternions,
                cartesian_positions_and_quaternions_to_spherical_angular_arms
            )
            self.to_cartesian = spherical_angular_arms_to_cartesian_positions_and_quaternions
            self.from_cartesian = cartesian_positions_and_quaternions_to_spherical_angular_arms

        else:
            self.to_cartesian = coordinate_conversion_functions.get('to_cartesian')
            self.from_cartesian = coordinate_conversion_functions.get('from_cartesian')
    
    def apply_symmetry(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply bilateral symmetry to a spherical genome.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            Genome with bilateral symmetry applied
        """
        if not self.config.enabled:
            return genome.copy()
        
        # Validate input
        if genome.shape[1] != 6:
            raise ValueError("Spherical genome must have 6 parameters per arm")

        # Find valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return genome.copy()
        max_narms = genome.shape[0]
        narm_valid = np.sum(valid_arms_mask)
        if 2*narm_valid > max_narms:
            raise ValueError(
                f"Genome has {narm_valid} valid arms, but cannot create symmetry with max_narms={max_narms}"
            )
        valid_arms = genome[valid_arms_mask]
        num_valid_arms = len(valid_arms)
        
        # Create result genome
        result = np.full_like(genome, np.nan)
        result[:num_valid_arms] = valid_arms
        
        # Apply symmetry transformation
        if self.config.plane == SymmetryPlane.XY:
            # Mirror across z=0 plane
            target_arms = self._mirror_across_xy_plane(valid_arms)
        elif self.config.plane == SymmetryPlane.XZ:
            # Mirror across y=0 plane
            target_arms = self._mirror_across_xz_plane(valid_arms)
        elif self.config.plane == SymmetryPlane.YZ:
            # Mirror across x=0 plane
            target_arms = self._mirror_across_yz_plane(valid_arms)
        else:
            # No specific plane, use general mirroring
            raise ValueError(f"Unsupported symmetry plane: {self.config.plane}")
        
        result[num_valid_arms:2*num_valid_arms] = target_arms

        return result
    
    def _mirror_across_xy_plane(self, arms: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Mirror arms across the xy plane (z=0)."""
        # Convert to Cartesian for easier mirroring
        positions, quaternions = self.to_cartesian(arms[:, :5])

        # Mirror positions across z=0
        positions[:, 2] = -positions[:, 2]
        
        # Mirror quaternions appropriately for z-axis reflection, #TODO: Visually check this is correct
        quaternions[:, 1] = -quaternions[:, 1]  # x component
        quaternions[:, 2] = -quaternions[:, 2]  # y component
        
        # Convert back to spherical
        mirrored_arms = self.from_cartesian(positions, quaternions)
        
        # Combine with direction (flipped)
        result = np.zeros((len(arms), 6))
        result[:, :5] = mirrored_arms
        result[:, 5] = 1 - arms[:, 5]  # Flip direction
        
        return result
        
    
    def _mirror_across_xz_plane(self, arms: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Mirror arms across the xz plane (y=0)."""
        positions, quaternions = self.to_cartesian(arms[:, :5])
        
        # Mirror positions across y=0
        positions[:, 1] = -positions[:, 1]
        
        # Mirror quaternions appropriately for y-axis reflection, #TODO: Visually check this is correct
        quaternions[:, 1] = -quaternions[:, 1]  # w component
        quaternions[:, 3] = -quaternions[:, 3]  # z component
        
        # Convert back to spherical
        mirrored_arms = self.from_cartesian(positions, quaternions)
        
        # Combine with direction (flipped)
        result = np.zeros((len(arms), 6))
        result[:, :5] = mirrored_arms
        result[:, 5] = 1 - arms[:, 5]  # Flip direction
        
        return result
    
    def _mirror_across_yz_plane(self, arms: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Mirror arms across the yz plane (x=0)."""
        positions, quaternions = self.to_cartesian(arms[:, :5])
        
        # Mirror positions across x=0
        positions[:, 0] = -positions[:, 0]
        
        # Mirror quaternions appropriately for x-axis reflection, 
        quaternions[:, 2] = -quaternions[:, 2]  # y component
        quaternions[:, 3] = -quaternions[:, 3]  # z component

        # Convert back to spherical
        mirrored_arms = self.from_cartesian(positions, quaternions)
        
        # Combine with direction (flipped)
        result = np.zeros((len(arms), 6))
        result[:, :5] = mirrored_arms
        result[:, 5] = 1 - arms[:, 5]  # Flip direction
        
        return result
    
    def _mirror_arms_direct(self, arms: npt.NDArray[Any], mirror_axis: int) -> npt.NDArray[Any]:
        """Direct angular mirroring for spherical coordinates."""
        
        if mirror_axis == "xy":
            result = self._mirror_across_xy_plane(arms)
        elif mirror_axis == "xz":
            result = self._mirror_across_xz_plane(arms)
        elif mirror_axis == "yz":
            result = self._mirror_across_yz_plane(arms)
        else:
            raise ValueError(f"Unsupported symmetry plane: {mirror_axis}")

        return result
    
    def validate_symmetry(self, genome: npt.NDArray[Any]) -> bool:
        """
        Validate that a genome satisfies bilateral symmetry constraints.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            True if genome satisfies symmetry constraints
        """
        if not self.config.enabled:
            return True
        
        # Validate input
        if genome.shape[1] != 6:
            # print("Invalid genome shape for symmetry validation. Expected (max_narms, 6).")
            return False
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        if not np.any(valid_arms_mask):
            return True  # Empty genome is trivially symmetric
        
        valid_arms = genome[valid_arms_mask]
        num_valid_arms = len(valid_arms)
        
        # Check even number of arms
        if num_valid_arms % 2 != 0:
            # print(f"Invalid genome for symmetry validation: {num_valid_arms} valid arms found, expected even number.")
            return False
        
        half_arms = num_valid_arms // 2
        tolerance = self.config.tolerance
        
        # Check symmetry by comparing first half with second half
        for i in range(half_arms):
            source_arm = valid_arms[i]
            target_arm = valid_arms[half_arms + i]
            
            # Check if target arm is the symmetric version of source arm
            expected_target = self._mirror_arms_direct(np.array([source_arm]), self.config.plane.value)[0]
            
            # Compare with tolerance
            for j in range(5):  # Don't check direction yet
                if not SymmetryUtilities.check_value_symmetry(
                    target_arm[j], expected_target[j], tolerance, should_be_equal=True
                ):
                    # print(f"Validation failed for arm {i}: \n{source_arm} vs \n{target_arm}")
                    # print(f"Expected: {expected_target}")
                    return False
            
            # Check direction flipping
            if target_arm[5] != (1 - source_arm[5]):
                # print(f"Direction mismatch for arm {i}: {source_arm[5]} vs {target_arm[5]}")
                return False
        
        return True

    
    def get_symmetry_pairs(self, genome: npt.NDArray[Any]) -> List[tuple]:
        """
        Get pairs of indices that should be symmetric.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            List of (source_index, target_index) tuples
        """
        if not self.config.enabled:
            return []
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        valid_indices = np.where(valid_arms_mask)[0]
        num_valid_arms = len(valid_indices)
        
        if num_valid_arms % 2 != 0:
            return []  # Can't have pairs with odd number of arms
        
        half_arms = num_valid_arms // 2
        pairs = []
        
        for i in range(half_arms):
            source_idx = valid_indices[i]
            target_idx = valid_indices[half_arms + i]
            pairs.append((source_idx, target_idx))
        
        return pairs
    
    def apply_symmetry_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply symmetry to a population of spherical genomes.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 6)
            
        Returns:
            Population with symmetry applied
        """
        if not self.config.enabled:
            return population.copy()
        
        result = np.empty_like(population)
        for i in range(population.shape[0]):
            result[i] = self.apply_symmetry(population[i])
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
        
        if num_valid_arms % 2 != 0:
            raise ValueError("Genome must have an even number of valid arms for symmetry removal. Found: ", num_valid_arms)
        
        half_arms = num_valid_arms // 2
        
        # Create result with only first half
        result = genome.copy()
        result[half_arms:] = np.nan  # Set second half to NaN
        
        return result
    
    def unapply_symmetry_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Remove symmetry from a population of genomes.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 6)
            
        Returns:
            Population with symmetry removed
        """
        if not self.config.enabled:
            return population.copy()
        
        result = np.empty_like(population)
        for i in range(population.shape[0]):
            result[i] = self.unapply_symmetry(population[i])
        
        return result
    
    def is_enabled(self) -> bool:
        """Check if symmetry is enabled."""
        return self.config.enabled
    
    def __repr__(self) -> str:
        return f"SphericalSymmetryOperator(config={self.config})"