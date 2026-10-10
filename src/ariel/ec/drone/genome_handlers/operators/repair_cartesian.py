"""
Cartesian repair operator for drone genome handlers.

This module implements repair operations for Cartesian coordinate
representations with Euler angles.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .repair_base import RepairOperator, RepairConfig, RepairUtilities
from .symmetry_cartesian import CartesianSymmetryOperator
from ..conversions.arm_conversions import arms_to_cylinders_cartesian_euler, cylinders_to_arms_cartesian_euler
from .particle_repair_operator import particle_repair_individual, are_there_cylinder_collisions
from .symmetry_base import SymmetryPlane

class CartesianRepairOperator(RepairOperator):
    """
    Repair operator for Cartesian coordinate genomes with Euler angles.
    
    Genome format expected: (max_narms, 7) with columns:
    [x, y, z, roll, pitch, yaw, direction]
    
    Uses NaN masking for variable arm counts.
    """
    
    def __init__(
        self,
        config: Optional[RepairConfig] = None,
        parameter_limits: Optional[npt.NDArray[Any]] = None,
        min_narms: int = 1,
        max_narms: int = 8,
        symmetry_operator: Optional[CartesianSymmetryOperator] = None,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the Cartesian repair operator.
        
        Args:
            config: Repair configuration
            parameter_limits: Array of shape (7, 2) with [min, max] for each parameter
            min_narms: Minimum number of arms
            max_narms: Maximum number of arms
            symmetry_operator: Optional symmetry operator for symmetry restoration
            rng: Random number generator
        """
        super().__init__(config)
        self.min_narms = min_narms
        self.max_narms = max_narms
        self.symmetry_operator = symmetry_operator
        self.rng = rng if rng is not None else np.random.default_rng()
        
        # Set default parameter limits if not provided
        if parameter_limits is None:
            self.parameter_limits = np.array([
                [-1.0, 1.0],        # x position
                [-1.0, 1.0],        # y position
                [-1.0, 1.0],        # z position
                [-np.pi, np.pi],    # roll
                [-np.pi, np.pi],    # pitch
                [-np.pi, np.pi],    # yaw
                [0, 1]              # direction (binary)
            ])
        else:
            if parameter_limits.shape != (7, 2):
                raise ValueError("parameter_limits must have shape (7, 2)")
            self.parameter_limits = parameter_limits.copy()
    
    def repair(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a Cartesian genome to make it valid.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            Repaired genome
        """
        if genome.shape[1] != 7:
            raise ValueError("Cartesian genome must have 7 parameters per arm")
        narms = np.sum(~np.isnan(genome[:, 0]))
        if narms <= 1:
            return genome.copy()
        
        result = genome.copy()

        # Unapply symmetry if it was applied
        if self.config.apply_symmetry and self.symmetry_operator is not None:
            result = self.symmetry_operator.unapply_symmetry(result)
            if self.symmetry_operator.get_plane() is  SymmetryPlane.XY:
                axis = [0, 0, 1]
            elif self.symmetry_operator.get_plane() is  SymmetryPlane.XZ:
                axis = [0, 1, 0]
            elif self.symmetry_operator.get_plane() is SymmetryPlane.YZ:
                axis = [1, 0, 0]
        
        # Step 2: Clip parameters to bounds
        result = self._clip_parameters(result)
        
        # Step 3: Apply collision repair if enabled
        if self.config.enable_collision_repair:
            result = self.repair_collisions(result)
        
        # Check collsions
        cylinders = arms_to_cylinders_cartesian_euler(result)
        f = are_there_cylinder_collisions(cylinders)
        # Step 4: Apply symmetry restoration if enabled
        if self.config.apply_symmetry and self.symmetry_operator is not None:
            result = self.symmetry_operator.apply_symmetry(result)
            # Apply repair again but on reflected axis
            result = self.repair_collisions(result, repair_along_fixed_axis=axis)
        # Check collsions
        cylinders = arms_to_cylinders_cartesian_euler(result)
        f = are_there_cylinder_collisions(cylinders)
        return result
    
    def validate(self, genome: npt.NDArray[Any]) -> bool:
        """
        Check if a Cartesian genome is valid.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            True if genome is valid
        """
        # Check genome shape first
        if len(genome.shape) != 2 or genome.shape[1] != 7:
            print("Invalid genome shape, expected (max_narms, 7)")
            return False
        
        # Find valid arms (non-NaN) and check arm count constraints
        valid_arms_mask = ~np.isnan(genome[:, 0])
        num_valid_arms = np.sum(valid_arms_mask)
        max_narms = genome.shape[0]
        
        if num_valid_arms < self.min_narms or num_valid_arms > self.max_narms:
            print(f"Invalid arm count: {num_valid_arms} (expected between {self.min_narms} and {self.max_narms})")
            return False
        
        if max_narms > self.max_narms:
            print(f"Max arms exceeded: {max_narms} (expected <= {self.max_narms})")
            return False
        
        # Check for invalid values only in valid arms
        if np.any(valid_arms_mask):
            valid_genome = genome[valid_arms_mask]
            if not np.isfinite(valid_genome).all():
                print("Genome contains NaN or infinite values in valid arms")
                return False
        
        # Check parameter bounds only for valid arms
        if np.any(valid_arms_mask):
            valid_genome = genome[valid_arms_mask]
            for i in range(7):
                param_values = valid_genome[:, i]
                if np.any(param_values < self.parameter_limits[i, 0]) or \
                   np.any(param_values > self.parameter_limits[i, 1]):
                    print(f"Parameter {i} out of bounds, values: {param_values}")
                    return False
            
            # Check propeller direction values
            directions = valid_genome[:, 6]
            if not np.all(np.isin(directions, [0, 1])):
                print("Direction values must be binary (0 or 1)")
                return False
        
        # Check symmetry constraints if symmetry operator is provided
        if self.symmetry_operator is not None:
            if not self.symmetry_operator.validate_symmetry(genome):
                print("Genome does not satisfy symmetry constraints")
                return False
        
        return True
    
    def get_bounds(self) -> npt.NDArray[Any]:
        """
        Get the parameter bounds for Cartesian genomes.
        
        Returns:
            Array of shape (7, 2) with [min, max] for each parameter
        """
        return self.parameter_limits.copy()
    
    def repair_collisions(self, genome: npt.NDArray[Any], repair_along_fixed_axis=None) -> npt.NDArray[Any]:
        """
        Repair collisions in a Cartesian genome using the particle repair operator.
        
        Args:
            genome: Cartesian genome array of shape (max_narms, 7)
            
        Returns:
            Genome with collisions repaired
        """
        if genome.shape[1] != 7:
            raise ValueError("Cartesian genome must have 7 parameters per arm")
        
        # Apply collision repair using the particle repair operator
        repaired_valid_genome = particle_repair_individual(
            genome,
            propeller_radius=self.config.propeller_radius,
            inner_boundary_radius=self.config.inner_boundary_radius,
            outer_boundary_radius=self.config.outer_boundary_radius,
            max_iterations=self.config.max_repair_iterations,
            step_size=self.config.repair_step_size,
            propeller_tolerance=self.config.propeller_tolerance,
            repair_along_fixed_axis=repair_along_fixed_axis,
            arms_to_cylinders=arms_to_cylinders_cartesian_euler,
            cylinders_to_arms=cylinders_to_arms_cartesian_euler
        )
        
        return repaired_valid_genome
    
    def _clip_parameters(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Clip parameters to their bounds."""
        result = genome.copy()
        
        # Find valid arms and clip only those parameters
        valid_arms_mask = ~np.isnan(genome[:, 0])
        
        if np.any(valid_arms_mask):
            # Clip all parameters to their bounds for valid arms only
            for i in range(7):
                valid_values = result[valid_arms_mask, i]
                clipped_values = np.clip(
                    valid_values,
                    self.parameter_limits[i, 0],
                    self.parameter_limits[i, 1]
                )
                result[valid_arms_mask, i] = clipped_values
        
        return result
    
    def _handle_invalid_values(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Handle NaN and infinite values."""
        # Detect invalid values
        invalid_mask = RepairUtilities.detect_invalid_values(genome)
        
        if not np.any(invalid_mask):
            return genome  # No invalid values to repair
        
        result = genome.copy()
        
        # Replace invalid values with random valid values
        replacement_genome = self._generate_random_genome(genome.shape)
        result = RepairUtilities.replace_invalid_values(
            result, replacement_genome, invalid_mask
        )
        
        return result
    
    def _generate_random_genome(self, shape: tuple) -> npt.NDArray[Any]:
        """Generate a random genome for replacement values."""
        return RepairUtilities.generate_random_replacement(
            shape, self.parameter_limits, self.rng
        )
    
    def repair_with_symmetry_plane(
        self, 
        genome: npt.NDArray[Any], 
        plane: str
    ) -> npt.NDArray[Any]:
        """
        Repair genome with specific symmetry plane.
        
        Args:
            genome: Genome to repair
            plane: Symmetry plane ("xy", "xz", "yz", or None)
            
        Returns:
            Repaired genome with symmetry applied
        """
        # Create temporary symmetry operator with specified plane
        from .symmetry_base import SymmetryConfig
        
        temp_config = SymmetryConfig(plane=plane)
        temp_symmetry = CartesianSymmetryOperator(
            config=temp_config
        )
        
        # Temporarily set symmetry operator
        original_symmetry = self.symmetry_operator
        self.symmetry_operator = temp_symmetry
        
        try:
            result = self.repair(genome)
        finally:
            # Restore original symmetry operator
            self.symmetry_operator = original_symmetry
        
        return result
    
    def validate_with_tolerance(
        self, 
        genome: npt.NDArray[Any], 
        tolerance: float = 1e-9
    ) -> bool:
        """
        Validate genome with custom tolerance.
        
        Args:
            genome: Genome to validate
            tolerance: Tolerance for bound checking
            
        Returns:
            True if genome is valid within tolerance
        """
        # Check shape
        if genome.shape[1] != 7:
            return False
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        
        if np.any(valid_arms_mask):
            valid_genome = genome[valid_arms_mask]
            
            # Check for NaN or infinite values in valid arms
            if not np.isfinite(valid_genome).all():
                return False
            
            # Check bounds with tolerance
            bounds_valid = RepairUtilities.validate_bounds(valid_genome, self.parameter_limits, tolerance)
            if not np.all(bounds_valid):
                return False
            
            # Check binary direction values
            directions = valid_genome[:, 6]
            if not np.all(np.isin(directions, [0, 1])):
                return False
        
        return True
    
    def _count_bounds_violations(self, genome: npt.NDArray[Any]) -> dict:
        """Count bounds violations for each parameter."""
        violations = {}
        param_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'direction']
        
        # Find valid arms
        valid_arms_mask = ~np.isnan(genome[:, 0])
        
        if np.any(valid_arms_mask):
            valid_genome = genome[valid_arms_mask]
            
            for i, name in enumerate(param_names):
                values = valid_genome[:, i]
                min_bound, max_bound = self.parameter_limits[i]
                
                violations[name] = {
                    'below_min': np.sum(values < min_bound),
                    'above_max': np.sum(values > max_bound),
                    'total': np.sum((values < min_bound) | (values > max_bound))
                }
        else:
            # No valid arms, so no violations
            for name in param_names:
                violations[name] = {
                    'below_min': 0,
                    'above_max': 0,
                    'total': 0
                }
        
        return violations
    
    def set_parameter_limits(self, parameter_limits: npt.NDArray[Any]) -> None:
        """
        Update parameter limits.
        
        Args:
            parameter_limits: New parameter limits array of shape (7, 2)
        """
        if parameter_limits.shape != (7, 2):
            raise ValueError("parameter_limits must have shape (7, 2)")
        self.parameter_limits = parameter_limits.copy()
    
    def set_arm_count_limits(self, min_narms: int, max_narms: int) -> None:
        """
        Update arm count limits.
        
        Args:
            min_narms: New minimum number of arms
            max_narms: New maximum number of arms
        """
        if min_narms < 1:
            raise ValueError("min_narms must be at least 1")
        if max_narms < min_narms:
            raise ValueError("max_narms must be >= min_narms")
        
        self.min_narms = min_narms
        self.max_narms = max_narms
    
    def set_symmetry_operator(self, symmetry_operator: CartesianSymmetryOperator) -> None:
        """
        Set the symmetry operator.
        
        Args:
            symmetry_operator: New symmetry operator
        """
        self.symmetry_operator = symmetry_operator
    
    def __repr__(self) -> str:
        return (f"CartesianRepairOperator(config={self.config}, "
                f"arm_limits=({self.min_narms}, {self.max_narms}), "
                f"has_symmetry={self.symmetry_operator is not None})")