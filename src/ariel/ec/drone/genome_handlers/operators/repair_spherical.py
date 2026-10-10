"""
Spherical repair operator for drone genome handlers.

This module implements repair operations for spherical coordinate
representations with angular orientations.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from .repair_base import RepairOperator, RepairConfig, RepairUtilities
from .symmetry_spherical import SphericalSymmetryOperator
from ..conversions.arm_conversions import arms_to_cylinders_polar_angular, cylinders_to_arms_polar_angular
from .particle_repair_operator import particle_repair_individual
from .symmetry_base import SymmetryPlane

class SphericalRepairOperator(RepairOperator):
    """
    Repair operator for spherical coordinate genomes with angular orientations.
    
    Genome format expected: (max_narms, 6) with columns:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    
    Uses NaN masking for variable arm counts.
    """
    
    def __init__(
        self,
        config: Optional[RepairConfig] = None,
        parameter_limits: Optional[npt.NDArray[Any]] = None,
        min_narms: int = 1,
        max_narms: int = 8,
        symmetry_operator: Optional[SphericalSymmetryOperator] = None,
        rng: Optional[np.random.Generator] = None
    ):
        """
        Initialize the Spherical repair operator.
        
        Args:
            config: Repair configuration
            parameter_limits: Array of shape (6, 2) with [min, max] for each parameter
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
                [0.5, 2.0],          # magnitude
                [-np.pi, np.pi],     # arm rotation
                [-np.pi, np.pi],     # arm pitch
                [-np.pi, np.pi],     # motor rotation
                [-np.pi, np.pi],     # motor pitch
                [0, 1]               # direction (binary)
            ])
        else:
            if parameter_limits.shape != (6, 2):
                raise ValueError("parameter_limits must have shape (6, 2)")
            self.parameter_limits = parameter_limits.copy()
    
    def repair(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a spherical genome to make it valid.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            Repaired genome
        """
        if genome.shape[1] != 6:
            raise ValueError("Spherical genome must have 6 parameters per arm")
        narms = np.sum(~np.isnan(genome[:, 0]))
        if narms <= 1:
            return genome.copy()
        
        result = genome.copy()
        # Unapply symmetry if it was applied
        if self.config.apply_symmetry and self.symmetry_operator.get_plane() is not None:
            result = self.symmetry_operator.unapply_symmetry(result)
            if self.symmetry_operator.get_plane() is  SymmetryPlane.XY:
                axis = [0, 0, 1]
            elif self.symmetry_operator.get_plane() is  SymmetryPlane.XZ:
                axis = [0, 1, 0]
            elif self.symmetry_operator.get_plane() is SymmetryPlane.YZ:
                axis = [1, 0, 0]
        
        # Step 1: Wrap angular parameters to their valid ranges
        result = self._wrap_angular_parameters(result)

        # Step 2: Clip parameters to bounds for valid arms
        result = self._clip_parameters(result)

        # Step 3: Apply collision repair if enabled
        if self.config.enable_collision_repair:
            result = self.repair_collisions(result)

        # Step 4: Apply symmetry restoration if enabled
        if self.config.apply_symmetry and self.symmetry_operator is not None:
            result = self.symmetry_operator.apply_symmetry(result)
            # Apply collision repair again but on certain axis
            if self.config.enable_collision_repair:
                result = self.repair_collisions(result, repair_along_fixed_axis=axis)

        return result
    
    def validate(self, genome: npt.NDArray[Any]) -> bool:
        """
        Check if a spherical genome is valid.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            True if genome is valid
        """
        # Check genome shape first
        if len(genome.shape) != 2 or genome.shape[1] != 6:
            print(f"Invalid genome shape: {genome.shape}, expected (max_narms, 6)")
            return False

        # Count valid arms (non-NaN)
        valid_arms_mask = ~np.isnan(genome[:, 0])
        num_valid_arms = np.sum(valid_arms_mask)
        
        # Check arm count constraints
        if num_valid_arms < self.min_narms or num_valid_arms > self.max_narms:
            print(f"Invalid arm count: {num_valid_arms} (expected between {self.min_narms} and {self.max_narms})")
            return False
        
        # Check parameter bounds for valid arms
        if num_valid_arms > 0:
            valid_arms = genome[valid_arms_mask]
            
            for i in range(6):
                param_values = valid_arms[:, i]
                if np.any(param_values < self.parameter_limits[i, 0]) or \
                   np.any(param_values > self.parameter_limits[i, 1]):
                    print(f"Parameter {i} out of bounds: {param_values}, Genome: \n{genome}. Upper bounds: {self.parameter_limits[i, 1]}, Lower bounds: {self.parameter_limits[i, 0]}")
                    return False
            
            # Check that direction values are 0 or 1
            directions = valid_arms[:, 5]
            if not np.all(np.isin(directions, [0, 1])):
                print(f"Invalid direction values: {directions}")
                return False
        
        # Check symmetry constraints if symmetry operator is provided and enabled
        if (self.symmetry_operator is not None and 
            hasattr(self.symmetry_operator, 'is_enabled') and 
            self.symmetry_operator.is_enabled()):
            if not self.symmetry_operator.validate_symmetry(genome):
                print("Genome does not satisfy symmetry constraints")
                return False
        
        return True
    
    def get_bounds(self) -> npt.NDArray[Any]:
        """
        Get the parameter bounds for spherical genomes.
        
        Returns:
            Array of shape (6, 2) with [min, max] for each parameter
        """
        return self.parameter_limits.copy()
    
    def repair_collisions(self, genome: npt.NDArray[Any], repair_along_fixed_axis=None) -> npt.NDArray[Any]:
        """
        Repair collisions in a spherical genome using the particle repair operator.
        
        Args:
            genome: Spherical genome array of shape (max_narms, 6)
            
        Returns:
            Genome with collisions repaired
        """
        if genome.shape[1] != 6:
            raise ValueError("Spherical genome must have 6 parameters per arm")
        
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
            arms_to_cylinders=arms_to_cylinders_polar_angular,
            cylinders_to_arms=cylinders_to_arms_polar_angular
        )
        
        return repaired_valid_genome
    
    def _clip_parameters(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Clip parameters to bounds for valid arms only."""
        result = genome.copy()
        valid_arms_mask = ~np.isnan(result[:, 0])
        
        # Clip parameters for valid arms
        for i in range(6):
            new_params = np.clip(
                result[valid_arms_mask, i],
                self.parameter_limits[i, 0],
                self.parameter_limits[i, 1]
            )
            # Special handling for direction parameter (index 5) - must be 0 or 1
            if i == 5:
                new_params = np.round(new_params).astype(int)
            result[valid_arms_mask, i] = new_params
        
        return result
    
    def _wrap_angular_parameters(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Wrap angular parameters to their valid ranges."""
        result = genome.copy()
        valid_arms_mask = ~np.isnan(result[:, 0])
        
        # Wrap angular parameters that should be periodic
        angular_params = [1, 2, 3, 4]  # arm_rotation, arm_pitch, motor_rotation, motor_pitch
        
        for param_idx in angular_params:
            values = result[valid_arms_mask, param_idx]
            # Wrap all angular parameters to [-π, π] range
            wrapped = np.mod(values + np.pi, 2*np.pi) - np.pi
            result[valid_arms_mask, param_idx] = wrapped
        
        return result
    
    def repair_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a population of spherical genomes.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 6)
            
        Returns:
            Repaired population
        """
        result = np.empty_like(population)
        
        for i in range(population.shape[0]):
            result[i] = self.repair(population[i])
        
        return result
    
    def validate_population(self, population: npt.NDArray[Any]) -> npt.NDArray[bool]:
        """
        Validate a population of spherical genomes.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 6)
            
        Returns:
            Boolean array indicating which individuals are valid
        """
        result = np.empty(population.shape[0], dtype=bool)
        
        for i in range(population.shape[0]):
            result[i] = self.validate(population[i])
        
        return result
    
    def get_arm_count(self, genome: npt.NDArray[Any]) -> int:
        """
        Get the number of valid arms in a genome.
        
        Args:
            genome: Spherical genome array
            
        Returns:
            Number of valid arms
        """
        valid_arms_mask = ~np.isnan(genome[:, 0])
        return np.sum(valid_arms_mask)
    
    def get_valid_arms(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Get only the valid (non-NaN) arms from a genome.
        
        Args:
            genome: Spherical genome array
            
        Returns:
            Array of valid arms
        """
        valid_arms_mask = ~np.isnan(genome[:, 0])
        return genome[valid_arms_mask].copy()

    
    def remove_random_arm(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Remove a random arm (set to NaN).
        
        Args:
            genome: Spherical genome array
            
        Returns:
            Genome with removed arm, or original if min arms reached
        """
        result = genome.copy()
        valid_arms_mask = ~np.isnan(result[:, 0])
        num_valid_arms = np.sum(valid_arms_mask)
        
        if num_valid_arms <= self.min_narms:
            return result  # Can't remove, would go below minimum
        
        # Select random valid arm
        valid_indices = np.where(valid_arms_mask)[0]
        selected_index = self.rng.choice(valid_indices)
        
        # Remove arm
        result[selected_index] = np.nan
        
        return result
    
    def _count_bounds_violations(self, arms: npt.NDArray[Any]) -> dict:
        """Count bounds violations for each parameter."""
        violations = {}
        
        param_names = ['magnitude', 'arm_rotation', 'arm_pitch', 
                      'motor_rotation', 'motor_pitch', 'direction']
        
        for i, name in enumerate(param_names):
            values = arms[:, i]
            min_bound, max_bound = self.parameter_limits[i]
            
            violations[name] = {
                'below_min': np.sum(values < min_bound),
                'above_max': np.sum(values > max_bound),
                'total': np.sum((values < min_bound) | (values > max_bound))
            }
        
        return violations
    
    def set_parameter_limits(self, parameter_limits: npt.NDArray[Any]) -> None:
        """
        Update parameter limits.
        
        Args:
            parameter_limits: New parameter limits array of shape (6, 2)
        """
        if parameter_limits.shape != (6, 2):
            raise ValueError("parameter_limits must have shape (6, 2)")
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
    
    def set_symmetry_operator(self, symmetry_operator: SphericalSymmetryOperator) -> None:
        """
        Set the symmetry operator.
        
        Args:
            symmetry_operator: New symmetry operator
        """
        self.symmetry_operator = symmetry_operator
    
    def __repr__(self) -> str:
        return (f"SphericalRepairOperator(config={self.config}, "
                f"arm_limits=({self.min_narms}, {self.max_narms}), "
                f"has_symmetry={self.symmetry_operator is not None})")