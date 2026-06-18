from __future__ import annotations

from typing import Any, List, Tuple, Optional

import numpy as np
import numpy.typing as npt

from .base import GenomeHandler
from .operators import (
    CartesianSymmetryOperator, 
    CartesianRepairOperator,
    SymmetryConfig,
    RepairConfig
)


class CartesianEulerDroneGenomeHandler(GenomeHandler):
    """
    Genome handler for drone designs using Cartesian coordinates and Euler angles.
    
    Genome is defined by an n by m matrix, where n is the maximum number of arms/motors 
    and m is the number of parameters (7 total):
    - 3 for motor position (x, y, z) in Cartesian coordinates
    - 3 for motor orientation (roll, pitch, yaw) in Euler angles
    - 1 for propeller direction of spin (0 or 1)
    
    Uses NaN masking for variable arm counts, where inactive arms are marked with NaN.
    """

    def __init__(
        self,
        genome: npt.NDArray[Any] | None = None,
        min_max_narms: Tuple[int, int] | None = None,
        parameter_limits: npt.NDArray[Any] | None = None,
        append_arm_chance: float = 0.0,
        mutation_probs: Optional[List[float] | npt.NDArray[Any]] = None,
        mutation_scales_percentage: Optional[npt.NDArray[Any]] = None,
        bilateral_plane_for_symmetry: str | None = None,
        repair: bool = True,
        enable_collision_repair: bool = True,
        propeller_radius: float = 0.0254,  # 2-inch propeller radius in meters
        inner_boundary_radius: float = 0.09,
        outer_boundary_radius: float = 0.4,
        max_repair_iterations: int = 100,
        repair_step_size: float = 1.0,
        propeller_tolerance: float = 0.1,
        rnd: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the Cartesian Euler genome handler.
        
        Args:
            genome: Pre-existing genome array of shape (narms, 7)
            min_max_narms: Tuple of (min_arms, max_arms). If None, defaults to (4, 4)
            parameter_limits: Array of shape (7, 2) with [min, max] for each parameter. If None, uses defaults
            append_arm_chance: Probability of adding/removing arms during mutation (not used for fixed arms)
            mutation_probs: Custom mutation probabilities for each parameter. If None, uses defaults
            mutation_scales_percentage: Mutation scales as percentage of parameter ranges. If None, uses defaults
            bilateral_plane_for_symmetry: Plane for bilateral symmetry ("xy", "xz", "yz", or None)
            repair: Whether to apply repair operations
            enable_collision_repair: Whether to enable collision detection and repair
            propeller_radius: Radius of propellers for collision detection
            inner_boundary_radius: Minimum distance from origin
            outer_boundary_radius: Maximum distance from origin
            max_repair_iterations: Maximum iterations for collision repair
            repair_step_size: Step size multiplier for collision resolution
            propeller_tolerance: Additional clearance tolerance for propellers
            rnd: Random number generator
        """
        self.rnd = rnd if rnd is not None else np.random.default_rng()
        
        # Set basic attributes from standardized parameters
        if min_max_narms is None:
            self.min_narms, self.max_narms = 3, 8  # Variable number of arms by default
        else:
            self.min_narms, self.max_narms = min_max_narms
        
        # Setup parameter limits
        if parameter_limits is None:
            # Default limits: [x, y, z, roll, pitch, yaw, direction]
            self.parameter_limits = np.array([
                [-0.4, 0.4],    # x position
                [-0.4, 0.4],    # y position  
                [-0.4, 0.4],    # z position
                [-np.pi, np.pi], # roll
                [-np.pi, np.pi], # pitch
                [-np.pi, np.pi], # yaw
                [0, 1]          # direction
            ])
        else:
            self.parameter_limits = np.asarray(parameter_limits)
            if self.parameter_limits.shape != (7, 2):
                raise ValueError("parameter_limits must have shape (7, 2)")
        
        # Validate inputs
        self._validate_initialization_parameters(
            (self.min_narms, self.max_narms), self.parameter_limits, append_arm_chance, 
            mutation_probs, mutation_scales_percentage
        )
        
        self.append_arm_chance = append_arm_chance
        if self.min_narms == self.max_narms and self.append_arm_chance != 0:
            raise ValueError("When min_narms == max_narms, append_arm_chance must be 0!")
        
        self.repair_enabled = repair
        
        # Store collision repair parameters
        self.enable_collision_repair = enable_collision_repair
        self.propeller_radius = propeller_radius
        self.inner_boundary_radius = inner_boundary_radius
        self.outer_boundary_radius = outer_boundary_radius
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size
        self.propeller_tolerance = propeller_tolerance
        
        # Validate and store bilateral symmetry plane
        valid_planes = {"xy", "xz", "yz", None}
        if bilateral_plane_for_symmetry not in valid_planes:
            raise ValueError(f"bilateral_plane_for_symmetry must be one of {valid_planes}")
        
        # Check for odd arms with bilateral symmetry
        if bilateral_plane_for_symmetry is not None and self.max_narms % 2 != 0:
            raise ValueError("Bilateral symmetry requires an even number of arms")
            
        self.bilateral_plane_for_symmetry = bilateral_plane_for_symmetry

        # Setup mutation parameters
        self._setup_mutation_parameters(mutation_probs, mutation_scales_percentage)
        
        # Store original mutation_probs for later access (before normalization)
        if mutation_probs is not None:
            self.mutation_probs = np.asarray(mutation_probs)
        else:
            # Default uniform probabilities for 7 parameters
            self.mutation_probs = np.array([1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7])
        
        # Initialize operators
        self._setup_operators()
        
        # Initialize genome using parent class
        super().__init__(genome)

    def _validate_initialization_parameters(
        self,
        min_max_narms: Tuple[int, int],
        parameter_limits: npt.NDArray[Any],
        append_arm_chance: float,
        mutation_probs: Optional[List[float] | npt.NDArray[Any]],
        mutation_scales_percentage: Optional[npt.NDArray[Any]]
    ) -> None:
        """Validate initialization parameters."""
        min_narms, max_narms = min_max_narms
        
        if min_narms < 1:
            raise ValueError("min_narms must be at least 1")
        if max_narms < min_narms:
            raise ValueError("max_narms must be >= min_narms")
        if not (0 <= append_arm_chance <= 0.5):
            raise ValueError("append_arm_chance must be between 0 and 0.5")
        
        parameter_limits = np.asarray(parameter_limits)
        if parameter_limits.shape != (7, 2):
            raise ValueError("parameter_limits must have shape (7, 2)")
        if np.any(parameter_limits[:, 0] >= parameter_limits[:, 1]):
            raise ValueError("parameter_limits: min values must be < max values")
        
        if mutation_probs is not None:
            mutation_probs = np.asarray(mutation_probs)
            if len(mutation_probs) != 7:
                raise ValueError("mutation_probs must have length 7")
            if np.any(mutation_probs < 0):
                raise ValueError("mutation_probs must be non-negative")
        
        if mutation_scales_percentage is not None:
            mutation_scales_percentage = np.asarray(mutation_scales_percentage)
            if len(mutation_scales_percentage) != 7:
                raise ValueError("mutation_scales_percentage must have length 7")
            if np.any(mutation_scales_percentage < 0):
                raise ValueError("mutation_scales_percentage must be non-negative")

    def _setup_mutation_parameters(
        self,
        mutation_probs: Optional[List[float] | npt.NDArray[Any]],
        mutation_scales_percentage: Optional[npt.NDArray[Any]]
    ) -> None:
        """Setup mutation probabilities and scales."""
        nparms = 7
        mutation_window = 1 - 2 * self.append_arm_chance
        
        # Setup mutation probabilities
        if mutation_probs is None:
            # Uniform distribution across parameters
            param_prob = mutation_window / nparms
            mutation_probs = np.repeat(param_prob, nparms)
        else:
            mutation_probs = np.asarray(mutation_probs)
            # Normalize to fit in mutation window
            total_prob = np.sum(mutation_probs)
            if total_prob > 0:
                mutation_probs = mutation_probs * (mutation_window / total_prob)
        
        # Combine with add/remove probabilities
        self.mutation_probabilities = np.array([
            self.append_arm_chance,  # Add arm probability
            self.append_arm_chance,  # Remove arm probability
            *mutation_probs          # Parameter mutation probabilities
        ])
        
        # Validate probabilities sum to 1
        prob_sum = np.sum(self.mutation_probabilities)
        if not np.isclose(prob_sum, 1.0, rtol=1e-6):
            raise ValueError(f"Mutation probabilities must sum to 1, got {prob_sum}")
        
        # Setup mutation scales
        if mutation_scales_percentage is None:
            mutation_scales_percentage = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        
        # Convert percentages to absolute scales based on parameter ranges
        parameter_ranges = self.parameter_limits[:, 1] - self.parameter_limits[:, 0]
        self.mutation_scales = parameter_ranges * mutation_scales_percentage

    def _setup_operators(self) -> None:
        """Initialize the symmetry and repair operators."""
        # Setup symmetry operator
        symmetry_config = SymmetryConfig(
            plane=self.bilateral_plane_for_symmetry,
            enabled=self.bilateral_plane_for_symmetry is not None
        )
        self.symmetry_operator = CartesianSymmetryOperator(
            config=symmetry_config
        )
        
        # Setup repair operator
        repair_config = RepairConfig(
            apply_symmetry=self.bilateral_plane_for_symmetry is not None,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance
        )
        self.repair_operator = CartesianRepairOperator(
            config=repair_config,
            parameter_limits=self.parameter_limits,
            min_narms=self.min_narms,
            max_narms=self.max_narms,
            symmetry_operator=self.symmetry_operator,
            rng=self.rnd
        )

    def _generate_random_genome(self) -> npt.NDArray[Any]:
        """Generate a random genome with variable arm count using NaN masking."""
        # Determine the number of arms to generate (can be variable)
        if self.min_narms == self.max_narms:
            num_arms = self.max_narms
        else:
            num_arms = self.rnd.integers(self.min_narms, self.max_narms + 1)
        
        # If symmetry is enabled, only generate half the arms initially
        if self.bilateral_plane_for_symmetry is not None:
            num_arms_to_generate = self.max_narms // 2
        else:
            num_arms_to_generate = num_arms
        
        # Create genome with max_narms rows, filled with NaN initially
        genome = np.full((self.max_narms, 7), np.nan)
        
        if num_arms_to_generate > 0:
            # Motor positions: random uniform within parameter limits
            motor_positions = self.rnd.uniform(
                low=self.parameter_limits[:3, 0],
                high=self.parameter_limits[:3, 1],
                size=(num_arms_to_generate, 3),
            )
            
            # Motor orientations: random uniform within parameter limits  
            motor_orientations = self.rnd.uniform(
                low=self.parameter_limits[3:6, 0],
                high=self.parameter_limits[3:6, 1],
                size=(num_arms_to_generate, 3),
            )
            
            # Propeller directions: random choice within parameter limits
            propeller_directions = self.rnd.integers(
                low=int(self.parameter_limits[6, 0]),
                high=int(self.parameter_limits[6, 1]) + 1,
                size=(num_arms_to_generate, 1),
            )
            
            # Fill in the valid arms
            valid_genome = np.concatenate(
                (motor_positions, motor_orientations, propeller_directions),
                axis=1,
            )
            genome[:num_arms_to_generate] = valid_genome

        if self.bilateral_plane_for_symmetry is not None:
            # If symmetry is enabled, apply symmetry to the genome
            genome = self.symmetry_operator.apply_symmetry(genome)
        # Apply repair if enabled (this will handle symmetry application)
        if self.repair_enabled:
            genome = self.repair_operator.repair(genome)
        
        return genome

    def generate_random_population(self, population_size: int) -> List[CartesianEulerDroneGenomeHandler]:
        """
        Generate a population of random genome handlers.
        
        Args:
            population_size: Number of individuals to generate
            
        Returns:
            List of random genome handler instances
        """
        population = []
        for _ in range(population_size):
            random_genome = self._generate_random_genome()
            individual = CartesianEulerDroneGenomeHandler(
                genome=random_genome,
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
                append_arm_chance=self.append_arm_chance,
                mutation_probs=self.mutation_probs,
                mutation_scales_percentage=(self.mutation_scales / (self.parameter_limits[:, 1] - self.parameter_limits[:, 0])),
                bilateral_plane_for_symmetry=self.bilateral_plane_for_symmetry,
                repair=self.repair_enabled,
                enable_collision_repair=self.enable_collision_repair,
                propeller_radius=self.propeller_radius,
                inner_boundary_radius=self.inner_boundary_radius,
                outer_boundary_radius=self.outer_boundary_radius,
                max_repair_iterations=self.max_repair_iterations,
                repair_step_size=self.repair_step_size,
                propeller_tolerance=self.propeller_tolerance,
                rnd=self.rnd,
            )
            population.append(individual)
        return population

    def random_population(self, pop_size: int) -> npt.NDArray[Any]:
        """
        Generate a random population as a numpy array (vectorized version).
        
        Args:
            pop_size: Size of the population to generate
            
        Returns:
            Population array of shape (pop_size, max_narms, 7)
        """
        array_shape = (pop_size, self.max_narms, 7)
        population = np.empty(array_shape)
        
        # Generate random arms for all individuals
        population[:, :, :6] = self.rnd.uniform(
            low=self.parameter_limits[:6, 0], 
            high=self.parameter_limits[:6, 1], 
            size=(pop_size, self.max_narms, 6)
        )
        population[:, :, 6] = self.rnd.integers(0, 2, size=(pop_size, self.max_narms))
        
        if self.bilateral_plane_for_symmetry is not None:
            # If symmetry is enabled, only keep half of the genome
            population[:, self.max_narms // 2:, :] = np.nan
        else:
            # Remove random arms from each drone to create variable arm counts
            num_arms_to_remove = self.rnd.integers(
                0, self.max_narms + 1 - self.min_narms, 
                size=pop_size
            )
            
            # Create mask for arms to remove
            arms_to_remove_mask = np.zeros(array_shape, dtype=bool)
            for i in range(pop_size):
                if num_arms_to_remove[i] > 0:
                    indices_to_remove = self.rnd.choice(
                        self.max_narms, 
                        num_arms_to_remove[i], 
                        replace=False
                    )
                    arms_to_remove_mask[i, indices_to_remove, :] = True
            
            # Apply mask (set removed arms to NaN)
            population[arms_to_remove_mask] = np.nan
        
        # Apply post-processing
        if self.repair_enabled:
            population = self.repair_population(population)
        
        return population

    def crossover(self, other: CartesianEulerDroneGenomeHandler) -> CartesianEulerDroneGenomeHandler:
        """
        Perform crossover with another genome handler to produce offspring.
        
        Args:
            other: The other parent genome handler
            
        Returns:
            Child genome handler from crossover
        """
        if not isinstance(other, CartesianEulerDroneGenomeHandler):
            raise TypeError("Other parent must be CartesianEulerDroneGenomeHandler")
        
        if self.genome.shape != other.genome.shape:
            raise ValueError("Parents must have the same genome shape")

        # Create child with same parameters as parents
        child = CartesianEulerDroneGenomeHandler(
            genome=None,  # Will be set below
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
            mutation_probs=self.mutation_probs,
            mutation_scales_percentage=(self.mutation_scales / (self.parameter_limits[:, 1] - self.parameter_limits[:, 0])),
            bilateral_plane_for_symmetry=self.bilateral_plane_for_symmetry,
            repair=self.repair_enabled,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
            rnd=self.rnd,
        )

        # Arm-wise crossover: randomly select each arm from either parent
        random_choices = self.rnd.choice([0, 1], size=self.max_narms)
        child_genome = np.where(
            random_choices[:, np.newaxis] == 0, 
            self.genome, 
            other.genome
        )
        
        child.genome = child_genome
        
        # Apply bilateral symmetry if specified
        if self.bilateral_plane_for_symmetry is not None:
            child.apply_symmetry()
        
        return child

    def crossover_vectorized(
        self, 
        population: npt.NDArray[Any], 
        mating_pool1: npt.NDArray[Any], 
        mating_pool2: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """
        Vectorized crossover operation for efficiency.
        
        Args:
            population: Population array of shape (pop_size, max_narms, 7)
            mating_pool1: Indices of first parents
            mating_pool2: Indices of second parents
            
        Returns:
            Children population array
        """
        pop_size, max_narms, nparms = population.shape
        
        # Shuffle arms of each individual for genetic diversity
        shuffle_indices = np.array([
            self.rnd.permutation(max_narms) for _ in range(pop_size)
        ])
        population_shuffled = population[np.arange(pop_size)[:, None], shuffle_indices]
        
        # Select parents
        parent1s = population_shuffled[mating_pool1]
        parent2s = population_shuffled[mating_pool2]
        
        if parent1s.shape != parent2s.shape:
            raise ValueError("Parent populations must have the same shape")
        
        # Generate random choices for arm selection
        random_choices = self.rnd.choice([0, 1], size=parent1s.shape[:-1])
        
        # Perform crossover
        children = np.where(
            random_choices[..., np.newaxis] == 0, 
            parent1s, 
            parent2s
        )
        
        # Apply repair if needed
        if self.repair_enabled:
            children = self.repair_population(children)
        
        return children

    def _mutate_add_arm(self, genome: npt.NDArray[Any]) -> None:
        """Add a random arm to the genome."""
        # Find empty slots (NaN values)
        empty_mask = np.isnan(genome[:, 0])
        
        if not np.any(empty_mask) or (self.bilateral_plane_for_symmetry is not None and not np.any(empty_mask[:self.max_narms // 2])):
            return  # No empty slots available

        # Select random empty slot
        empty_indices = np.where(empty_mask)[0]
        selected_index = self.rnd.choice(empty_indices)
        
        # Generate new arm parameters
        new_arm = np.zeros(7)
        new_arm[:6] = self.rnd.uniform(
            low=self.parameter_limits[:6, 0],
            high=self.parameter_limits[:6, 1]
        )
        new_arm[6] = self.rnd.integers(0, 2)
        
        genome[selected_index] = new_arm

        return genome

    def _mutate_remove_arm(self, genome: npt.NDArray[Any]) -> None:
        """Remove a random arm from the genome."""
        # Find non-empty slots
        non_empty_mask = ~np.isnan(genome[:, 0])
        if not np.any(non_empty_mask):
            return  # No arms to remove
        
        # Select random non-empty slot
        non_empty_indices = np.where(non_empty_mask)[0]
        selected_index = self.rnd.choice(non_empty_indices)
        
        genome[selected_index] = np.nan

        return genome

    def _mutate_parameter(self, genome: npt.NDArray[Any], param_index: int) -> None:
        """Mutate a specific parameter of a random arm."""
        # Find non-empty arms
        non_empty_mask = ~np.isnan(genome[:, 0])
        if not np.any(non_empty_mask):
            return  # No arms to mutate
        
        # Select random arm
        non_empty_indices = np.where(non_empty_mask)[0]
        selected_arm = self.rnd.choice(non_empty_indices)
        
        if param_index == 6:
            # Flip direction parameter
            genome[selected_arm, param_index] = 1 - genome[selected_arm, param_index]
        else:
            # Add Gaussian perturbation
            perturbation = self.rnd.normal(0, self.mutation_scales[param_index])
            genome[selected_arm, param_index] += perturbation
            
            # Clip to parameter limits
            genome[selected_arm, param_index] = np.clip(
                genome[selected_arm, param_index],
                self.parameter_limits[param_index, 0],
                self.parameter_limits[param_index, 1]
            )

        return genome
    
    def mutate(self) -> None:
        """Mutate this genome in place."""

        if self.bilateral_plane_for_symmetry is not None:
            # Temporarily remove symmetry for mutation
            self.apply_symmetry()
            genome_half = self.genome.copy()
        else:
            genome_half = self.genome.copy()
        
        # Choose mutation type
        mutation_type = self.rnd.choice(
            len(self.mutation_probabilities),
            p=self.mutation_probabilities
        )

        if mutation_type == 0:
            # Add arm mutation
            genome_half = self._mutate_add_arm(genome_half)
        elif mutation_type == 1:
            # Remove arm mutation
            genome_half = self._mutate_remove_arm(genome_half)
        else:
            # Parameter mutation
            param_index = mutation_type - 2
            genome_half = self._mutate_parameter(genome_half, param_index)
        
        # Update genome
        self.genome = genome_half
        
        if self.bilateral_plane_for_symmetry is not None:
            # Reapply symmetry after mutation
            self.apply_symmetry()
        # Apply repair if enabled (this will handle symmetry application)
        if self.repair_enabled:
            self.repair()
    def mutation_vectorized(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Vectorized mutation operation for efficiency.
        
        Args:
            population: Population array to mutate
            
        Returns:
            Mutated population array
        """
        if self.bilateral_plane_for_symmetry is not None:
            population = self.unapply_symmetry_pop(population)
        
        pop_size, max_narms, nparms = population.shape
        pop = population.copy()
        
        # Choose mutation type for each individual
        mutation_choices = np.nonzero(self.mutation_probabilities)[0]
        mutations = self.rnd.choice(
            mutation_choices,
            size=pop_size,
            p=self.mutation_probabilities[self.mutation_probabilities != 0]
        )
        
        # Apply different mutation types
        self._apply_add_mutations(pop, mutations)
        self._apply_remove_mutations(pop, mutations)
        self._apply_parameter_mutations(pop, mutations)
        
        # Apply post-processing
        if self.repair_enabled:
            pop = self.repair_population(pop)
        elif self.bilateral_plane_for_symmetry is not None:
            pop = self.apply_symmetry_pop(pop)

        return pop

    def _apply_add_mutations(
        self, 
        population: npt.NDArray[Any], 
        mutations: npt.NDArray[Any]
    ) -> None:
        """Apply add arm mutations to population."""
        add_mask = (mutations == 0)
        if not np.any(add_mask):
            return
        
        # Find individuals with empty slots
        empty_indices = np.isnan(population[:, :, 0])
        valid_add_mask = np.any(empty_indices, axis=1)
        add_mask &= valid_add_mask
        
        if not np.any(add_mask):
            return
        
        # Get available indices for adding
        available_indices = np.argwhere(empty_indices[add_mask])
        if len(available_indices) == 0:
            return
        
        # Select random indices
        selected_indices = self.rnd.choice(
            len(available_indices), 
            size=np.sum(add_mask), 
            replace=True
        )
        
        # Generate new arms
        num_new_arms = np.sum(add_mask)
        new_arms = np.zeros((num_new_arms, 7))
        new_arms[:, :6] = self.rnd.uniform(
            low=self.parameter_limits[:6, 0],
            high=self.parameter_limits[:6, 1],
            size=(num_new_arms, 6)
        )
        new_arms[:, 6] = self.rnd.integers(0, 2, size=num_new_arms)
        
        # Apply new arms
        for i, idx in enumerate(selected_indices):
            pop_idx, arm_idx = available_indices[idx]
            actual_pop_idx = np.where(add_mask)[0][pop_idx]
            population[actual_pop_idx, arm_idx] = new_arms[i]

    def _apply_remove_mutations(
        self, 
        population: npt.NDArray[Any], 
        mutations: npt.NDArray[Any]
    ) -> None:
        """Apply remove arm mutations to population."""
        remove_mask = (mutations == 1)
        if not np.any(remove_mask):
            return
        
        # Find individuals with non-empty arms
        non_empty_indices = ~np.isnan(population[:, :, 0])
        valid_remove_mask = np.any(non_empty_indices, axis=1)
        remove_mask &= valid_remove_mask
        
        if not np.any(remove_mask):
            return
        
        # Get available indices for removal
        available_indices = np.argwhere(non_empty_indices[remove_mask])
        if len(available_indices) == 0:
            return
        
        # Select random indices
        selected_indices = self.rnd.choice(
            len(available_indices), 
            size=np.sum(remove_mask), 
            replace=True
        )
        
        # Remove arms
        for idx in selected_indices:
            pop_idx, arm_idx = available_indices[idx]
            actual_pop_idx = np.where(remove_mask)[0][pop_idx]
            population[actual_pop_idx, arm_idx] = np.nan

    def _apply_parameter_mutations(
        self, 
        population: npt.NDArray[Any], 
        mutations: npt.NDArray[Any]
    ) -> None:
        """Apply parameter mutations to population."""
        perturb_mask = (mutations >= 2)
        if not np.any(perturb_mask):
            return
        
        drone_indices = np.where(perturb_mask)[0]
        
        # Find non-empty motors for each drone
        non_empty_mask = ~np.isnan(population[drone_indices, :, 0])
        num_non_empty = non_empty_mask.sum(axis=1)
        
        # Filter out drones with no motors
        valid_drones = num_non_empty > 0
        if not np.any(valid_drones):
            return
        
        valid_drone_indices = drone_indices[valid_drones]
        valid_non_empty_mask = non_empty_mask[valid_drones]
        valid_num_non_empty = num_non_empty[valid_drones]
        
        # Select random motor for each valid drone
        random_indices = self.rnd.integers(0, valid_num_non_empty)
        cumsum_non_empty = np.cumsum(valid_non_empty_mask, axis=1)
        selected_motors = np.argmax(
            cumsum_non_empty == (random_indices[:, None] + 1), 
            axis=1
        )
        
        # Get parameter indices
        selected_params = mutations[perturb_mask][valid_drones] - 2
        
        # Apply mutations
        for i, (drone_idx, motor_idx, param_idx) in enumerate(
            zip(valid_drone_indices, selected_motors, selected_params)
        ):
            if param_idx == 6:
                # Flip direction
                population[drone_idx, motor_idx, param_idx] = \
                    1 - population[drone_idx, motor_idx, param_idx]
            else:
                # Add perturbation
                perturbation = self.rnd.normal(0, self.mutation_scales[param_idx])
                population[drone_idx, motor_idx, param_idx] += perturbation
                
                # Clip to limits
                population[drone_idx, motor_idx, param_idx] = np.clip(
                    population[drone_idx, motor_idx, param_idx],
                    self.parameter_limits[param_idx, 0],
                    self.parameter_limits[param_idx, 1]
                )

    def copy(self) -> CartesianEulerDroneGenomeHandler:
        """
        Create a deep copy of this genome handler.
        
        Returns:
            Copy of this genome handler
        """
        return CartesianEulerDroneGenomeHandler(
            genome=self.genome.copy(),
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits.copy(),
            append_arm_chance=self.append_arm_chance,
            mutation_probs=None,  # Will use default
            mutation_scales_percentage=None,  # Will use default
            bilateral_plane_for_symmetry=self.bilateral_plane_for_symmetry,
            repair=self.repair_enabled,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance,
            rnd=self.rnd,
        )

    def is_valid(self) -> bool:
        """
        Check if the genome represents a valid drone configuration.
        
        Returns:
            True if valid, False otherwise
        """
        # Check that genome has the expected shape
        if self.genome.shape != (self.max_narms, 7):
            return False
        
        # Use the repair operator to validate the genome
        return self.repair_operator.validate(self.genome)

    def repair(self) -> None:
        """
        Repair the genome to make it valid by clipping out-of-bounds values.
        """
        # Use the repair operator to repair the genome
        self.genome = self.repair_operator.repair(self.genome)

    def get_motor_positions(self, include_nans=False) -> npt.NDArray[Any]:
        """
        Get the motor positions as a (valid_narms, 3) array.
        
        Returns:
            Array of motor positions [x, y, z] for valid arms only
        """
        if include_nans:
            # Return all positions including NaNs
            return self.genome[:, :3].copy()
        
        valid_arms_mask = ~np.isnan(self.genome[:, 0])
        return self.genome[valid_arms_mask, :3].copy()

    def get_motor_orientations(self) -> npt.NDArray[Any]:
        """
        Get the motor orientations as a (valid_narms, 3) array.
        
        Returns:
            Array of motor orientations [roll, pitch, yaw] for valid arms only
        """
        valid_arms_mask = ~np.isnan(self.genome[:, 0])
        return self.genome[valid_arms_mask, 3:6].copy()

    def get_propeller_directions(self) -> npt.NDArray[Any]:
        """
        Get the propeller directions as a (valid_narms,) array.
        
        Returns:
            Array of propeller directions (0 or 1) for valid arms only
        """
        valid_arms_mask = ~np.isnan(self.genome[:, 0])
        return self.genome[valid_arms_mask, 6].copy()

    def set_motor_position(self, arm_index: int, position: npt.NDArray[Any]) -> None:
        """
        Set the position of a specific motor.
        
        Args:
            arm_index: Index of the arm/motor
            position: New position [x, y, z]
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")
        
        if len(position) != 3:
            raise ValueError("position must have exactly 3 elements")
        
        self.genome[arm_index, :3] = position

    def set_motor_orientation(self, arm_index: int, orientation: npt.NDArray[Any]) -> None:
        """
        Set the orientation of a specific motor.
        
        Args:
            arm_index: Index of the arm/motor
            orientation: New orientation [roll, pitch, yaw]
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")
        
        if len(orientation) != 3:
            raise ValueError("orientation must have exactly 3 elements")
        
        self.genome[arm_index, 3:6] = orientation

    def set_propeller_direction(self, arm_index: int, direction: int) -> None:
        """
        Set the propeller direction of a specific motor.
        
        Args:
            arm_index: Index of the arm/motor
            direction: Propeller direction (0 or 1)
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")
        
        if direction not in [0, 1]:
            raise ValueError("direction must be 0 or 1")
        
        self.genome[arm_index, 6] = direction

    def apply_symmetry(self) -> None:
        """
        Apply bilateral symmetry to the genome based on the specified plane.
        
        This method enforces that the drone design is symmetric across the 
        specified plane by mirroring the first half of the arms to the second half.
        For odd number of arms, the middle arm remains unchanged.
        """
        if self.bilateral_plane_for_symmetry is None:
            return
        
        # Use the symmetry operator to apply symmetry
        self.genome = self.symmetry_operator.apply_symmetry(self.genome)

    def unapply_symmetry(self) -> None:
        """
        Remove bilateral symmetry from the genome.
        
        This method restores the genome to a non-symmetric state by removing 
        any enforced symmetry constraints.
        """
        if self.bilateral_plane_for_symmetry is None:
            return
        
        # Use the symmetry operator to remove symmetry
        self.genome = self.symmetry_operator.unapply_symmetry(self.genome)

    def repair_population(self, population):
        """
        Repair a population of genome handlers or population array.
        
        Args:
            population: List of genome handlers or population array to repair
            
        Returns:
            Repaired population (same type as input)
        """
        if isinstance(population, list):
            # Handle list of genome handlers
            repaired_population = []
            for individual in population:
                repaired_individual = individual.copy()
                repaired_individual.repair()
                repaired_population.append(repaired_individual)
            return repaired_population
        else:
            # Handle numpy array
            repaired_pop = self.repair_operator.repair_population(population)
            return repaired_pop

    def apply_symmetry_pop(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply symmetry to a population.
        
        Args:
            population: Population array to make symmetric
            
        Returns:
            Symmetric population array
        """
        if self.bilateral_plane_for_symmetry is None:
            return population
        
        # Use the symmetry operator to apply symmetry to population
        return self.symmetry_operator.apply_symmetry_population(population)

    def unapply_symmetry_pop(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Remove symmetry from a population.
        
        Args:
            population: Symmetric population array
            
        Returns:
            Population array with symmetry removed
        """
        if self.bilateral_plane_for_symmetry is None:
            return population
        
        # Use the symmetry operator to remove symmetry from population
        return self.symmetry_operator.unapply_symmetry_population(population)
    
    def validate_symmetry(self) -> bool:
        """
        Check if the genome satisfies symmetry constraints.
        
        Returns:
            True if genome satisfies symmetry constraints
        """
        if self.bilateral_plane_for_symmetry is None:
            return True
        return self.symmetry_operator.validate_symmetry(self.genome)
    
    def get_symmetry_pairs(self) -> List[tuple]:
        """
        Get pairs of indices that should be symmetric.
        
        Returns:
            List of (source_index, target_index) tuples
        """
        return self.symmetry_operator.get_symmetry_pairs(self.genome)
    
    def get_arm_count(self) -> int:
        """
        Get the number of valid arms in the genome.
        
        Returns:
            Number of valid (non-NaN) arms
        """
        valid_arms_mask = ~np.isnan(self.genome[:, 0])
        return np.sum(valid_arms_mask)

    def get_max_arm_count(self) -> int:
        """
        Get the maximum number of arms this genome can support.
        
        Returns:
            Maximum number of arms
        """
        return self.max_narms

    def get_valid_arms(self) -> npt.NDArray[Any]:
        """
        Get only the valid (non-NaN) arms from the genome.
        
        Returns:
            Array of valid arms with shape (num_valid_arms, 7)
        """
        valid_mask = ~np.isnan(self.genome[:, 0])
        return self.genome[valid_mask].copy()

    def add_random_arm(self) -> bool:
        """
        Add a random arm to an empty slot.
        
        Returns:
            True if arm was added, False if no empty slots available
        """
        empty_mask = np.isnan(self.genome[:, 0])
        if not np.any(empty_mask):
            return False
        
        empty_indices = np.where(empty_mask)[0]
        selected_index = self.rnd.choice(empty_indices)
        
        new_arm = np.zeros(7)
        new_arm[:6] = self.rnd.uniform(
            low=self.parameter_limits[:6, 0],
            high=self.parameter_limits[:6, 1]
        )
        new_arm[6] = self.rnd.integers(0, 2)
        
        self.genome[selected_index] = new_arm
        return True

    def remove_arm(self, arm_index: int) -> None:
        """
        Remove an arm by setting it to NaN.
        
        Args:
            arm_index: Index of the arm to remove
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")
        
        self.genome[arm_index] = np.nan

    def compact_genome(self) -> npt.NDArray[Any]:
        """
        Get a compact representation with only valid arms.
        
        Returns:
            Array containing only non-NaN arms
        """
        return self.get_valid_arms()

    def __str__(self) -> str:
        """String representation of the genome handler."""
        symmetry_str = f", symmetry={self.bilateral_plane_for_symmetry}" if self.bilateral_plane_for_symmetry else ""
        return f"CartesianEulerDroneGenomeHandler(max_narms={self.max_narms}, shape={self.genome.shape}{symmetry_str})"

    def __repr__(self) -> str:
        """Detailed string representation of the genome handler."""
        return (f"CartesianEulerDroneGenomeHandler("
                f"min_max_narms=({self.min_narms}, {self.max_narms}), "
                f"append_arm_chance={self.append_arm_chance}, "
                f"bilateral_plane_for_symmetry={self.bilateral_plane_for_symmetry}, "
                f"repair={self.repair_enabled}, "
                f"genome_shape={self.genome.shape})")