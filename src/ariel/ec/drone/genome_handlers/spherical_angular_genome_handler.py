from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import numpy as np
import numpy.typing as npt

from .base import GenomeHandler
from .cppn.innovation import InnovationCounter
from .operators import (
    SphericalSymmetryOperator,
    SphericalRepairOperator,
    SymmetryConfig,
    RepairConfig
)


@dataclass
class SphericalNeatGenome:
    """Wraps a spherical arm array with NEAT-style innovation IDs.

    Attributes:
        arms: (max_narms, 6) array with NaN padding for empty slots.
        innovation_ids: (max_narms,) int array; -1 marks empty slots.
    """

    arms: npt.NDArray[Any]
    innovation_ids: npt.NDArray[Any]

    def copy(self) -> SphericalNeatGenome:
        return SphericalNeatGenome(
            arms=self.arms.copy(),
            innovation_ids=self.innovation_ids.copy(),
        )


class SphericalAngularDroneGenomeHandler(GenomeHandler):
    """
    Genome handler for drone designs using spherical coordinates and angular orientations.
    
    Genome format: (narms, 6) array with columns:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
    
    Features:
    - Dynamic arm addition/removal during mutation
    - Optional symmetry enforcement
    - Vectorized operations for efficiency
    - Collision repair capabilities
    - Variable population sizes with NaN masking
    """

    _innovation_counter: InnovationCounter = InnovationCounter()

    def __init__(
        self,
        genome: npt.NDArray[Any] | SphericalNeatGenome | None = None,
        min_max_narms: Tuple[int, int] | None = None,
        parameter_limits: npt.NDArray[Any] | None = None,
        append_arm_chance: float = 0.1,
        mutation_probs: Optional[List[float] | npt.NDArray[Any]] = None,
        mutation_scales_percentage: Optional[npt.NDArray[Any]] = None,
        bilateral_plane_for_symmetry: str | None = None,
        repair: bool = False,
        enable_collision_repair: bool = False,
        propeller_radius: float = 0.0508/2,  # 2-inch propeller radius in meters
        inner_boundary_radius: float = 0.0055,
        outer_boundary_radius: float = 0.11,
        max_repair_iterations: int = 100,
        repair_step_size: float = 1.0,
        propeller_tolerance: float = 0.1,
        rnd: np.random.Generator | None = None,
    ) -> None:
        """
        Initialize the Spherical Angular genome handler.
        
        Args:
            genome: Pre-existing genome array of shape (max_narms, 6)
            min_max_narms: Tuple of (min_arms, max_arms). If None, defaults to (3, 8)
            parameter_limits: Array of shape (6, 2) with [min, max] for each parameter. If None, uses defaults
            append_arm_chance: Probability of adding/removing arms during mutation
            mutation_probs: Custom mutation probabilities for each parameter
            mutation_scales_percentage: Mutation scales as percentage of parameter ranges
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
            self.min_narms, self.max_narms = 3, 8  # Default range for spherical
        else:
            self.min_narms, self.max_narms = min_max_narms
        
        # Setup parameter limits
        if parameter_limits is None:
            # Default limits: [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
            # arm_pitch uses elevation convention: 0 = horizontal,
            # +π/2 = straight up, −π/2 = straight down.
            self.parameter_limits = np.array([
                [0.055, 0.17],           # magnitude
                [-np.pi, np.pi],         # arm rotation (azimuth)
                [-np.pi/2, np.pi/2],     # arm pitch (elevation)
                [-np.pi, np.pi],         # motor rotation (azimuth)
                [-np.pi, np.pi],         # motor pitch
                [0, 1]                   # direction
            ])
        else:
            self.parameter_limits = np.asarray(parameter_limits)
            if self.parameter_limits.shape != (6, 2):
                raise ValueError("parameter_limits must have shape (6, 2)")
        
        # Validate inputs
        self._validate_initialization_parameters(
            (self.min_narms, self.max_narms), self.parameter_limits, append_arm_chance, 
            mutation_probs, mutation_scales_percentage
        )
        
        self.append_arm_chance = append_arm_chance
        
        # Handle bilateral symmetry parameter
        valid_planes = {"xy", "xz", "yz", None}
        if bilateral_plane_for_symmetry not in valid_planes:
            raise ValueError(f"bilateral_plane_for_symmetry must be one of {valid_planes}")
        self.bilateral_plane_for_symmetry = bilateral_plane_for_symmetry
        self.symmetry = bilateral_plane_for_symmetry is not None
        
        self.repair_enabled = repair
        
        # Store collision repair parameters
        self.enable_collision_repair = enable_collision_repair
        self.propeller_radius = propeller_radius
        self.inner_boundary_radius = inner_boundary_radius
        self.outer_boundary_radius = outer_boundary_radius
        self.max_repair_iterations = max_repair_iterations
        self.repair_step_size = repair_step_size
        self.propeller_tolerance = propeller_tolerance
        
        # Setup mutation parameters
        self._setup_mutation_parameters(mutation_probs, mutation_scales_percentage)
        
        # Initialize operators
        self._setup_operators()
        
        # Initialize genome — handle SphericalNeatGenome, ndarray, or None
        self.fitness: float | None = None
        if isinstance(genome, SphericalNeatGenome):
            self.genome = genome.copy()
        elif genome is not None:
            # Raw ndarray — wrap with sequential innovation IDs for
            # backward compatibility with CMA-ES / mu+lambda strategies.
            arms = genome.copy()
            valid_mask = ~np.isnan(arms[:, 0])
            inno = np.full(arms.shape[0], -1, dtype=int)
            inno[valid_mask] = np.arange(int(valid_mask.sum()))
            self.genome = SphericalNeatGenome(arms=arms, innovation_ids=inno)
        else:
            self.genome = self._generate_random_genome()

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
        if parameter_limits.shape != (6, 2):
            raise ValueError("parameter_limits must have shape (6, 2)")
        if np.any(parameter_limits[:, 0] >= parameter_limits[:, 1]):
            raise ValueError("parameter_limits: min values must be < max values")
        
        if mutation_probs is not None:
            mutation_probs = np.asarray(mutation_probs)
            if len(mutation_probs) != 6:
                raise ValueError("mutation_probs must have length 6")
            if np.any(mutation_probs < 0):
                raise ValueError("mutation_probs must be non-negative")
        
        if mutation_scales_percentage is not None:
            mutation_scales_percentage = np.asarray(mutation_scales_percentage)
            if len(mutation_scales_percentage) != 6:
                raise ValueError("mutation_scales_percentage must have length 6")
            if np.any(mutation_scales_percentage < 0):
                raise ValueError("mutation_scales_percentage must be non-negative")

    def _setup_mutation_parameters(
        self,
        mutation_probs: Optional[List[float] | npt.NDArray[Any]],
        mutation_scales_percentage: Optional[npt.NDArray[Any]]
    ) -> None:
        """Setup mutation probabilities and scales."""
        nparms = 6
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
            mutation_scales_percentage = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.0])
        
        # Convert percentages to absolute scales
        parameter_ranges = self.parameter_limits[:, 1] - self.parameter_limits[:, 0]
        self.mutation_scales = parameter_ranges * mutation_scales_percentage

    def _setup_operators(self) -> None:
        """Initialize the symmetry and repair operators."""
        # Setup symmetry operator
        from .operators.symmetry_base import SymmetryPlane
        
        # Map bilateral plane string to SymmetryPlane enum
        plane_mapping = {
            "xy": SymmetryPlane.XY,
            "xz": SymmetryPlane.XZ, 
            "yz": SymmetryPlane.YZ,
            None: None
        }
        
        symmetry_config = SymmetryConfig(
            plane=plane_mapping.get(self.bilateral_plane_for_symmetry),
            enabled=self.symmetry
        )
        self.symmetry_operator = SphericalSymmetryOperator(
            config=symmetry_config,
        )
        
        # Setup repair operator
        repair_config = RepairConfig(
            apply_symmetry=self.symmetry,
            enable_collision_repair=self.enable_collision_repair,
            propeller_radius=self.propeller_radius,
            inner_boundary_radius=self.inner_boundary_radius,
            outer_boundary_radius=self.outer_boundary_radius,
            max_repair_iterations=self.max_repair_iterations,
            repair_step_size=self.repair_step_size,
            propeller_tolerance=self.propeller_tolerance
        )
        self.repair_operator = SphericalRepairOperator(
            config=repair_config,
            min_narms=self.min_narms,
            max_narms=self.max_narms,
            parameter_limits=self.parameter_limits,
            symmetry_operator=self.symmetry_operator,
            rng=self.rnd
        )

    def _generate_random_genome(self, innovation_ids: npt.NDArray[Any] | None = None) -> SphericalNeatGenome:
        """Generate a single random genome.

        Parameters
        ----------
        innovation_ids : array, optional
            Pre-assigned innovation IDs for the initial arms.  When
            generating the initial population all individuals share the
            same IDs so that NEAT crossover alignment works from gen 0.
            If *None*, new IDs are drawn from the class-level counter.
        """
        arms = np.full((self.max_narms, 6), np.nan)
        inno = np.full(self.max_narms, -1, dtype=int)

        # Determine number of arms for this individual
        num_arms = self.rnd.integers(self.min_narms, self.max_narms + 1)

        # Generate random parameters for the arms (excluding phi)
        for i in [0, 1, 3, 4]:  # r, theta, pitch, yaw
            arms[:num_arms, i] = self.rnd.uniform(
                low=self.parameter_limits[i, 0],
                high=self.parameter_limits[i, 1],
                size=num_arms
            )

        # For phi (index 2), sample to achieve uniform spatial distribution on sphere.
        # phi is an elevation angle (0 = horizontal, π/2 = up, −π/2 = down).
        # Uniform sphere coverage requires sampling sin(phi) uniformly and inverting
        # with arcsin.  This is valid for any elevation range including negatives.
        phi_min, phi_max = self.parameter_limits[2, 0], self.parameter_limits[2, 1]
        sin_phi = self.rnd.uniform(low=np.sin(phi_min), high=np.sin(phi_max), size=num_arms)
        arms[:num_arms, 2] = np.arcsin(sin_phi)

        arms[:num_arms, 5] = self.rnd.integers(0, 2, size=num_arms)

        # Assign innovation IDs
        if innovation_ids is not None:
            inno[:num_arms] = innovation_ids[:num_arms]
        else:
            for j in range(num_arms):
                inno[j] = self._innovation_counter.next_innovation()

        # If symmetry is enabled, apply symmetry to the genome
        if self.symmetry:
            # Reduce to half if symmetry is enabled
            arms[self.max_narms // 2:] = np.nan
            inno[self.max_narms // 2:] = -1

        if self.symmetry:
            arms = self.symmetry_operator.apply_symmetry(arms)

        if self.repair_enabled:
            arms = self.repair_operator.repair(arms)

        return SphericalNeatGenome(arms=arms, innovation_ids=inno)

    def generate_random_population(self, population_size: int, as_nparray : bool = False) -> List[SphericalAngularDroneGenomeHandler]:
        """
        Generate a population of random genome handlers.

        All individuals share the same innovation IDs for their starting
        arms so that NEAT crossover alignment works from generation 0.

        Args:
            population_size: Number of individuals to generate

        Returns:
            List of random genome handler instances
        """
        # Pre-allocate shared innovation IDs for the starting topology
        shared_innos = np.array(
            [self._innovation_counter.next_innovation() for _ in range(self.max_narms)],
            dtype=int,
        )

        population = []
        for _ in range(population_size):
            individual = SphericalAngularDroneGenomeHandler(
                genome=None,
                min_max_narms=(self.min_narms, self.max_narms),
                parameter_limits=self.parameter_limits,
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
            individual.genome = individual._generate_random_genome(
                innovation_ids=shared_innos,
            )
            population.append(individual)

        return population

    def random_population(self, pop_size: int) -> npt.NDArray[Any]:
        """
        Generate a random population as a numpy array (vectorized version).

        Args:
            pop_size: Size of the population to generate

        Returns:
            Population array of shape (pop_size, max_narms, 6)
        """
        array_shape = (pop_size, self.max_narms, 6)
        population = np.empty(array_shape)

        # Generate random arms for all individuals
        # For parameters other than phi (index 2), sample uniformly
        for i in [0, 1, 3, 4]:  # r, theta, pitch, yaw
            population[:, :, i] = self.rnd.uniform(
                low=self.parameter_limits[i, 0],
                high=self.parameter_limits[i, 1],
                size=(pop_size, self.max_narms)
            )

        # For phi (index 2), sample to achieve uniform spatial distribution on sphere.
        # phi is an elevation angle (0 = horizontal, π/2 = up, −π/2 = down).
        # Uniform sphere coverage requires sampling sin(phi) uniformly and inverting
        # with arcsin.  This is valid for any elevation range including negatives.
        phi_min, phi_max = self.parameter_limits[2, 0], self.parameter_limits[2, 1]
        sin_phi = self.rnd.uniform(
            low=np.sin(phi_min),
            high=np.sin(phi_max),
            size=(pop_size, self.max_narms)
        )
        population[:, :, 2] = np.arcsin(sin_phi)

        population[:, :, 5] = self.rnd.integers(0, 2, size=(pop_size, self.max_narms))
        
        if self.symmetry:
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
        
        if self.symmetry:
            # Apply symmetry to the population
            population = self.symmetry_operator.apply_symmetry_population(population)
        # Apply post-processing
        if self.repair_enabled:
            population = self.repair_population(population)
        return population

    def crossover(self, other: SphericalAngularDroneGenomeHandler) -> SphericalAngularDroneGenomeHandler:
        """NEAT-style crossover using innovation-number gene alignment.

        Matching genes (same innovation ID in both parents) are inherited
        randomly.  Disjoint / excess genes are inherited from the fitter
        parent; when fitness is equal each disjoint gene has a 50 %
        chance of inclusion.
        """
        if not isinstance(other, SphericalAngularDroneGenomeHandler):
            raise TypeError("Other parent must be SphericalAngularDroneGenomeHandler")

        # Determine fitter parent (self.fitness / other.fitness set by NEAT loop)
        f1 = self.fitness if self.fitness is not None else 0.0
        f2 = other.fitness if other.fitness is not None else 0.0
        equal_fitness = np.isclose(f1, f2)
        self_is_fitter = f1 > f2

        # Build {innovation_id: arm_params} dicts for each parent
        g1_arms, g1_inno = self.genome.arms, self.genome.innovation_ids
        g2_arms, g2_inno = other.genome.arms, other.genome.innovation_ids

        d1 = {int(inno): g1_arms[i] for i, inno in enumerate(g1_inno) if inno >= 0}
        d2 = {int(inno): g2_arms[i] for i, inno in enumerate(g2_inno) if inno >= 0}

        all_innos = sorted(set(d1.keys()) | set(d2.keys()))

        child_arms_list: list[tuple[int, npt.NDArray[Any]]] = []
        for inno in all_innos:
            in1, in2 = inno in d1, inno in d2
            if in1 and in2:
                # Matching gene — inherit randomly
                child_arms_list.append(
                    (inno, d1[inno].copy() if self.rnd.random() < 0.5 else d2[inno].copy())
                )
            elif in1 and not in2:
                # Only in self
                if self_is_fitter or (equal_fitness and self.rnd.random() < 0.5):
                    child_arms_list.append((inno, d1[inno].copy()))
            else:
                # Only in other
                if not self_is_fitter or (equal_fitness and self.rnd.random() < 0.5):
                    child_arms_list.append((inno, d2[inno].copy()))

        # Enforce min/max arm constraints
        if len(child_arms_list) > self.max_narms:
            child_arms_list = child_arms_list[:self.max_narms]
        while len(child_arms_list) < self.min_narms:
            # Fill from fitter parent's genes not yet included
            source = d1 if self_is_fitter or equal_fitness else d2
            for k, v in source.items():
                if k not in {x[0] for x in child_arms_list}:
                    child_arms_list.append((k, v.copy()))
                    break
            else:
                break

        # Pack into (max_narms, 6) array
        child_arr = np.full((self.max_narms, 6), np.nan)
        child_inno = np.full(self.max_narms, -1, dtype=int)
        for idx, (inno, params) in enumerate(child_arms_list[:self.max_narms]):
            child_arr[idx] = params
            child_inno[idx] = inno

        # Symmetry / repair
        if self.symmetry:
            child_arr = self._apply_symmetry_single(child_arr)
        if self.repair_enabled:
            child_arr = self.repair_operator.repair(child_arr)

        child_genome = SphericalNeatGenome(arms=child_arr, innovation_ids=child_inno)

        child = SphericalAngularDroneGenomeHandler(
            genome=child_genome,
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits,
            append_arm_chance=self.append_arm_chance,
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
            population: Population array of shape (pop_size, max_narms, 6)
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
        
        if self.symmetry:
            # If symmetry is enabled, apply symmetry to the children
            children = self.apply_symmetry_pop(children)
        # Apply repair if needed
        if self.repair_enabled:
            children = self.repair_population(children)
        
        return children

    def mutate(self, genome=None) -> None:
        """Mutate this genome in place."""
        if genome is not None:
            if isinstance(genome, SphericalNeatGenome):
                self.genome = genome
            else:
                # Raw ndarray passed in — wrap it
                arms = genome
                valid_mask = ~np.isnan(arms[:, 0])
                inno = np.full(arms.shape[0], -1, dtype=int)
                inno[valid_mask] = np.arange(int(valid_mask.sum()))
                self.genome = SphericalNeatGenome(arms=arms, innovation_ids=inno)

        arms = self.genome.arms
        inno = self.genome.innovation_ids

        if self.symmetry:
            # Temporarily remove symmetry for mutation
            arms = self._unapply_symmetry_single(arms)
        else:
            arms = arms.copy()

        # Choose mutation type
        mutation_type = self.rnd.choice(
            len(self.mutation_probabilities),
            p=self.mutation_probabilities
        )

        if mutation_type == 0:
            # Add arm mutation — assign new innovation ID
            empty_mask = np.isnan(arms[:, 0])
            arms = self._mutate_add_arm(arms)
            # Find newly filled slot
            new_filled = np.isnan(self.genome.arms[:, 0]) if not self.symmetry else empty_mask
            for i in range(len(arms)):
                if empty_mask[i] and not np.isnan(arms[i, 0]):
                    inno[i] = self._innovation_counter.next_innovation()
                    break
        elif mutation_type == 1:
            # Remove arm mutation — set innovation ID to -1
            non_empty_before = ~np.isnan(arms[:, 0])
            arms = self._mutate_remove_arm(arms)
            for i in range(len(arms)):
                if non_empty_before[i] and np.isnan(arms[i, 0]):
                    inno[i] = -1
                    break
        else:
            # Parameter mutation
            param_index = mutation_type - 2
            arms = self._mutate_parameter(arms, param_index)

        # Ensure genome is valid has the correct number of arms for symmetry
        if self.symmetry:
            arms = self._apply_symmetry_single(arms)

        self.genome = SphericalNeatGenome(arms=arms, innovation_ids=inno)

        if self.repair_enabled:
            self.repair()

        return self.genome
    
    def mutation_vectorized(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Vectorized mutation operation for efficiency.
        
        Args:
            population: Population array to mutate
            
        Returns:
            Mutated population array
        """
        if self.symmetry:
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
        
        if self.symmetry:
            pop = self.apply_symmetry_pop(pop)
        # Apply post-processing
        if self.repair_enabled:
            pop = self.repair_population(pop)
        
        return pop

    def _mutate_add_arm(self, genome: npt.NDArray[Any]) -> None:
        """Add a random arm to the genome."""
        # Find empty slots (NaN values)
        empty_mask = np.isnan(genome[:, 0])
        
        if not np.any(empty_mask) or (self.symmetry and not np.any(empty_mask[:self.max_narms // 2])):
            return  genome # No empty slots available

        # Select random empty slot
        empty_indices = np.where(empty_mask)[0]
        selected_index = self.rnd.choice(empty_indices)
        
        # Generate new arm parameters
        new_arm = np.zeros(6)
        new_arm[:5] = self.rnd.uniform(
            low=self.parameter_limits[:5, 0],
            high=self.parameter_limits[:5, 1]
        )
        new_arm[5] = self.rnd.integers(0, 2)
        
        genome[selected_index] = new_arm

        return genome

    def _mutate_remove_arm(self, genome: npt.NDArray[Any]) -> None:
        """Remove a random arm from the genome."""
        # Find non-empty slots
        non_empty_mask = ~np.isnan(genome[:, 0])
        if not np.any(non_empty_mask):
            return  genome # No arms to remove
        
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
            return genome # No arms to mutate
        
        # Select random arm
        non_empty_indices = np.where(non_empty_mask)[0]
        selected_arm = self.rnd.choice(non_empty_indices)
        
        if param_index == 5:
            # Flip direction parameter
            genome[selected_arm, param_index] = 1 - genome[selected_arm, param_index]
        else:
            # Add Gaussian perturbation
            perturbation = self.rnd.normal(0, self.mutation_scales[param_index])
            genome[selected_arm, param_index] += perturbation
            
            # Handle angular parameters (wrap around)
            if param_index in [1, 2, 3, 4]:  # Angular parameters
                genome[selected_arm, param_index] = self._wrap_angle(
                    genome[selected_arm, param_index],
                    self.parameter_limits[param_index, 0],
                    self.parameter_limits[param_index, 1],
                )
            else:
                # Clip non-angular parameters to limits
                genome[selected_arm, param_index] = np.clip(
                    genome[selected_arm, param_index],
                    self.parameter_limits[param_index, 0],
                    self.parameter_limits[param_index, 1]
                )
        
        return genome

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
        new_arms = np.zeros((num_new_arms, 6))
        new_arms[:, :5] = self.rnd.uniform(
            low=self.parameter_limits[:5, 0],
            high=self.parameter_limits[:5, 1],
            size=(num_new_arms, 5)
        )
        new_arms[:, 5] = self.rnd.integers(0, 2, size=num_new_arms)
        
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
            if param_idx == 5:
                # Flip direction
                population[drone_idx, motor_idx, param_idx] = \
                    1 - population[drone_idx, motor_idx, param_idx]
            else:
                # Add perturbation
                perturbation = self.rnd.normal(0, self.mutation_scales[param_idx])
                population[drone_idx, motor_idx, param_idx] += perturbation
                
                # Handle angular parameters
                if param_idx in [1, 2, 3, 4]:
                    population[drone_idx, motor_idx, param_idx] = self._wrap_angle(
                        population[drone_idx, motor_idx, param_idx],
                        self.parameter_limits[param_idx, 0],
                        self.parameter_limits[param_idx, 1],
                    )
                else:
                    # Clip non-angular parameters to limits
                    population[drone_idx, motor_idx, param_idx] = np.clip(
                        population[drone_idx, motor_idx, param_idx],
                        self.parameter_limits[param_idx, 0],
                        self.parameter_limits[param_idx, 1]
                    )

    def _wrap_angle(self, angle: float, low: float, high: float) -> float:
        """Wrap angle to [low, high) range."""
        span = high - low
        return low + (angle - low) % span

    def copy(self) -> SphericalAngularDroneGenomeHandler:
        """
        Create a deep copy of this genome handler.
        
        Returns:
            Copy of this genome handler
        """
        return SphericalAngularDroneGenomeHandler(
            genome=self.genome.copy(),
            min_max_narms=(self.min_narms, self.max_narms),
            parameter_limits=self.parameter_limits.copy(),
            append_arm_chance=self.append_arm_chance,
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
        # Use the repair operator to validate the genome
        return self.repair_operator.validate(self.genome.arms)

    def compatibility_distance(self, other: SphericalAngularDroneGenomeHandler) -> float:
        """Phenotypic compatibility distance using edit distance.

        Uses Hungarian-algorithm optimal arm matching on normalised
        parameters plus an arm-count penalty.  This is permutation-
        invariant and produces meaningful distances even when all
        genomes share the same innovation IDs (fixed arm count).
        """
        from ariel.ec.drone.evaluators.edit_distance import (
            compute_edit_distance,
        )

        return float(
            compute_edit_distance(
                self.genome.arms,
                other.genome.arms,
                min_vals=self.parameter_limits[:, 0],
                max_vals=self.parameter_limits[:, 1],
            )
        )

    def repair(self) -> None:
        """
        Repair the genome to make it valid by clipping out-of-bounds values.
        """
        # Use the repair operator on the arms array, preserving innovation IDs
        self.genome.arms = self.repair_operator.repair(self.genome.arms)

    def repair_population(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a population array.
        
        Args:
            population: Population array to repair
            
        Returns:
            Repaired population array
        """
        # This is a placeholder - implement based on your specific repair logic
        # For now, just clip parameters to bounds
        pop = population.copy()
        
        repaired_pop = self.repair_operator.repair_population(pop)
        
        return repaired_pop

    def apply_symmetry_pop(self, population: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Apply symmetry to a population.
        
        Args:
            population: Population array to make symmetric
            
        Returns:
            Symmetric population array
        """
        if not self.symmetry:
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
        if not self.symmetry:
            return population
        
        # Use the symmetry operator to remove symmetry from population
        return self.symmetry_operator.unapply_symmetry_population(population)

    def _apply_symmetry_single(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply symmetry to a single genome."""
        if not self.symmetry:
            return genome
        
        # Use the symmetry operator to apply symmetry
        return self.symmetry_operator.apply_symmetry(genome)

    def _unapply_symmetry_single(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Remove symmetry from a single genome."""
        if not self.symmetry:
            return genome
        
        # Use the symmetry operator to remove symmetry
        return self.symmetry_operator.unapply_symmetry(genome)

    def validate_symmetry(self) -> bool:
        """
        Check if the genome satisfies symmetry constraints.
        
        Returns:
            True if genome satisfies symmetry constraints
        """
        if not self.symmetry:
            return True
        return self.symmetry_operator.validate_symmetry(self.genome.arms)

    def get_symmetry_pairs(self) -> List[tuple]:
        """
        Get pairs of indices that should be symmetric.

        Returns:
            List of (source_index, target_index) tuples
        """
        return self.symmetry_operator.get_symmetry_pairs(self.genome.arms)

    def apply_symmetry(self) -> None:
        """
        Apply symmetry to the current genome.
        """
        if self.symmetry:
            self.genome.arms = self.symmetry_operator.apply_symmetry(self.genome.arms)

    def unapply_symmetry(self) -> None:
        """
        Remove symmetry from the current genome (keep only first half).
        """
        if self.symmetry:
            self.genome.arms = self.symmetry_operator.unapply_symmetry(self.genome.arms)

    def get_valid_arms(self) -> npt.NDArray[Any]:
        """
        Get only the valid (non-NaN) arms from the genome.

        Returns:
            Array of valid arms with shape (num_valid_arms, 6)
        """
        valid_mask = ~np.isnan(self.genome.arms[:, 0])
        return self.genome.arms[valid_mask].copy()

    def get_arm_count(self) -> int:
        """
        Get the number of valid arms in the genome.

        Returns:
            Number of valid arms
        """
        valid_mask = ~np.isnan(self.genome.arms[:, 0])
        return int(np.sum(valid_mask))

    def get_spherical_coordinates(self) -> npt.NDArray[Any]:
        """
        Get spherical coordinates of valid arms.
        
        Returns:
            Array of shape (num_valid_arms, 3) with [magnitude, arm_rotation, arm_pitch]
        """
        valid_arms = self.get_valid_arms()
        return valid_arms[:, :3].copy()

    def get_motor_orientations(self) -> npt.NDArray[Any]:
        """
        Get motor orientations of valid arms.
        
        Returns:
            Array of shape (num_valid_arms, 2) with [motor_rotation, motor_pitch]
        """
        valid_arms = self.get_valid_arms()
        return valid_arms[:, 3:5].copy()

    def get_propeller_directions(self) -> npt.NDArray[Any]:
        """
        Get propeller directions of valid arms.
        
        Returns:
            Array of shape (num_valid_arms,) with direction values
        """
        valid_arms = self.get_valid_arms()
        return valid_arms[:, 5].copy()

    def set_arm(self, arm_index: int, arm_data: npt.NDArray[Any]) -> None:
        """
        Set data for a specific arm.
        
        Args:
            arm_index: Index of the arm to set
            arm_data: Array of 6 parameters for the arm
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")
        
        if len(arm_data) != 6:
            raise ValueError("arm_data must have exactly 6 elements")
        
        self.genome.arms[arm_index] = arm_data

    def remove_arm(self, arm_index: int) -> None:
        """
        Remove an arm by setting it to NaN.

        Args:
            arm_index: Index of the arm to remove
        """
        if not (0 <= arm_index < self.max_narms):
            raise ValueError(f"arm_index must be between 0 and {self.max_narms-1}")

        self.genome.arms[arm_index] = np.nan
        self.genome.innovation_ids[arm_index] = -1

    def add_random_arm(self) -> bool:
        """
        Add a random arm to an empty slot.

        Returns:
            True if arm was added, False if no empty slots available
        """
        empty_mask = np.isnan(self.genome.arms[:, 0])
        if not np.any(empty_mask):
            return False

        empty_indices = np.where(empty_mask)[0]
        selected_index = self.rnd.choice(empty_indices)

        new_arm = np.zeros(6)
        new_arm[:5] = self.rnd.uniform(
            low=self.parameter_limits[:5, 0],
            high=self.parameter_limits[:5, 1]
        )
        new_arm[5] = self.rnd.integers(0, 2)

        self.genome.arms[selected_index] = new_arm
        self.genome.innovation_ids[selected_index] = self._innovation_counter.next_innovation()
        return True

    def compact_genome(self) -> npt.NDArray[Any]:
        """
        Get a compact representation with only valid arms.
        
        Returns:
            Array containing only non-NaN arms
        """
        return self.get_valid_arms()

    def __str__(self) -> str:
        """String representation of the genome handler."""
        num_arms = self.get_arm_count()
        return (f"SphericalAngularDroneGenomeHandler("
                f"arms={num_arms}/{self.max_narms}, "
                f"bilateral_plane_for_symmetry={self.bilateral_plane_for_symmetry}, "
                f"repair={self.repair_enabled})")

    def __repr__(self) -> str:
        """Detailed string representation of the genome handler."""
        return (f"SphericalAngularDroneGenomeHandler("
                f"min_max_narms=({self.min_narms}, {self.max_narms}), "
                f"append_arm_chance={self.append_arm_chance}, "
                f"bilateral_plane_for_symmetry={self.bilateral_plane_for_symmetry}, "
                f"repair={self.repair_enabled}, "
                f"current_arms={self.get_arm_count()})")


# Utility functions for backward compatibility
def create_spherical_angular_handler(
    min_narms: int = 3,
    max_narms: int = 8,
    magnitude_range: Tuple[float, float] = (0.5, 2.0),
    angle_range: Tuple[float, float] = (0.0, 2*np.pi),
    **kwargs
) -> SphericalAngularDroneGenomeHandler:
    """
    Convenience function to create a SphericalAngularDroneGenomeHandler with common parameters.
    
    Args:
        min_narms: Minimum number of arms
        max_narms: Maximum number of arms
        magnitude_range: Range for magnitude parameter
        angle_range: Range for angular parameters
        **kwargs: Additional arguments passed to the constructor
        
    Returns:
        Configured SphericalAngularDroneGenomeHandler instance
    """
    # Default parameter limits: [magnitude, arm_rot, arm_pitch, motor_rot, motor_pitch, direction]
    parameter_limits = np.array([
        [magnitude_range[0], magnitude_range[1]],  # magnitude
        [angle_range[0], angle_range[1]],          # arm rotation
        [angle_range[0], angle_range[1]],          # arm pitch
        [angle_range[0], angle_range[1]],          # motor rotation
        [angle_range[0], angle_range[1]],          # motor pitch
        [0, 1]                                     # direction (discrete)
    ])
    
    return SphericalAngularDroneGenomeHandler(
        genome=None,
        min_max_narms=(min_narms, max_narms),
        parameter_limits=parameter_limits,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("=== SphericalAngularDroneGenomeHandler Testing ===")
    
    # Create parameter limits
    parameter_limits = np.array([
        [0.5, 2.0],      # magnitude
        [0.0, 2*np.pi],  # arm rotation
        [0.0, np.pi],    # arm pitch
        [0.0, 2*np.pi],  # motor rotation
        [0.0, np.pi],    # motor pitch
        [0, 1]           # direction
    ])
    
    # Create genome handler
    handler = SphericalAngularDroneGenomeHandler(
        genome=None,
        min_max_narms=(3, 6),
        parameter_limits=parameter_limits,
        append_arm_chance=0.1,
        bilateral_plane_for_symmetry=None,
        repair=True
    )
    
    print(f"Created handler: {handler}")
    
    # Generate random individual
    individual = handler.copy()
    individual.genome = individual._generate_random_genome()
    
    print(f"Generated individual with {individual.get_arm_count()} arms")
    print(f"Valid: {individual.is_valid()}")
    
    # Test mutations
    print("\n--- Testing Mutations ---")
    original_arms = individual.get_arm_count()
    
    for i in range(5):
        individual.mutate()
        new_arms = individual.get_arm_count()
        print(f"Mutation {i+1}: {original_arms} -> {new_arms} arms")
        original_arms = new_arms
    
    # Test population operations
    print("\n--- Testing Population Operations ---")
    population = handler.random_population(10)
    print(f"Generated population shape: {population.shape}")
    
    arm_counts = [np.sum(~np.isnan(pop[:, 0])) for pop in population]
    print(f"Arm counts: {arm_counts}")
    
    # Test crossover
    print("\n--- Testing Crossover ---")
    parent1 = handler.copy()
    parent1.genome = handler._generate_random_genome()
    parent2 = handler.copy()
    parent2.genome = handler._generate_random_genome()
    
    child = parent1.crossover(parent2)
    print(f"Parent 1: {parent1.get_arm_count()} arms")
    print(f"Parent 2: {parent2.get_arm_count()} arms")
    print(f"Child: {child.get_arm_count()} arms")
    