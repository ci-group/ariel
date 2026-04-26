"""
Incubation Phase for Spatial EA.

This module implements a non-spatial evolutionary algorithm that trains robots
for basic locomotion before they enter the spatial EA. The incubation phase:

1. Uses simple generational GA (no spatial constraints)
2. Focuses solely on fitness maximization (distance traveled)
3. Applies standard selection, crossover, and mutation
4. Bootstraps movement capabilities before spatial interactions

This helps ensure robots have basic locomotion skills before being subjected
to the additional challenges of energy management, mating zones, and spatial
competition.
"""

import mujoco
import numpy as np
import copy
import time
import random
from hyperneat import CPPNNode, CPPNConnection, ACTIVATION_FUNCTIONS
from mujoco import viewer
from typing import Any
from spatial_individual import SpatialIndividual
from genetic_operators import (
    create_initial_hyperneat_genome,
)
from hyperneat import CPPNNode, CPPNConnection
from evaluation import evaluate_population
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments import SimpleFlatWorld
from simulation_utils import generate_spawn_positions
from hyperneat import CPPN, SubstrateNetwork, create_substrate_for_gecko

def crossover_hyperneat_genotypes(genotype1: dict, genotype2: dict) -> dict:
    """
    Crossover two HyperNEAT genotypes (NEAT-style).
    
    Matching genes are randomly selected from either parent.
    Disjoint genes are included from both parents.
    
    Args:
        genotype1: First parent genotype
        genotype2: Second parent genotype
        
    Returns:
        Offspring genotype
    """  
    # Get connection innovation numbers (using from_node, to_node as innovation)
    connections1 = {(c.from_node, c.to_node): c for c in genotype1['connections']}
    connections2 = {(c.from_node, c.to_node): c for c in genotype2['connections']}
    
    # Find matching and disjoint genes
    innovations1 = set(connections1.keys())
    innovations2 = set(connections2.keys())
    matching = innovations1 & innovations2
    disjoint1 = innovations1 - innovations2
    disjoint2 = innovations2 - innovations1
    
    # Create offspring connections
    offspring_connections = []
    
    # For matching genes, randomly select from either parent
    for innovation in matching:
        if random.random() < 0.5:
            offspring_connections.append(copy.deepcopy(connections1[innovation]))
        else:
            offspring_connections.append(copy.deepcopy(connections2[innovation]))
    
    # Include all disjoint genes from both parents
    for innovation in disjoint1:
        offspring_connections.append(copy.deepcopy(connections1[innovation]))
    for innovation in disjoint2:
        offspring_connections.append(copy.deepcopy(connections2[innovation]))
    
    # Combine nodes from both parents (deduplicate by node_id)
    nodes_dict = {}
    for node in genotype1['nodes']:
        nodes_dict[node.node_id] = copy.deepcopy(node)
    for node in genotype2['nodes']:
        if node.node_id not in nodes_dict:
            nodes_dict[node.node_id] = copy.deepcopy(node)
    
    offspring_nodes = list(nodes_dict.values())
    
    return {
        'nodes': offspring_nodes,
        'connections': offspring_connections
    }


class IncubationEvolution:
    """
    Non-spatial evolutionary algorithm for pre-training robot locomotion.
    
    This is a simple generational GA that evolves robot controllers based
    purely on distance traveled, without any spatial dynamics, energy systems,
    or movement-based mating.
    """
    
    def __init__(
        self,
        population_size: int,
        num_generations: int,
        num_joints: int,
        world_size: list[float],
        simulation_time: float,
        control_clip_min: float,
        control_clip_max: float,
        mutation_rate: float = 0.8,
        mutation_power: float = 0.5,
        add_connection_rate: float = 0.05,
        add_node_rate: float = 0.03,
        crossover_rate: float = 0.9,
        tournament_size: int = 3,
        elitism_count: int = 1,
        use_directional_fitness: bool = True,
        target_distance_min: float = 5.0,
        target_distance_max: float = 10.0,
        progress_weight: float = 2.0,
        distance_weight: float = 0.2
    ):
        """
        Initialize incubation evolution.
        
        Args:
            population_size: Number of individuals in population
            num_generations: Number of generations to evolve
            num_joints: Number of controllable joints
            world_size: Size of simulation world [width, height]
            simulation_time: Duration of each fitness evaluation
            control_clip_min: Minimum control value
            control_clip_max: Maximum control value
            mutation_rate: Weight mutation probability
            mutation_power: Weight mutation strength
            add_connection_rate: Probability of adding new connection
            add_node_rate: Probability of adding new node
            crossover_rate: Probability of crossover vs cloning
            tournament_size: Number of individuals in tournament selection
            elitism_count: Number of best individuals to preserve
            use_directional_fitness: Use directional fitness (movement toward target)
            target_distance_min: Minimum distance to target (meters)
            target_distance_max: Maximum distance to target (meters)
            progress_weight: Weight for progress toward target component
            distance_weight: Weight for total distance traveled component
        """
        self.population_size = population_size
        self.num_generations = num_generations
        self.num_joints = num_joints
        self.world_size = world_size
        self.simulation_time = simulation_time
        self.control_clip_min = control_clip_min
        self.control_clip_max = control_clip_max
        self.mutation_rate = mutation_rate
        self.mutation_power = mutation_power
        self.add_connection_rate = add_connection_rate
        self.add_node_rate = add_node_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        self.use_directional_fitness = use_directional_fitness
        self.target_distance_min = target_distance_min
        self.target_distance_max = target_distance_max
        self.progress_weight = progress_weight
        self.distance_weight = distance_weight
        
        self.population: list[SpatialIndividual] = []
        self.generation = 0
        self.next_unique_id = 0
        
        # Statistics tracking
        self.best_fitness_history: list[float] = []
        self.avg_fitness_history: list[float] = []
        self.worst_fitness_history: list[float] = []
        
        # Store evaluation environment for reuse
        self.isolated_model = None
        self.num_joints = None
        
    def create_individual(self) -> SpatialIndividual:
        """Create a new individual with random HyperNEAT genotype."""
        individual = SpatialIndividual(
            unique_id=self.next_unique_id,
            generation=self.generation
        )
        self.next_unique_id += 1
        
        # Create HyperNEAT genome
        individual.genotype = create_initial_hyperneat_genome(
            num_inputs=4,
            num_outputs=1,
            activation='sine'
        )
        
        individual.fitness = 0.0
        individual.evaluated = False
        
        # Spatial attributes (not used in incubation but needed for compatibility)
        individual.x = 0.0
        individual.y = 0.0
        individual.energy = 100.0
        
        return individual
    
    def initialize_population(self) -> None:
        """Initialize population with random individuals."""
        print(f"  Initializing incubation population of {self.population_size} individuals")
        self.population = [self.create_individual() for _ in range(self.population_size)]
        
    def tournament_selection(self) -> SpatialIndividual:
        """
        Select an individual using tournament selection.
        
        Returns:
            Selected individual (not a copy, just a reference)
        """
        # Randomly sample tournament_size individuals
        tournament = np.random.choice(
            self.population,
            size=min(self.tournament_size, len(self.population)),
            replace=False
        )
        
        # Return the one with highest fitness
        return max(tournament, key=lambda ind: ind.fitness)
    
    def _evaluate_population_with_environment(
        self,
        isolated_model,
        core_geom_id: int,
        num_joints: int
    ) -> list[float]:
        """
        Evaluate population using a pre-created environment with directional fitness.
        
        Each robot is evaluated on its ability to move toward a randomly placed target.
        This teaches directional locomotion, preparing robots for the spatial EA's
        movement-based mating phase.
        
        Fitness combines:
        1. Progress toward target (primary component)
        2. Total distance traveled (bonus for movement)
        
        Args:
            isolated_model: Pre-compiled MuJoCo model
            core_geom_id: ID of core geometry for position tracking
            num_joints: Number of controllable joints
            
        Returns:
            List of fitness values
        """        
        print(f"  Testing each robot in isolated environment (directional fitness)...")
        
        fitness_values = []
        
        # Evaluate each robot individually using the same environment
        for i, individual in enumerate(self.population):
            # Create fresh data for this evaluation
            isolated_data = mujoco.MjData(isolated_model)
            
            # Set random orientation for the robot to avoid directional bias
            # Find the freejoint for robot-0
            joint_id = mujoco.mj_name2id(isolated_model, mujoco.mjtObj.mjOBJ_JOINT, "robot-0")
            robot_yaw = 0.0  # Track the robot's orientation for target placement
            if joint_id >= 0:
                qpos_addr = isolated_model.jnt_qposadr[joint_id]
                # Generate random yaw angle
                robot_yaw = np.random.uniform(0, 2 * np.pi)
                # Set quaternion for rotation around z-axis
                isolated_data.qpos[qpos_addr + 3] = np.cos(robot_yaw / 2)  # qw
                isolated_data.qpos[qpos_addr + 4] = 0.0                     # qx
                isolated_data.qpos[qpos_addr + 5] = 0.0                     # qy
                isolated_data.qpos[qpos_addr + 6] = np.sin(robot_yaw / 2)  # qz
            
            # Record start position
            mujoco.mj_forward(isolated_model, isolated_data)
            start_position = isolated_data.geom_xpos[core_geom_id].copy()
            
            # Generate random target position if using directional fitness
            if self.use_directional_fitness:
                target_distance = np.random.uniform(self.target_distance_min, self.target_distance_max)
                # Generate target angle RELATIVE to robot's facing direction
                relative_angle = np.random.uniform(-np.pi, np.pi)
                # Absolute angle in world frame = robot's yaw + relative angle
                target_angle = robot_yaw + relative_angle
                
                target_position = start_position + np.array([
                    target_distance * np.cos(target_angle),
                    target_distance * np.sin(target_angle),
                    0.0  # Same height
                ])
                
                # Store target for potential visualization
                individual.target_position = target_position.copy()
            else:
                target_position = None
            
            # Create HyperNEAT controller
            cppn = CPPN(individual.genotype)
            input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
                num_joints=num_joints,
                use_hidden_layer=True,
                hidden_layer_size=num_joints
            )
            substrate = SubstrateNetwork(
                input_coords=input_coords,
                hidden_coords=hidden_coords,
                output_coords=output_coords,
                cppn=cppn,
                weight_threshold=0.2
            )
            
            def single_robot_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
                # Get joint angles
                joint_angles = data.qpos[7:7+num_joints].copy()  # Skip free joint (7 DOF)
                
                # Add CPG-style oscillator inputs (multiple frequencies for rich patterns)
                # These provide pre-built oscillatory components for the network
                osc1 = np.sin(data.time * 1.0)  # 1 Hz oscillator
                osc2 = np.cos(data.time * 1.0)  # 1 Hz (90° phase shift)
                osc3 = np.sin(data.time * 2.0)  # 2 Hz oscillator
                osc4 = np.cos(data.time * 2.0)  # 2 Hz (90° phase shift)
                
                cpg_inputs = np.array([osc1, osc2, osc3, osc4])
                bias_input = np.array([1.0])
                
                # Calculate directional input to target (if directional fitness enabled)
                if self.use_directional_fitness and target_position is not None:
                    current_position = data.geom_xpos[core_geom_id][:2]  # x, y only
                    target_vector = target_position[:2] - current_position
                    distance = np.linalg.norm(target_vector)
                    if distance > 0.01:  # Avoid division by zero
                        directional_inputs = target_vector / distance  # Normalized direction
                    else:
                        directional_inputs = np.array([0.0, 0.0])  # At target
                else:
                    directional_inputs = np.array([0.0, 0.0])  # No target info
                
                # Combine inputs: joint angles + CPG oscillators + directional + bias
                sensor_inputs = np.concatenate([joint_angles, cpg_inputs, directional_inputs, bias_input])
                
                # Get motor outputs from substrate
                motor_outputs = substrate.activate(sensor_inputs)
                
                # Apply to actuators
                data.ctrl[:] = np.clip(
                    motor_outputs,
                    self.control_clip_min,
                    self.control_clip_max
                )
            
            # Set controller and run simulation
            mujoco.set_mjcb_control(single_robot_controller)
            sim_steps = int(self.simulation_time / isolated_model.opt.timestep)
            for _ in range(sim_steps):
                mujoco.mj_step(isolated_model, isolated_data)
            
            # Record end position
            end_position = isolated_data.geom_xpos[core_geom_id].copy()
            
            # Calculate fitness based on mode
            if self.use_directional_fitness and target_position is not None:
                # Directional fitness: reward progress toward target
                start_to_target = np.linalg.norm(target_position - start_position)
                end_to_target = np.linalg.norm(target_position - end_position)
                progress_toward_target = start_to_target - end_to_target  # Positive if moved closer
                
                # Total distance traveled (always positive, rewards movement)
                total_distance = np.linalg.norm(end_position - start_position)
                
                # Base fitness on movement, with directional bonus
                # Formula: fitness = distance * (1 + directional_bonus)
                # Where directional_bonus = progress_weight * (progress / distance) if distance > 0
                
                if total_distance > 0.01:  # Avoid division by zero
                    # Calculate directional bonus (ranges from -1 to +1)
                    direction_quality = progress_toward_target / total_distance
                    # Apply progress weight to directional component
                    directional_bonus = self.progress_weight * direction_quality
                    # Combined fitness: base movement + directional multiplier
                    fitness = total_distance * (1.0 + directional_bonus)
                else:
                    # No movement - minimal fitness
                    fitness = 0.0
                
                individual.fitness = fitness
                individual.start_position = start_position
                individual.end_position = end_position
                individual.progress_toward_target = progress_toward_target
                individual.total_distance = total_distance
            else:
                # Simple distance-only fitness
                distance = np.linalg.norm(end_position - start_position)
                individual.fitness = distance
                individual.start_position = start_position
                individual.end_position = end_position
                individual.progress_toward_target = 0.0
                individual.total_distance = distance
            
            fitness_values.append(individual.fitness)
            
            if (i + 1) % 5 == 0:
                print(f"    Evaluated {i + 1}/{len(self.population)} robots")
        
        print(f"  FITNESS EVALUATION COMPLETE")
        return fitness_values
    
    def _mutate_genotype(
        self,
        genotype: dict,
        weight_mutation_rate: float,
        weight_mutation_power: float,
        add_connection_rate: float,
        add_node_rate: float
    ) -> dict:
        """
        Mutate a HyperNEAT genotype in-place.
        
        Args:
            genotype: Genotype to mutate
            weight_mutation_rate: Probability of mutating each weight
            weight_mutation_power: Standard deviation of weight perturbation
            add_connection_rate: Probability of adding new connection
            add_node_rate: Probability of adding new node
            
        Returns:
            Mutated genotype
        """
        # Weight mutation
        for connection in genotype['connections']:
            if random.random() < weight_mutation_rate:
                if random.random() < 0.9:
                    # Perturb weight
                    connection.weight += np.random.normal(0, weight_mutation_power)
                else:
                    # Replace weight
                    connection.weight = np.random.normal(0, 1.0)
        
        # Add connection mutation
        if random.random() < add_connection_rate:
            # Find nodes that could be connected
            # Layer 0 = input, 1+ = hidden, max = output
            max_layer = max(n.layer for n in genotype['nodes'])
            input_nodes = [n for n in genotype['nodes'] if n.layer == 0]
            output_nodes = [n for n in genotype['nodes'] if n.layer == max_layer]
            hidden_nodes = [n for n in genotype['nodes'] if 0 < n.layer < max_layer]
            
            # Only connect from input/hidden to hidden/output (feed-forward)
            source_nodes = input_nodes + hidden_nodes
            target_nodes = hidden_nodes + output_nodes
            
            if source_nodes and target_nodes:
                # Check existing connections
                existing = {(c.from_node, c.to_node) for c in genotype['connections']}
                
                # Find possible new connections
                possible = [(s.node_id, t.node_id) for s in source_nodes for t in target_nodes
                           if (s.node_id, t.node_id) not in existing and s.layer < t.layer]
                
                if possible:
                    from_id, to_id = random.choice(possible)
                    new_connection = CPPNConnection(
                        from_node=from_id,
                        to_node=to_id,
                        weight=np.random.normal(0, 1.0),
                        enabled=True
                    )
                    genotype['connections'].append(new_connection)
        
        # Add node mutation
        if random.random() < add_node_rate and genotype['connections']:
            # Select a random enabled connection to split
            enabled_connections = [c for c in genotype['connections'] if c.enabled]
            if enabled_connections:
                connection_to_split = random.choice(enabled_connections)
                
                # Disable the old connection
                connection_to_split.enabled = False
                
                # Get source and target layers
                source_node = next(n for n in genotype['nodes'] if n.node_id == connection_to_split.from_node)
                target_node = next(n for n in genotype['nodes'] if n.node_id == connection_to_split.to_node)
                
                # New node goes in layer between source and target
                # If they're adjacent layers, create new intermediate layer
                new_layer = (source_node.layer + target_node.layer) // 2
                if new_layer == source_node.layer:
                    new_layer = source_node.layer + 1
                    # Increment all nodes at or above this layer
                    for node in genotype['nodes']:
                        if node.layer >= new_layer and node.node_id != connection_to_split.from_node:
                            node.layer += 1
                
                # Create new hidden node
                new_node_id = max([n.node_id for n in genotype['nodes']]) + 1
                new_activation = random.choice(list(ACTIVATION_FUNCTIONS.keys()))
                new_node = CPPNNode(
                    node_id=new_node_id,
                    activation=new_activation,
                    layer=new_layer
                )
                genotype['nodes'].append(new_node)
                
                # Create two new connections
                conn1 = CPPNConnection(
                    from_node=connection_to_split.from_node,
                    to_node=new_node_id,
                    weight=1.0,  # Identity weight
                    enabled=True
                )
                conn2 = CPPNConnection(
                    from_node=new_node_id,
                    to_node=connection_to_split.to_node,
                    weight=connection_to_split.weight,  # Preserve original weight
                    enabled=True
                )
                genotype['connections'].extend([conn1, conn2])
        
        return genotype
    
    def create_offspring(self, parent1: SpatialIndividual, parent2: SpatialIndividual) -> SpatialIndividual:
        """
        Create an offspring from two parents using crossover and mutation.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            New offspring individual
        """
        offspring = SpatialIndividual(
            unique_id=self.next_unique_id,
            generation=self.generation + 1
        )
        self.next_unique_id += 1
        
        # Apply crossover with probability
        if np.random.random() < self.crossover_rate:
            offspring.genotype = crossover_hyperneat_genotypes(
                parent1.genotype,
                parent2.genotype
            )
        else:
            # Clone parent1
            offspring.genotype = copy.deepcopy(parent1.genotype)
        
        # Apply mutation (mutate the genotype in-place)
        offspring.genotype = self._mutate_genotype(
            offspring.genotype,
            weight_mutation_rate=self.mutation_rate,
            weight_mutation_power=self.mutation_power,
            add_connection_rate=self.add_connection_rate,
            add_node_rate=self.add_node_rate
        )
        
        offspring.fitness = 0.0
        offspring.evaluated = False
        offspring.x = 0.0
        offspring.y = 0.0
        offspring.energy = 100.0
        
        # Track parent IDs
        offspring.parent_ids = [parent1.unique_id, parent2.unique_id]
        
        return offspring
    
    def create_next_generation(self) -> None:
        """
        Create next generation using tournament selection, crossover, and mutation.
        """
        # Sort population by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        # Start with elite individuals (exact copies)
        next_generation = []
        for i in range(min(self.elitism_count, len(self.population))):
            elite = SpatialIndividual(
                unique_id=self.next_unique_id,
                generation=self.generation + 1
            )
            self.next_unique_id += 1
            
            elite.genotype = copy.deepcopy(self.population[i].genotype)
            elite.fitness = 0.0  # Will be re-evaluated
            elite.evaluated = False
            elite.x = 0.0
            elite.y = 0.0
            elite.energy = 100.0
            elite.parent_ids = [self.population[i].unique_id]
            
            next_generation.append(elite)
        
        # Fill rest of population with offspring
        while len(next_generation) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            offspring = self.create_offspring(parent1, parent2)
            next_generation.append(offspring)
        
        self.population = next_generation
        
    def run(self) -> list[SpatialIndividual]:
        """
        Run incubation evolution.
        
        Returns:
            Final population after incubation
        """
        print("\n" + "="*60)
        print("INCUBATION PHASE - Non-Spatial Evolution")
        print("="*60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Controller type: HyperNEAT")
        print(f"Fitness mode: {'DIRECTIONAL (toward target)' if self.use_directional_fitness else 'DISTANCE (any direction)'}")
        if self.use_directional_fitness:
            print(f"  Target distance range: {self.target_distance_min:.1f}-{self.target_distance_max:.1f}m")
            print(f"  Progress weight: {self.progress_weight:.1f}x")
            print(f"  Distance bonus weight: {self.distance_weight:.1f}x")
        print(f"Tournament size: {self.tournament_size}")
        print(f"Elitism: {self.elitism_count} best preserved")
        print(f"Crossover rate: {self.crossover_rate}")
        print(f"Mutation - weight: {self.mutation_rate}, add conn: {self.add_connection_rate}, add node: {self.add_node_rate}")
        print("="*60)
        
        # Initialize population
        self.initialize_population()
        
        # Print initial diversity statistics
        print("\n  Initial Population Diversity:")
        structures = {}
        activations = {}
        weight_ranges = []
        
        for ind in self.population:
            # Count structure type
            num_nodes = len(ind.genotype['nodes'])
            num_conns = len(ind.genotype['connections'])
            struct_key = f"{num_nodes}n_{num_conns}c"
            structures[struct_key] = structures.get(struct_key, 0) + 1
            
            # Count output activations
            max_layer = max(n.layer for n in ind.genotype['nodes'])
            output_nodes = [n for n in ind.genotype['nodes'] if n.layer == max_layer]
            for out_node in output_nodes:
                activations[out_node.activation] = activations.get(out_node.activation, 0) + 1
            
            # Track weight ranges
            weights = [abs(c.weight) for c in ind.genotype['connections']]
            if weights:
                weight_ranges.append((min(weights), max(weights), np.mean(weights)))
        
        print(f"    Network structures: {dict(structures)}")
        print(f"    Output activations: {dict(activations)}")
        if weight_ranges:
            avg_min = np.mean([r[0] for r in weight_ranges])
            avg_max = np.mean([r[1] for r in weight_ranges])
            avg_mean = np.mean([r[2] for r in weight_ranges])
            print(f"    Weight magnitudes: min={avg_min:.3f}, max={avg_max:.3f}, avg={avg_mean:.3f}")
        print()
        
        # Create reusable evaluation environment (to avoid MuJoCo compilation issues)
        print("  Creating reusable evaluation environment...")
        isolated_world = SimpleFlatWorld(self.world_size)
        isolated_robot = gecko()
        isolated_world.spawn(
            isolated_robot.spec, 
            spawn_position=[0, 0, 0.5],
            correct_for_bounding_box=False
        )
        isolated_model = isolated_world.spec.compile()
        num_joints = isolated_model.nu
        
        # Store for later use in demonstration
        self.isolated_model = isolated_model
        self.num_joints = num_joints
        
        # Get core geom ID
        all_geoms = isolated_world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        core_geom_id = None
        for geom in all_geoms:
            if geom.name == "robot-0core":
                core_geom_id = mujoco.mj_name2id(
                    isolated_model, mujoco.mjtObj.mjOBJ_GEOM, "robot-0core"
                )
                break
        
        print(f"  Environment ready ({num_joints} joints)")
        
        # Evolution loop
        for gen in range(self.num_generations):
            self.generation = gen
            
            print(f"\nIncubation Generation {gen + 1}/{self.num_generations}:")
            
            # Evaluate fitness using reusable environment
            fitness_values = self._evaluate_population_with_environment(
                isolated_model=isolated_model,
                core_geom_id=core_geom_id,
                num_joints=num_joints
            )
            
            # Mark as evaluated
            for ind in self.population:
                ind.evaluated = True
            
            # Track statistics
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            worst_fitness = min(fitness_values)
            
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)
            self.worst_fitness_history.append(worst_fitness)
            
            # Calculate directional fitness statistics
            best_ind = max(self.population, key=lambda ind: ind.fitness)
            avg_progress = np.mean([ind.progress_toward_target for ind in self.population])
            avg_distance = np.mean([ind.total_distance for ind in self.population])
            
            # Calculate direction quality for best individual
            if best_ind.total_distance > 0.01:
                best_direction_quality = best_ind.progress_toward_target / best_ind.total_distance
            else:
                best_direction_quality = 0.0
            
            # Calculate average direction quality
            direction_qualities = []
            for ind in self.population:
                if ind.total_distance > 0.01:
                    direction_qualities.append(ind.progress_toward_target / ind.total_distance)
            avg_direction_quality = np.mean(direction_qualities) if direction_qualities else 0.0
            
            print(f"  Best fitness: {best_fitness:.4f}")
            print(f"    Progress toward target: {best_ind.progress_toward_target:.4f}m")
            print(f"    Total distance: {best_ind.total_distance:.4f}m")
            print(f"    Direction quality: {best_direction_quality:.3f} (1.0=perfect, 0.0=perpendicular, -1.0=opposite)")
            print(f"  Average fitness: {avg_fitness:.4f}")
            print(f"    Avg progress toward target: {avg_progress:.4f}m")
            print(f"    Avg total distance: {avg_distance:.4f}m")
            print(f"    Avg direction quality: {avg_direction_quality:.3f}")
            print(f"  Worst fitness: {worst_fitness:.4f}")
            
            # Create next generation (except on last generation)
            if gen < self.num_generations - 1:
                self.create_next_generation()
                print(f"  Created next generation ({len(self.population)} individuals)")
        
        # Sort final population by fitness
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)
        
        print("\n" + "="*60)
        print("INCUBATION PHASE COMPLETE")
        print(f"Best fitness achieved: {self.best_fitness_history[-1]:.4f}")
        print(f"Average fitness final generation: {self.avg_fitness_history[-1]:.4f}")
        print(f"Fitness improvement: {self.best_fitness_history[-1] - self.best_fitness_history[0]:.4f}")
        print("="*60 + "\n")
        
        return self.population
    
    def demonstrate_best(self, duration: float = 15.0) -> None:
        """
        Demonstrate the best individual from incubation using MuJoCo visualizer.
        
        Args:
            duration: Duration of demonstration in seconds
        """
        print("\n" + "="*60)
        print("DEMONSTRATING BEST INCUBATION INDIVIDUAL")
        print("="*60)
        
        if not self.population:
            print("No individuals in population to demonstrate")
            return
        
        if self.isolated_model is None or self.num_joints is None:
            print("No evaluation environment available. Run incubation first.")
            return
        
        # Get best individual
        best = max(self.population, key=lambda ind: ind.fitness)
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Individual ID: {best.unique_id}")
        print(f"Network structure: {len(best.genotype['nodes'])} nodes, {len(best.genotype['connections'])} connections")
        
        # Create new data instance (reuse model)
        print("\nPreparing demonstration...")
        isolated_data = mujoco.MjData(self.isolated_model)
        
        # Create HyperNEAT controller for best individual
        cppn = CPPN(best.genotype)
        input_coords, hidden_coords, output_coords = create_substrate_for_gecko(
            num_joints=self.num_joints,
            use_hidden_layer=True,
            hidden_layer_size=self.num_joints
        )
        substrate = SubstrateNetwork(
            input_coords=input_coords,
            hidden_coords=hidden_coords,
            output_coords=output_coords,
            cppn=cppn,
            weight_threshold=0.2
        )
        
        def demo_controller(model: mujoco.MjModel, data: mujoco.MjData) -> None:
            # Get joint angles
            joint_angles = data.qpos[7:7+self.num_joints].copy()  # Skip free joint (7 DOF)
            
            # Add CPG-style oscillator inputs (multiple frequencies for rich patterns)
            osc1 = np.sin(data.time * 1.0)  # 1 Hz oscillator
            osc2 = np.cos(data.time * 1.0)  # 1 Hz (90 deg phase shift)
            osc3 = np.sin(data.time * 2.0)  # 2 Hz oscillator
            osc4 = np.cos(data.time * 2.0)  # 2 Hz (90 deg phase shift)
            
            cpg_inputs = np.array([osc1, osc2, osc3, osc4])
            bias_input = np.array([1.0])
            
            # No directional input during demonstration (no specific target)
            directional_inputs = np.array([0.0, 0.0])
            
            # Combine inputs: joint angles + CPG oscillators + directional + bias
            sensor_inputs = np.concatenate([joint_angles, cpg_inputs, directional_inputs, bias_input])
            
            # Get motor outputs from substrate
            motor_outputs = substrate.activate(sensor_inputs)
            
            # Debug: Print motor outputs periodically
            if int(data.time * 10) % 50 == 0:  # Print every 0.5 seconds
                print(f"  Time: {data.time:.2f}s | Motors: {motor_outputs[:4]}... | OSC: [{osc1:.2f}, {osc2:.2f}, {osc3:.2f}, {osc4:.2f}]")
            
            # Apply to actuators
            data.ctrl[:] = np.clip(
                motor_outputs,
                self.control_clip_min,
                self.control_clip_max
            )
        
        # Set controller
        mujoco.set_mjcb_control(demo_controller)
        
        # Launch interactive visualizer
        print(f"\nLaunching MuJoCo visualizer for {duration} seconds...")
        print("Press ESC to exit early, SPACE to pause/resume")
        print("Use mouse to rotate view: Left-drag=rotate, Right-drag=zoom, Ctrl+drag=pan")
        
        # Use passive viewer for interactive visualization
        print("\nStarting visualization...")
        with viewer.launch_passive(self.isolated_model, isolated_data) as v:
            start_time = time.time()
            while time.time() - start_time < duration:
                mujoco.mj_step(self.isolated_model, isolated_data)
                v.sync()
                # Small sleep to prevent spinning too fast (matches ~60 FPS)
                time.sleep(0.001)
        
        print("\nDemonstration complete!")
        print("="*60)


def seed_spatial_population_from_incubation(
    incubation_population: list[SpatialIndividual],
    target_population_size: int,
    starting_generation: int,
    next_unique_id: int,
    world_size: list[float],
    spawn_x_range: tuple[float, float],
    spawn_y_range: tuple[float, float],
    min_spawn_distance: float,
    initial_energy: float = 100.0
) -> tuple[list[SpatialIndividual], int]:
    """
    Seed spatial EA population from incubation results.
    
    Takes the best individuals from incubation and creates the initial
    spatial population. If incubation population is smaller than target,
    creates clones of the best individuals. If larger, takes the top performers.
    
    Args:
        incubation_population: Population from incubation (sorted by fitness)
        target_population_size: Desired population size for spatial EA
        starting_generation: Generation number for spatial EA start
        next_unique_id: Next available unique ID
        world_size: Size of simulation world
        spawn_x_range: (min, max) x coordinates for spawning
        spawn_y_range: (min, max) y coordinates for spawning
        min_spawn_distance: Minimum distance between individuals
        initial_energy: Starting energy for each individual
        
    Returns:
        (spatial_population, updated_next_unique_id)
    """
    print(f"\nSeeding spatial population from incubation:")
    print(f"  Incubation population: {len(incubation_population)} individuals")
    print(f"  Target spatial population: {target_population_size}")
    
    spatial_population = []
    
    # Determine how many unique genotypes to use
    num_unique = min(len(incubation_population), target_population_size)
    
    # Generate spawn positions
    positions = generate_spawn_positions(
        population_size=target_population_size,
        spawn_x_range=spawn_x_range,
        spawn_y_range=spawn_y_range,
        spawn_z=0.0,  # Will be set properly during spawning
        min_spawn_distance=min_spawn_distance
    )
    
    # Create spatial population
    for i in range(target_population_size):
        # Select source individual (cycle through best if needed)
        source_idx = i % num_unique
        source = incubation_population[source_idx]
        
        # Create new individual
        individual = SpatialIndividual(
            unique_id=next_unique_id,
            generation=starting_generation
        )
        next_unique_id += 1
        
        # Copy genotype from incubation
        individual.genotype = copy.deepcopy(source.genotype)
        
        # Copy fitness (already evaluated)
        individual.fitness = source.fitness
        individual.evaluated = True  # Don't re-evaluate
        
        # Set spatial attributes
        individual.x = positions[i][0]
        individual.y = positions[i][1]
        individual.energy = initial_energy
        
        # Track lineage
        individual.parent_ids = [source.unique_id]
        
        spatial_population.append(individual)
    
    print(f"  Created {len(spatial_population)} spatial individuals")
    print(f"  Used {num_unique} unique genotypes from incubation")
    if target_population_size > num_unique:
        print(f"  Cloned best {target_population_size - num_unique} individuals to fill population")
    
    # Report fitness statistics
    fitnesses = [ind.fitness for ind in spatial_population]
    print(f"  Fitness range: {min(fitnesses):.4f} - {max(fitnesses):.4f}")
    print(f"  Average fitness: {np.mean(fitnesses):.4f}")
    
    return spatial_population, next_unique_id
