"""Grid-based spatial evolutionary algorithm with sector-based mating preferences."""

import mujoco
import numpy as np
from pathlib import Path

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from pprint import pprint

from grid import Grid
from individual import SpatialIndividual
from genetic_operators import crossover, mutate, create_individual
from simulation import spatial_controller, evaluate_population
from visualization import visualize_grid_state, plot_fitness_evolution, plot_movement_analysis
from ea_config import config


class GridSpatialEA:
    """Grid-based spatial evolutionary algorithm."""
    
    def __init__(
        self, 
        population_size: int = None, 
        num_generations: int = None, 
        num_joints: int = None
    ):
        self.population_size = population_size or config.population_size
        self.num_generations = num_generations or config.num_generations
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.best_individual_history = []
        
        # Grid structure
        self.grid = Grid(config.world_size, grid_size=3)
        
        # World and simulation
        self.world = SimpleFlatWorld(config.world_size)
        self.gecko = gecko()
        self.gecko_spec = self.gecko.spec

        self.world.spawn(self.gecko_spec, spawn_position=[0, 0, 0.5])
        
        self.model = self.world.spec.compile()
        self.num_joints = self.model.nu

        self.data = None
        self.robots = []
        self.tracked_geoms = []
        
        # Store current positions for cross-generation persistence
        self.current_positions = []
        
        # Counter for assigning unique IDs to individuals
        self.next_unique_id = 0
        
        # Track movement history for visualization
        self.movement_history = []  # List of (gen, individual_id, from_sector, to_sector)
        
        # Track sector distribution over generations as 2D matrix (3x3 grid)
        # Shape: (num_generations, 3, 3) - stores count of individuals per sector per generation
        self.sector_distribution_history = []
    
    def get_sector_distribution_matrix(self):
        """Get current sector distribution as a 3x3 matrix."""
        distribution = np.zeros((3, 3), dtype=int)
        
        for individual in self.population:
            sector_id = individual.sector_id
            row = sector_id // 3
            col = sector_id % 3
            distribution[row, col] += 1
        
        return distribution
    
    def initialize_population(self):
        """Initialize population with random individuals."""
        print()
        print("*" * 200)
        print(f"Initializing population of {self.population_size} individuals")
        
        self.population = []
        for _ in range(self.population_size):
            individual, self.next_unique_id = create_individual(
                self.next_unique_id, self.num_joints, config
            )
            self.population.append(individual)
        
        # print 5 random individuals for verification
        for _ in range(5):
            rand_idx = np.random.randint(0, self.population_size)
            print(self.population[rand_idx])
        
        print("*" * 200)
        print('\n\n')  

    def spawn_population(self):
        """Spawn robots in the simulation world."""
        print(f"Spawning {self.population_size} robots in grid sectors")
        
        mujoco.set_mjcb_control(None)
        self.world = SimpleFlatWorld(config.world_size)
        self.robots = []
        
        ######### Determine spawn positions #########
        if len(self.current_positions) == self.population_size:
            print(f"  Using positions from previous generation")
            positions = [pos.copy() for pos in self.current_positions]
        else:
            # Initialize: distribute evenly across sectors
            print(f"  Generating initial grid-based positions")
            positions = []
            individuals_per_sector = max(1, self.population_size // 9)
            
            for sector_id in range(9):
                sector_center = self.grid.get_sector_center(sector_id, config.spawn_z)
                
                # How many to place in this sector?
                n_in_sector = min(individuals_per_sector, 
                                 self.population_size - len(positions))
                
                for i in range(n_in_sector):
                    # Add small random offset within sector
                    offset_x = np.random.uniform(-self.grid.sector_width * 0.3, 
                                                self.grid.sector_width * 0.3)
                    offset_y = np.random.uniform(-self.grid.sector_height * 0.3, 
                                                self.grid.sector_height * 0.3)
                    
                    pos = sector_center.copy()
                    pos[0] += offset_x
                    pos[1] += offset_y
                    
                    # Clamp to world bounds
                    pos[0] = np.clip(pos[0], 0.1, config.world_size[0] - 0.1)
                    pos[1] = np.clip(pos[1], 0.1, config.world_size[1] - 0.1)
                    
                    positions.append(pos)
        
        ######### Spawn robots #########
        for i, individual in enumerate(self.population):
            robot = gecko()
            self.robots.append(robot)
            pos = positions[i]
            individual.spawn_position = np.array(pos)
            individual.robot_index = i
            individual.sector_id = self.grid.get_sector_id(pos)
            self.world.spawn(robot.spec, spawn_position=pos, prefix_id=i)

        # Update grid sector membership
        self.grid.update_sectors(self.population, positions)
        self.current_positions = positions
        
        # Compile world
        self.model = self.world.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        mujoco.mj_forward(self.model, self.data)
        
        ######### Track all robot core geoms #########
        all_geoms = self.world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        self.tracked_geoms = []
        for i in range(self.population_size):
            core_name = f"robot-{i}core"
            for geom in all_geoms:
                if geom.name == core_name:
                    self.tracked_geoms.append(self.data.bind(geom))
                    break
        
        # Get and store sector distribution
        sector_dist = self.get_sector_distribution_matrix()
        
        # Print sector distribution
        print(f"Grid sector distribution (3x3 matrix):")
        print(sector_dist)
        
        # print random robots for verification
        for _ in range(5):
            rand_idx = np.random.randint(0, self.population_size)
            print(self.population[rand_idx])
        
        print(f"Spawned {len(self.population)} robots successfully")
    
    def evaluate(self):
        """Evaluate fitness for all individuals."""
        return evaluate_population(self.population, config)
    
    def movement_phase(self, duration: float = 30.0):
        """Allow individuals to move between adjacent sectors."""
        print(f"  Movement phase: individuals migrating between sectors...")
        
        # Track sector changes
        initial_sectors = [ind.sector_id for ind in self.population]

        # Each individual decides whether to move to adjacent sector
        for i, individual in enumerate(self.population):
            current_sector = individual.sector_id
            adjacent_sectors = self.grid.get_adjacent_sectors(current_sector)

            if len(adjacent_sectors) == 0:
                continue

            # Movement probability based on fitness gradient
            fitness_rank = sorted(self.population, key=lambda x: x.fitness, reverse=True).index(individual)
            fitness_percentile = fitness_rank / len(self.population)
            
            move_probability = 0.3 + 0.4 * fitness_percentile
            
            if np.random.random() < move_probability:
                new_sector = np.random.choice(adjacent_sectors)
                
                new_center = self.grid.get_sector_center(new_sector, config.spawn_z)
                offset_x = np.random.uniform(-self.grid.sector_width * 0.3, 
                                            self.grid.sector_width * 0.3)
                offset_y = np.random.uniform(-self.grid.sector_height * 0.3, 
                                            self.grid.sector_height * 0.3)
                
                new_pos = new_center.copy()
                new_pos[0] += offset_x
                new_pos[1] += offset_y
                new_pos[0] = np.clip(new_pos[0], 0.1, config.world_size[0] - 0.1)
                new_pos[1] = np.clip(new_pos[1], 0.1, config.world_size[1] - 0.1)
                
                self.current_positions[i] = new_pos
                individual.sector_id = new_sector
                
                self.movement_history.append({
                    'generation': self.generation,
                    'individual_id': individual.unique_id,
                    'from_sector': current_sector,
                    'to_sector': new_sector
                })
        
        # Update grid sectors
        self.grid.update_sectors(self.population, self.current_positions)

        final_sectors = [ind.sector_id for ind in self.population]
        num_moved = sum(1 for i, f in zip(initial_sectors, final_sectors) if i != f)
        
        print(f"  Movement complete: {num_moved}/{self.population_size} individuals changed sectors")
        
        # Get and print updated sector distribution
        sector_dist = self.get_sector_distribution_matrix()
        print(f"  Updated sector distribution:")
        print(sector_dist)

    def apply_death_mechanism(self, target_population_size=None):
        """Kill worst individuals to maintain fixed population size."""
        if target_population_size is None:
            target_population_size = config.population_size
        
        if len(self.population) <= target_population_size:
            return 0
        
        sorted_indices = sorted(range(len(self.population)), 
                            key=lambda i: self.population[i].fitness)
        
        n_to_remove = len(self.population) - target_population_size
        indices_to_remove = sorted_indices[:n_to_remove]
        
        for idx in sorted(indices_to_remove, reverse=True):
            del self.population[idx]
            del self.current_positions[idx]
        
        self.population_size = len(self.population)
        self.grid.update_sectors(self.population, self.current_positions)
        
        return len(indices_to_remove)
   
    def create_next_generation(self):
        """Create offspring using sector-based mating with preferences."""
        print(f"  Creating next generation with sector-based mating...")
        
        new_population = []
        new_positions = []
        
        self.movement_phase()
        
        pairs = []
        paired_indices = set()

        fitness_ranking = sorted(enumerate(self.population), 
                               key=lambda x: x[1].fitness, reverse=True)

        for idx, individual in fitness_ranking:
            if idx in paired_indices:
                continue
            
            current_sector = individual.sector_id
            p_local = individual.p_local
            
            mate_locally = np.random.random() < p_local
            
            if mate_locally:
                candidate_indices = [i for i in self.grid.sectors[current_sector] 
                                   if i != idx and i not in paired_indices]
            else:
                adjacent_sectors = self.grid.get_adjacent_sectors(current_sector)
                candidate_indices = []
                for adj_sector in adjacent_sectors:
                    candidate_indices.extend([i for i in self.grid.sectors[adj_sector]
                                            if i not in paired_indices])
            
            if len(candidate_indices) == 0:
                continue
            
            best_mate_idx = max(candidate_indices, 
                              key=lambda i: self.population[i].fitness)

            pairs.append((idx, best_mate_idx))
            paired_indices.add(idx)
            paired_indices.add(best_mate_idx)
        
        print(f"  Created {len(pairs)} pairs from {self.population_size} individuals")
        
        for parent1_idx, parent2_idx in pairs:
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            if np.random.random() < config.crossover_rate:
                child1, child2, self.next_unique_id = crossover(
                    parent1, parent2, self.next_unique_id
                )
            else:
                child1 = SpatialIndividual(unique_id=self.next_unique_id)
                self.next_unique_id += 1
                child1.genotype = parent1.genotype.copy()
                child1.p_local = parent1.p_local
                child1.parent_ids = [parent1.unique_id]
                
                child2 = SpatialIndividual(unique_id=self.next_unique_id)
                self.next_unique_id += 1
                child2.genotype = parent2.genotype.copy()
                child2.p_local = parent2.p_local
                child2.parent_ids = [parent2.unique_id]
            
            child1, self.next_unique_id = mutate(child1, self.next_unique_id, config)
            child2, self.next_unique_id = mutate(child2, self.next_unique_id, config)
            
            parent1_sector = parent1.sector_id
            parent2_sector = parent2.sector_id
            
            child1_center = self.grid.get_sector_center(parent1_sector, config.spawn_z)
            child1_pos = child1_center.copy()
            child1_pos[0] += np.random.uniform(-self.grid.sector_width * 0.3, 
                                              self.grid.sector_width * 0.3)
            child1_pos[1] += np.random.uniform(-self.grid.sector_height * 0.3, 
                                              self.grid.sector_height * 0.3)
            child1_pos[0] = np.clip(child1_pos[0], 0.1, config.world_size[0] - 0.1)
            child1_pos[1] = np.clip(child1_pos[1], 0.1, config.world_size[1] - 0.1)
            
            child2_center = self.grid.get_sector_center(parent2_sector, config.spawn_z)
            child2_pos = child2_center.copy()
            child2_pos[0] += np.random.uniform(-self.grid.sector_width * 0.3, 
                                              self.grid.sector_width * 0.3)
            child2_pos[1] += np.random.uniform(-self.grid.sector_height * 0.3, 
                                              self.grid.sector_height * 0.3)
            child2_pos[0] = np.clip(child2_pos[0], 0.1, config.world_size[0] - 0.1)
            child2_pos[1] = np.clip(child2_pos[1], 0.1, config.world_size[1] - 0.1)
            
            new_population.append(child1)
            new_positions.append(child1_pos)
            
            new_population.append(child2)
            new_positions.append(child2_pos)
        
        self.population.extend(new_population)
        self.current_positions.extend(new_positions)
        
        old_size = self.population_size
        self.population_size = len(self.population)
        
        print(f"  Population grew from {old_size} to {self.population_size}")
        print(f"  Added {len(new_population)} offspring")
        
        n_removed = self.apply_death_mechanism(config.population_size)
        print(f"  Removed {n_removed} individuals via death mechanism")
        print(f"  Final population size: {self.population_size}")
    
    def run_evolution(self):
        """Run the evolutionary algorithm."""
        print("=" * 60)
        print("GRID-BASED SPATIAL EA WITH MATING PREFERENCES")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Grid: 3×3 sectors")
        print("=" * 60)
        
        self.initialize_population()
        
        for gen in range(self.num_generations):
            self.generation = gen
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            self.spawn_population()
            fitness_values = self.evaluate()

            # Store sector distribution for this generation
            sector_dist = self.get_sector_distribution_matrix()
            self.sector_distribution_history.append(sector_dist.copy())

            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            self.fitness_history.append(best_fitness)
            
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_individual_history.append(best_individual)
            
            if config.print_generation_stats:
                print(f"  Best fitness: {best_fitness:.4f}")
                print(f"  Average fitness: {avg_fitness:.4f}")
                print(f"  Average p_local: {np.mean([ind.p_local for ind in self.population]):.3f}")
            
            if gen < self.num_generations - 1:
                self.create_next_generation()
            
            visualize_grid_state(
                self.grid, self.population, self.current_positions,
                self.fitness_history, self.generation, config
            )
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        print("\nSector Distribution History (3x3 matrix per generation):")
        for gen, dist in enumerate(self.sector_distribution_history):
            print(f"\nGeneration {gen}:")
            print(dist)

        return self.get_best_individual()
    
    def get_best_individual(self):
        """Return the best individual from the population."""
        return max(self.population, key=lambda ind: ind.fitness)
    
    def get_sector_distribution_history(self):
        """Return the complete sector distribution history."""
        return np.array(self.sector_distribution_history)
    
    def demonstrate_best(self):
        """Demonstrate the best individual."""
        print(f"\n{'='*60}")
        print("DEMONSTRATING BEST INDIVIDUAL")
        print(f"{'='*60}")
        
        best = self.get_best_individual()
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Best p_local (mating preference): {best.p_local:.3f}")
        
        if config.print_final_genotype:
            print("\nBest genotype (amplitude, frequency, phase per joint):")
            genotype = best.genotype[:-1]
            for j in range(self.num_joints):
                if j * 3 + 2 < len(genotype):
                    amp, freq, phase = genotype[j*3], genotype[j*3+1], genotype[j*3+2]
                    print(f"  Joint {j}: amp={amp:.3f}, freq={freq:.3f}, phase={phase:.3f}")
        
        mujoco.set_mjcb_control(None)
        demo_world = SimpleFlatWorld(config.world_size)
        demo_robot = gecko()
        demo_world.spawn(demo_robot.spec, spawn_position=[0, 0, 0])
        demo_model = demo_world.spec.compile()
        demo_data = mujoco.MjData(demo_model)
        
        def demo_controller(model, data):
            genotype = best.genotype[:-1]
            for j in range(min(demo_model.nu, len(genotype) // 3)):
                if j * 3 + 2 < len(genotype):
                    amplitude = genotype[j * 3]
                    frequency = genotype[j * 3 + 1]
                    phase = genotype[j * 3 + 2]
                    control_value = amplitude * np.sin(frequency * data.time + phase)
                    data.ctrl[j] = np.clip(
                        control_value,
                        config.control_clip_min,
                        config.control_clip_max
                    )
        
        mujoco.set_mjcb_control(demo_controller)
        
        Path(config.video_folder).mkdir(parents=True, exist_ok=True)
        
        video_recorder = VideoRecorder(output_folder=config.video_folder)
        print("Recording best individual demonstration...")
        
        video_renderer(
            demo_model,
            demo_data,
            duration=config.final_demo_time,
            video_recorder=video_recorder,
        )
        
        print("Demonstration complete!")
    
    def demonstrate_final_population(self):
        """Demonstrate the final population."""
        print(f"\n{'='*60}")
        print("DEMONSTRATING FINAL POPULATION")
        print(f"{'='*60}")
        print(f"Recording {self.population_size} robots...")
        
        self.spawn_population()
        mujoco.set_mjcb_control(
            lambda m, d: spatial_controller(m, d, self.population, self.tracked_geoms, self.num_joints, config)
        )
        
        Path(config.video_folder).mkdir(parents=True, exist_ok=True)
        
        video_recorder = VideoRecorder(output_folder=config.video_folder)
        
        video_renderer(
            self.model,
            self.data,
            duration=config.multi_robot_demo_time,
            video_recorder=video_recorder,
        )
        
        print("Final population demonstration complete!")
    
    def plot_fitness_evolution(self):
        """Plot fitness and p_local evolution."""
        plot_fitness_evolution(self.fitness_history, self.best_individual_history, config)
    
    def plot_movement_analysis(self):
        """Analyze and visualize movement patterns."""
        plot_movement_analysis(self.movement_history, config)