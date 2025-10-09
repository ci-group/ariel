"""
Basic discrete generational evolutionary algorithm (EA) for evolving robot movement
using physical proximity for selection.

Loop:
1. Initialize population with random genotypes
2. For each generation:
    a. Spawn robots in non-overlapping positions
    b. Evaluate fitness (distance traveled in fixed time)
    c. Allow robots to move towards attractive neighbors based on fitness
    d. Pair up robots based on final proximity after movement
    e. Create next generation via crossover and mutation
3. After all generations, demonstrate best individual and final population

"""

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import viewer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

# Import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Import configuration
from ea_config import config


class SpatialIndividual:
    def __init__(self):
        self.genotype = []
        self.fitness = 0.0
        self.start_position = None
        self.end_position = None
        self.spawn_position = None
        self.robot_index = None


class SpatialEA:
    def __init__(
        self, 
        population_size : int = None, 
        num_generations : int = None, 
        num_joints : int = None
    ):
        self.population_size = population_size or config.population_size
        self.num_generations = num_generations or config.num_generations
        self.num_joints = num_joints
        
        self.population = []
        self.generation = 0
        self.fitness_history = []
        self.best_individual_history = []
        
        # World and simulation
        self.world = None
        self.model = None
        self.data = None
        self.robots = []
        self.tracked_geoms = []
        
        # For tracking positions during simulation
        self.position_histories = []
        
    def create_individual(self):
        individual = SpatialIndividual()
        
        genotype = []
        for _ in range(self.num_joints):
            amplitude = np.random.uniform(config.amplitude_init_min, config.amplitude_init_max)
            frequency = np.random.uniform(config.frequency_init_min, config.frequency_init_max)
            phase = np.random.uniform(config.phase_min, config.phase_max)
            genotype.extend([amplitude, frequency, phase])
        
        individual.genotype = genotype
        individual.fitness = 0.0
        return individual
    
    def initialize_population(self):
        print(f"Initializing population of {self.population_size} individuals")
        self.population = [self.create_individual() for _ in range(self.population_size)]
        
    def spawn_population(self):
        print(f"Spawning {self.population_size} robots in simulation space")
        
        mujoco.set_mjcb_control(None)
        self.world = SimpleFlatWorld(config.world_size)
        self.robots = []
        
        ######### Generate non-overlapping spawn positions for all robots #########
        positions = []
        min_distance = config.min_spawn_distance  # Get from config
        max_attempts = 1000  # Maximum attempts to find a valid position
        
        for i in range(self.population_size):
            attempts = 0
            position_found = False
            
            while not position_found and attempts < max_attempts:
                # Generate random position
                x = np.random.uniform(config.spawn_x_min, config.spawn_x_max)
                y = np.random.uniform(config.spawn_y_min, config.spawn_y_max)
                z = config.spawn_z
                new_pos = np.array([x, y, z])
                
                # Check distance to all existing positions
                valid = True
                for existing_pos in positions:
                    distance = np.linalg.norm(new_pos[:2] - existing_pos[:2])  # Only check x,y
                    if distance < min_distance:
                        valid = False
                        break
                
                if valid:
                    positions.append(new_pos)
                    position_found = True
                else:
                    attempts += 1
            
            if not position_found:
                # If we couldn't find a non-overlapping position, use grid placement
                print(f"  Warning: Could not find non-overlapping position for robot {i}, using grid fallback")
                grid_size = int(np.ceil(np.sqrt(self.population_size)))
                grid_x = (i % grid_size) * min_distance + config.spawn_x_min
                grid_y = (i // grid_size) * min_distance + config.spawn_y_min
                positions.append(np.array([grid_x, grid_y, config.spawn_z]))
        
        ######### Spawn robots #########
        for i, individual in enumerate(self.population):
            robot = gecko()
            self.robots.append(robot)
            pos = positions[i]
            individual.spawn_position = np.array(pos)
            individual.robot_index = i
            self.world.spawn(robot.spec, spawn_position=pos, prefix_id=i)
        
        # Compile world
        self.model = self.world.spec.compile()
        self.data = mujoco.MjData(self.model)
        
        mujoco.mj_forward(self.model, self.data) # Forward simulate to initialize positions
        
        ######### Track all robot core geoms #########
        all_geoms = self.world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        self.tracked_geoms = []
        for i in range(self.population_size):
            # Match exactly "robot-{i}core" - the main core of robot i
            core_name = f"robot-{i}core"
            for geom in all_geoms:
                if geom.name == core_name:
                    self.tracked_geoms.append(self.data.bind(geom))
                    break
        
        print(f"Spawned {len(self.population)} robots successfully")
        print(f"Tracking {len(self.tracked_geoms)} core geoms")
    
    def spatial_controller(
        self, 
        model : mujoco.MjModel, 
        data : mujoco.MjData
    ):
        num_joints_per_robot = self.num_joints
        
        for robot_idx, individual in enumerate(self.population):
            genotype = individual.genotype
            
            for j in range(num_joints_per_robot):
                ctrl_idx = robot_idx * num_joints_per_robot + j
                if ctrl_idx < model.nu and j * 3 + 2 < len(genotype):
                    amplitude = genotype[j * 3]
                    frequency = genotype[j * 3 + 1]
                    phase = genotype[j * 3 + 2]
                    
                    control_value = amplitude * np.sin(frequency * data.time + phase)
                    data.ctrl[ctrl_idx] = np.clip(
                        control_value, 
                        config.control_clip_min, 
                        config.control_clip_max
                    )
    
    def evaluate_population(self) -> list[float]:
        print(f"  Evaluating generation {self.generation + 1}")
        
        # Record start positions
        for i, individual in enumerate(self.population):
            individual.start_position = self.tracked_geoms[i].xpos.copy()
        
        # Set controller
        mujoco.set_mjcb_control(lambda m, d: self.spatial_controller(m, d))
        
        # Run simulation
        sim_steps = int(config.simulation_time / self.model.opt.timestep)
        for _ in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
        
        # Record end positions and calculate fitness
        fitness_values = []
        for i, individual in enumerate(self.population):
            individual.end_position = self.tracked_geoms[i].xpos.copy()
            distance = np.linalg.norm(individual.end_position - individual.start_position)
            individual.fitness = distance
            fitness_values.append(distance)
        
        return fitness_values

    def crossover(
        self, 
        parent1 : SpatialIndividual, 
        parent2 : SpatialIndividual
    ):
        """One-point crossover between two parents."""
        child1 = SpatialIndividual()
        child2 = SpatialIndividual()
        
        crossover_point = np.random.randint(1, len(parent1.genotype))
        
        child1.genotype = (parent1.genotype[:crossover_point] + 
                          parent2.genotype[crossover_point:])
        child2.genotype = (parent2.genotype[:crossover_point] + 
                          parent1.genotype[crossover_point:])
        
        return child1, child2
    
    def mutate(
        self, 
        individual : SpatialIndividual
    ):
        """Apply Gaussian mutation to individual's genotype."""
        mutated = SpatialIndividual()
        mutated.genotype = individual.genotype.copy()
        
        for i in range(len(mutated.genotype)):
            if np.random.random() < config.mutation_rate:
                # Add Gaussian noise
                mutated.genotype[i] += np.random.normal(0, config.mutation_strength)
                
                # Clamp values to reasonable ranges
                param_type = i % 3
                if param_type == 0:  # amplitude
                    mutated.genotype[i] = np.clip(
                        mutated.genotype[i], 
                        config.amplitude_min, 
                        config.amplitude_max
                    )
                elif param_type == 1:  # frequency
                    mutated.genotype[i] = np.clip(
                        mutated.genotype[i], 
                        config.frequency_min, 
                        config.frequency_max
                    )
                else:  # phase
                    mutated.genotype[i] = mutated.genotype[i] % config.phase_max
        
        return mutated
    
    def mating_movement_phase(
        self, 
        duration : float = 60.0, 
        save_trajectories : bool = True
    ):
        print(f"  Mating movement phase ({duration}s)...")
        
        # Calculate fitness-based attractiveness for each robot
        fitness_values = [ind.fitness for ind in self.population]
        max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
        
        # Normalize fitness scores (0 to 1)
        attractiveness = [f / max_fitness for f in fitness_values]
        
        # Track trajectories for visualization
        trajectories = [[] for _ in range(self.population_size)]
        sample_interval = max(1, int(duration / self.model.opt.timestep) // 100) 
        
        # Record initial positions
        for i in range(self.population_size):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])  # Store x, y only
        
        # Controller that biases movement towards attractive neighbors
        def mating_controller(model, data):
            num_joints_per_robot = self.num_joints
            
            for robot_idx, individual in enumerate(self.population):
                genotype = individual.genotype
                current_pos = self.tracked_geoms[robot_idx].xpos.copy()
                
                # Find most attractive neighbor
                best_neighbor_idx = None
                best_score = -1
                min_dist = float('inf')
                
                for other_idx, other in enumerate(self.population):
                    if other_idx == robot_idx:
                        continue
                    
                    other_pos = self.tracked_geoms[other_idx].xpos.copy()
                    distance = np.linalg.norm(current_pos - other_pos)
                    
                    # Score = attractiveness / distance (prefer close + fit partners)
                    if distance > 0.1:  # Avoid division by zero
                        score = attractiveness[other_idx] / distance
                        if score > best_score:
                            best_score = score
                            best_neighbor_idx = other_idx
                            min_dist = distance
                
                # Apply control with bias towards attractive neighbor
                for j in range(num_joints_per_robot):
                    ctrl_idx = robot_idx * num_joints_per_robot + j
                    if ctrl_idx < model.nu and j * 3 + 2 < len(genotype):
                        amplitude = genotype[j * 3]
                        frequency = genotype[j * 3 + 1]
                        phase = genotype[j * 3 + 2]
                        
                        # Base sinusoidal control
                        control_value = amplitude * np.sin(frequency * data.time + phase)
                        
                        # Add directional bias if found attractive neighbor
                        if best_neighbor_idx is not None and min_dist > 0.5:
                            neighbor_pos = self.tracked_geoms[best_neighbor_idx].xpos.copy()
                            direction = neighbor_pos - current_pos
                            direction_2d = np.array([direction[0], direction[1]])
                            
                            # Modulate control based on direction (simplified)
                            # This is a heuristic - you might need to tune this
                            bias = 0.1 * attractiveness[best_neighbor_idx] * np.sign(direction_2d[j % 2])
                            control_value += bias
                        
                        data.ctrl[ctrl_idx] = np.clip(
                            control_value,
                            config.control_clip_min,
                            config.control_clip_max
                        )
        
        # Run mating movement simulation and track positions
        mujoco.set_mjcb_control(lambda m, d: mating_controller(m, d))
        sim_steps = int(duration / self.model.opt.timestep)
        
        for step in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
            # Sample positions periodically
            if step % sample_interval == 0:
                for i in range(self.population_size):
                    pos = self.tracked_geoms[i].xpos.copy()
                    trajectories[i].append(pos[:2])
        
        # Record final positions
        for i in range(self.population_size):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
        
        print(f"  Mating movement complete!")
        
        # Save trajectory visualization
        if save_trajectories:
            self._save_mating_trajectories(trajectories, fitness_values, attractiveness)
    
    def _save_mating_trajectories(self, 
        trajectories : list[list[np.ndarray]], 
        fitness_values : list[float], 
        attractiveness : list[float]
    ):
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Draw world boundaries
        world_rect = patches.Rectangle(
            (0, 0), 
            config.world_size[0], 
            config.world_size[1],
            linewidth=2, 
            edgecolor='black', 
            facecolor='lightgray',
            alpha=0.1
        )
        ax.add_patch(world_rect)
        
        # Plot trajectories
        max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
        
        for i, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            
            # Color by fitness (use colormap)
            fitness_norm = fitness_values[i] / max_fitness
            color = plt.cm.viridis(fitness_norm)
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, alpha=0.6, linewidth=2, zorder=1)
            
            # Mark start position
            ax.plot(trajectory[0, 0], trajectory[0, 1], 
                   'o', color=color, markersize=10, 
                   markeredgecolor='black', markeredgewidth=1.5, 
                   alpha=0.8, zorder=2, label=f'Start {i}' if i < 5 else '')
            
            # Mark end position
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
                   's', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=2,
                   alpha=0.9, zorder=3)
            
            # Add robot ID at end position
            ax.text(trajectory[-1, 0], trajectory[-1, 1], str(i),
                   ha='center', va='center', fontsize=7, 
                   fontweight='bold', color='white', zorder=4)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=Normalize(vmin=0, vmax=max_fitness))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Fitness', fontsize=12, fontweight='bold')
        
        # Plot settings
        ax.set_xlim(-0.2, config.world_size[0] + 0.2)
        ax.set_ylim(-0.2, config.world_size[1] + 0.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        
        title = f'Mating Movement Trajectories - Generation {self.generation + 1}\n'
        title += f'Population: {self.population_size} | Duration: {config.simulation_time}s'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=10,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='Start Position'),
            Line2D([0], [0], marker='s', color='w',
                      markerfacecolor='gray', markersize=12,
                      markeredgecolor='black', markeredgewidth=2,
                      label='End Position'),
            Line2D([0], [0], color='gray', linewidth=2,
                      label='Trajectory')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{config.figures_folder}/mating_trajectories_gen_{self.generation + 1:03d}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved mating trajectories to {save_path}")
        plt.close()
    
    def create_next_generation(self):
        print(f"  Creating next generation with movement-based selection...")
        
        # Allow robots to move towards partners
        self.mating_movement_phase(duration=60.0, save_trajectories=True)
        
        new_population = []
        
        # Sort population by fitness for potential elitism
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)
        
        # Elitism: keep best individual
        if config.elitism:
            best = SpatialIndividual()
            best.genotype = sorted_pop[0].genotype.copy()
            new_population.append(best)

        # Pair up robots based on final proximity after movement
        pairs = []
        for i, individual in enumerate(self.population):
            current_pos = self.tracked_geoms[i].xpos.copy()
            closest_idx = None
            min_dist = float('inf')
            
            for j, other in enumerate(self.population):               
                other_pos = self.tracked_geoms[j].xpos.copy()
                distance = np.linalg.norm(current_pos - other_pos)
                
                if distance < min_dist:
                    min_dist = distance
                    closest_idx = j
            
            if closest_idx is not None:
                pairs.append((i, closest_idx))
        
        # Create offspring from pairs
        for parent1_idx, parent2_idx in pairs:
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = SpatialIndividual()
                child1.genotype = parent1.genotype.copy()
                child2 = SpatialIndividual()
                child2.genotype = parent2.genotype.copy()
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # Fill remaining slots with mutations of best individuals
        while len(new_population) < self.population_size:
            parent = np.random.choice(sorted_pop[:5])  # Top 5
            child = SpatialIndividual()
            child.genotype = parent.genotype.copy()
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population[:self.population_size]
    
    def run_evolution(self):
        print("=" * 60)
        print("SPATIAL EVOLUTIONARY ALGORITHM")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        print("=" * 60)
        
        # Initialize
        self.initialize_population()
        
        # Evolution loop
        for gen in range(self.num_generations):
            self.generation = gen
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            # Spawn population in simulation
            self.spawn_population()
            
            # Evaluate fitness
            fitness_values = self.evaluate_population()
            
            # Track statistics
            best_fitness = max(fitness_values)
            avg_fitness = np.mean(fitness_values)
            self.fitness_history.append(best_fitness)
            
            best_individual = max(self.population, key=lambda ind: ind.fitness)
            self.best_individual_history.append(best_individual)
            
            if config.print_generation_stats:
                print(f"  Best fitness: {best_fitness:.4f}")
                print(f"  Average fitness: {avg_fitness:.4f}")
                print(f"  Worst fitness: {min(fitness_values):.4f}")
            
            # Create next generation (except for last generation)
            if gen < self.num_generations - 1:
                self.create_next_generation()
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        return self.get_best_individual()
    
    def get_best_individual(self):
        return max(self.population, key=lambda ind: ind.fitness)
    
    def demonstrate_best(self):
        print(f"\n{'='*60}")
        print("DEMONSTRATING BEST INDIVIDUAL")
        print(f"{'='*60}")
        
        best = self.get_best_individual()
        print(f"Best fitness: {best.fitness:.4f}")
        
        if config.print_final_genotype:
            print("\nBest genotype (amplitude, frequency, phase per joint):")
            genotype = best.genotype
            for j in range(self.num_joints):
                if j * 3 + 2 < len(genotype):
                    amp, freq, phase = genotype[j*3], genotype[j*3+1], genotype[j*3+2]
                    print(f"  Joint {j}: amp={amp:.3f}, freq={freq:.3f}, phase={phase:.3f}")
        
        # Create single robot demo
        mujoco.set_mjcb_control(None)
        demo_world = SimpleFlatWorld(config.world_size)
        demo_robot = gecko()
        demo_world.spawn(demo_robot.spec, spawn_position=[0, 0, 0])
        demo_model = demo_world.spec.compile()
        demo_data = mujoco.MjData(demo_model)
        
        # Controller for single robot
        def demo_controller(model, data):
            genotype = best.genotype
            for j in range(self.num_joints):
                if j < model.nu and j * 3 + 2 < len(genotype):
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
        
        # Record video
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
        """Demonstrate the final evolved population."""
        print(f"\n{'='*60}")
        print("DEMONSTRATING FINAL POPULATION")
        print(f"{'='*60}")
        print(f"Recording {self.population_size} robots...")
        
        # Spawn final population
        self.spawn_population()
        
        # Set controller
        mujoco.set_mjcb_control(lambda m, d: self.spatial_controller(m, d))
        
        # Record video
        video_recorder = VideoRecorder(output_folder=config.video_folder)
        
        video_renderer(
            self.model,
            self.data,
            duration=config.multi_robot_demo_time,
            video_recorder=video_recorder,
        )
        
        print("Final population demonstration complete!")
    
    def plot_fitness_evolution(self):
        """Plot fitness over generations."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(self.fitness_history) + 1), 
                self.fitness_history, 'b-o', linewidth=2, markersize=6)
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Distance Traveled)")
        plt.title("Spatial EA: Evolution of Robot Movement")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = f"{config.figures_folder}/spatial_ea_fitness_evolution.png"
        plt.savefig(save_path)
        print(f"Fitness plot saved to {save_path}")
        plt.show()


def main():
    # Initialize world to get robot specs
    print("Initializing robot specifications...")
    temp_world = SimpleFlatWorld(config.world_size)
    temp_robot = gecko()
    temp_world.spawn(temp_robot.spec, spawn_position=[0, 0, 0])
    temp_model = temp_world.spec.compile()
    num_joints = temp_model.nu
    
    print(f"Robot has {num_joints} controllable joints")
    
    spatial_ea = SpatialEA(
        population_size=config.population_size,
        num_generations=config.num_generations,
        num_joints=num_joints
    )
    
    spatial_ea.run_evolution()
    
    # Demonstrate results
    spatial_ea.demonstrate_best()
    spatial_ea.demonstrate_final_population()
    
    # Plot results
    spatial_ea.plot_fitness_evolution()
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__": 
    main()
