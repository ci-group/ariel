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

# Import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld


# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

# Import configuration
from ea_config import config


class SpatialIndividual:
    def __init__(self, unique_id: int = None):
        self.genotype = []
        self.fitness = 0.0
        self.start_position = None
        self.end_position = None
        self.spawn_position = None
        self.robot_index = None
        self.unique_id = unique_id  
        self.parent_ids = []


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
        
        # Store current positions for cross-generation persistence
        self.current_positions = []
        
        # Counter for assigning unique IDs to individuals
        self.next_unique_id = 0
        
    def create_individual(self):
        individual = SpatialIndividual(unique_id=self.next_unique_id)
        self.next_unique_id += 1
        
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
        
        # Sanity check: verify positions match population size
        if len(self.current_positions) > 0 and len(self.current_positions) != self.population_size:
            print(f"  WARNING: Position count ({len(self.current_positions)}) != population size ({self.population_size})")
            print(f"  Regenerating all positions...")
            self.current_positions = []
        
        mujoco.set_mjcb_control(None)
        self.world = SimpleFlatWorld(config.world_size)
        self.robots = []
        
        ######### Determine spawn positions #########
        if len(self.current_positions) == self.population_size:
            print(f"  Using positions from previous generation")
            positions = [pos.copy() for pos in self.current_positions]
        else:
            # Generate new non-overlapping spawn positions for all robots
            print(f"  Generating new random spawn positions")
            positions = []
            min_distance = config.min_spawn_distance 
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
        """
        Controller that applies sinusoidal joint controls based on each robot's genotype.
        """
        num_joints_per_robot = self.num_joints
        
        num_spawned_robots = len(self.tracked_geoms)
        
        for robot_idx in range(min(num_spawned_robots, len(self.population))):
            individual = self.population[robot_idx]
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
        """
        Evaluate each individual in isolation and assign fitness based on distance traveled.
        """
        print(f"  Evaluating generation {self.generation + 1}")
        print(f"  Testing each robot in isolated environment...")
        
        # Create single isolated environment for reuse
        isolated_world = SimpleFlatWorld(config.world_size)
        isolated_robot = gecko()
        isolated_world.spawn(
            isolated_robot.spec, 
            spawn_position=[0, 0, 0.5],
            correct_for_bounding_box=False
        )
        isolated_model = isolated_world.spec.compile()
        
        # Get core geom name
        all_geoms = isolated_world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        core_geom_id = None
        for geom in all_geoms:
            if geom.name == "robot-0core":
                # Find the geom ID in the compiled model
                core_geom_id = mujoco.mj_name2id(isolated_model, mujoco.mjtObj.mjOBJ_GEOM, "robot-0core")
                break
        
        fitness_values = []
        
        # Evaluate each robot individually using the same environment
        for i, individual in enumerate(self.population):
            # Create fresh data for this evaluation
            isolated_data = mujoco.MjData(isolated_model)
            
            # Record start position
            mujoco.mj_forward(isolated_model, isolated_data)
            start_position = isolated_data.geom_xpos[core_geom_id].copy()
            
            # Controller for single robot
            def single_robot_controller(model, data):
                genotype = individual.genotype
                for j in range(min(model.nu, len(genotype) // 3)):
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
            
            # Set controller and run simulation
            mujoco.set_mjcb_control(single_robot_controller)
            sim_steps = int(config.simulation_time / isolated_model.opt.timestep)
            for _ in range(sim_steps):
                mujoco.mj_step(isolated_model, isolated_data)
            
            # Record end position and calculate fitness
            end_position = isolated_data.geom_xpos[core_geom_id].copy()
            distance = np.linalg.norm(end_position - start_position)
            
            individual.fitness = distance
            individual.start_position = start_position
            individual.end_position = end_position
            fitness_values.append(distance)
            
            if (i + 1) % 5 == 0:
                print(f"    Evaluated {i + 1}/{self.population_size} robots")
        
        print(f"  Fitness evaluation complete!")
        return fitness_values

    def crossover(
        self, 
        parent1 : SpatialIndividual, 
        parent2 : SpatialIndividual
    ):
        """One-point crossover between two parents."""
        child1 = SpatialIndividual(unique_id=self.next_unique_id)
        self.next_unique_id += 1
        child1.parent_ids = [parent1.unique_id, parent2.unique_id]
        
        child2 = SpatialIndividual(unique_id=self.next_unique_id)
        self.next_unique_id += 1
        child2.parent_ids = [parent1.unique_id, parent2.unique_id]
        
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
        mutated = SpatialIndividual(unique_id=self.next_unique_id)
        self.next_unique_id += 1
        mutated.parent_ids = [individual.unique_id]  # Track parent for mutations
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
        save_trajectories : bool = True,
        record_video : bool = False
    ):
        print(f"  Mating movement phase ({duration}s)...")
        
        # Calculate fitness-based attractiveness for each robot
        fitness_values = [ind.fitness for ind in self.population]
        max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
        
        # Normalize fitness scores (0 to 1)
        attractiveness = [f / max_fitness for f in fitness_values]
        
        # Track trajectories for visualization
        # Only track robots that are actually spawned
        num_spawned = len(self.tracked_geoms)
        trajectories = [[] for _ in range(num_spawned)]
        sample_interval = max(1, int(duration / self.model.opt.timestep) // 100) 
        
        print(f"  Tracking trajectories for {num_spawned} spawned robots")
        print(f"  Sample interval: {sample_interval} steps")
        
        # Record initial positions
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])  # Store x, y only
        
        # Controller that biases movement towards attractive neighbors
        # Use reciprocal attraction: both robots move toward each other
        def mating_controller(model, data):
            num_joints_per_robot = self.num_joints
            num_spawned_robots = len(self.tracked_geoms)
            
            for robot_idx in range(min(num_spawned_robots, len(self.population))):
                individual = self.population[robot_idx]
                genotype = individual.genotype
                current_pos = self.tracked_geoms[robot_idx].xpos.copy()
                
                # Find most attractive neighbor
                best_neighbor_idx = None
                best_score = -1
                min_dist = float('inf')
                
                for other_idx in range(min(num_spawned_robots, len(self.population))):
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
                        # ALL robots move toward their most attractive neighbor
                        if best_neighbor_idx is not None and min_dist > 0.5:
                            neighbor_pos = self.tracked_geoms[best_neighbor_idx].xpos.copy()
                            direction = neighbor_pos - current_pos
                            direction_2d = np.array([direction[0], direction[1]])
                            
                            # Simple directional bias scaled by neighbor's attractiveness
                            # Higher fitness neighbors = stronger attraction
                            bias = 0.2 * attractiveness[best_neighbor_idx] * np.sign(direction_2d[j % 2])
                            control_value += bias
                        
                        data.ctrl[ctrl_idx] = np.clip(
                            control_value,
                            config.control_clip_min,
                            config.control_clip_max
                        )
        
        # Set up video recording if requested
        video_recorder = None
        renderer = None
        if record_video:
            from ariel.utils.video_recorder import VideoRecorder
            video_name = f"generation_{self.generation + 1:03d}_mating_movement"
            video_recorder = VideoRecorder(
                file_name=video_name,
                output_folder=config.video_folder
            )
            
            # Create renderer for video
            scene_option = mujoco.MjvOption()
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            renderer = mujoco.Renderer(
                self.model,
                width=video_recorder.width,
                height=video_recorder.height
            )
            
            # Calculate steps per frame for smooth video
            steps_per_frame = max(1, int(self.model.opt.timestep * duration * video_recorder.fps / duration))
            print(f"  Recording video: {video_name}")
            print(f"  Steps per frame: {steps_per_frame}")
        
        # Run mating movement simulation and track positions
        mujoco.set_mjcb_control(lambda m, d: mating_controller(m, d))
        sim_steps = int(duration / self.model.opt.timestep)
        
        for step in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
            # Sample positions periodically
            if step % sample_interval == 0:
                for i in range(num_spawned):
                    pos = self.tracked_geoms[i].xpos.copy()
                    trajectories[i].append(pos[:2])
            print(f"    Mating step {step + 1}/{sim_steps}", end='\r')
            # Record video frame if enabled
            if record_video and renderer is not None and video_recorder is not None:
                if step % steps_per_frame == 0:
                    renderer.update_scene(self.data, scene_option=scene_option)
                    video_recorder.write(frame=renderer.render())
        
        # Record final positions
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
        
        print(f"  Mating movement complete!")
        print(f"  Trajectory points per robot: ~{len(trajectories[0])} samples")
        
        # Clean up video recording
        if record_video and video_recorder is not None:
            video_recorder.release()
            print(f"  Video saved: {video_recorder.frame_count} frames")
            if renderer is not None:
                renderer.close()
                
        self.current_positions = []
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            self.current_positions.append(pos)
        print(f"  Updated positions for next generation")
        print(f"    Tracked {len(self.current_positions)} positions")
        if len(self.current_positions) >= 3:
            print(f"    Sample positions: {self.current_positions[0][:2]}, {self.current_positions[1][:2]}, {self.current_positions[2][:2]}")
    
        # Save trajectory visualization
        if save_trajectories:
            self._save_mating_trajectories(trajectories, fitness_values, attractiveness)
    
    def _calculate_marker_size(self, robot_size_meters, ax, fig):
        # Get axis bounds in data coordinates
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        # Get figure size in inches
        fig_width, fig_height = fig.get_size_inches()
        
        # Calculate data units per inch
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        data_per_inch_x = data_width / fig_width
        data_per_inch_y = data_height / fig_height
        
        # Use average to account for aspect ratio
        data_per_inch = (data_per_inch_x + data_per_inch_y) / 2
        
        # Convert robot size from data units (meters) to inches
        robot_size_inches = robot_size_meters / data_per_inch
        
        # Convert inches to points (1 inch = 72 points)
        # For circular markers, markersize is the diameter in points
        marker_size_points = robot_size_inches * 72
        
        return marker_size_points
    
    def _save_mating_trajectories(self, 
        trajectories : list[list[np.ndarray]], 
        fitness_values : list[float], 
        attractiveness : list[float]
    ):
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Set axis limits first (needed for marker size calculation)
        ax.set_xlim(-0.2, config.world_size[0] + 0.2)
        ax.set_ylim(-0.2, config.world_size[1] + 0.2)
        ax.set_aspect('equal')
        
        # Calculate marker size to match robot size (from config)
        marker_size = self._calculate_marker_size(config.robot_size, ax, fig)
        
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
        
        # Get unique IDs from population
        unique_ids = [ind.unique_id for ind in self.population]
        
        # Create color mapping based on unique_id
        id_colors = [plt.cm.tab20(uid % 20 / 20) for uid in unique_ids]
        
        # Fitness range for context
        min_fitness = min(fitness_values)
        max_fitness = max(fitness_values) if max(fitness_values) > 0 else 1.0
        
        for i, trajectory in enumerate(trajectories):
            trajectory = np.array(trajectory)
            
            color = id_colors[i]
            
            # Plot trajectory line
            ax.plot(trajectory[:, 0], trajectory[:, 1], 
                   color=color, alpha=0.6, linewidth=2, zorder=1)
            
            # Mark start position (circle) - size matches robot
            ax.plot(trajectory[0, 0], trajectory[0, 1], 
                   'o', color=color, markersize=marker_size, 
                   markeredgecolor='black', markeredgewidth=1.5, 
                   alpha=0.8, zorder=2)
            
            # Mark end position (square) - size matches robot
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 
                   's', color=color, markersize=marker_size,
                   markeredgecolor='black', markeredgewidth=2,
                   alpha=0.9, zorder=3)
            
            # Add unique_id at end position
            ax.text(trajectory[-1, 0], trajectory[-1, 1], str(unique_ids[i]),
                   ha='center', va='center', fontsize=7, 
                   fontweight='bold', color='white', zorder=4)
            
            # Add fitness as small annotation near start position
            ax.text(trajectory[0, 0] + 0.15, trajectory[0, 1] + 0.15, 
                   f'{fitness_values[i]:.2f}',
                   ha='left', va='bottom', fontsize=6, 
                   color='black', alpha=0.7, zorder=4)
        
        
        # Plot settings (axis limits already set above for marker size calculation)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        
        title = f'Mating Movement Trajectories - Generation {self.generation + 1}\n'
        title += f'Population: {self.population_size} | Duration: {config.simulation_time}s\n'
        title += f'Fitness Range: {min_fitness:.3f} - {max_fitness:.3f}\n'
        title += f'Colors: Unique Individual IDs | Numbers: Individual IDs | Small text: Fitness'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add legend with matching marker sizes
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='gray', markersize=marker_size,
                      markeredgecolor='black', markeredgewidth=1.5,
                      label='Start Position (robot size)'),
            Line2D([0], [0], marker='s', color='w',
                      markerfacecolor='gray', markersize=marker_size,
                      markeredgecolor='black', markeredgewidth=2,
                      label='End Position (robot size, with ID)'),
            Line2D([0], [0], color='gray', linewidth=2,
                      label='Trajectory (colored by individual)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{config.figures_folder}/mating_trajectories_gen_{self.generation + 1:03d}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved mating trajectories to {save_path}")
        plt.close()

    def create_next_generation(self, record_video: bool = False):
        print(f"  Creating next generation with movement-based selection...")
        
        # Allow robots to move towards partners
        self.mating_movement_phase(
            duration=config.simulation_time, 
            save_trajectories=True,
            record_video=record_video
        )
        
        new_population = []
        new_positions = []  # Track which position each offspring inherits
        
        # Sort population by fitness for potential elitism
        sorted_pop = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)

        # Find best fitness partner within pairing radius
        pairs = []
        pair_positions = []  # Track positions for each pair
        paired_indices = set()  # Track which robots have been paired
        
        # Sort population by fitness (descending) to prioritize high-fitness individuals
        fitness_ranking = sorted(enumerate(self.population), 
                               key=lambda x: x[1].fitness, reverse=True)
        
        for idx, individual in fitness_ranking:
            if idx in paired_indices:
                continue  # Already paired
            
            current_pos = self.tracked_geoms[idx].xpos.copy()
            
            # Find highest fitness partner within pairing radius
            best_partner_idx = None
            best_partner_fitness = -1
            
            for other_idx, other_ind in enumerate(self.population):
                if other_idx == idx or other_idx in paired_indices:
                    continue
                
                other_pos = self.tracked_geoms[other_idx].xpos.copy()
                distance = np.linalg.norm(current_pos - other_pos)
                
                # Check if within pairing radius and has higher fitness than current best
                if distance <= config.pairing_radius and other_ind.fitness > best_partner_fitness:
                    best_partner_fitness = other_ind.fitness
                    best_partner_idx = other_idx
            
            # If found a partner within radius, create pair
            if best_partner_idx is not None:
                pairs.append((idx, best_partner_idx))
                paired_indices.add(idx)
                paired_indices.add(best_partner_idx)
                
                # Store positions for offspring
                parent1_pos = self.current_positions[idx]
                parent2_pos = self.current_positions[best_partner_idx]
                
                # Calculate random positions on circle edge around each parent
                # Child 1: random point on circle of radius offspring_radius around parent1
                angle1 = np.random.uniform(0, 2 * np.pi)
                child1_offset = np.array([
                    config.offspring_radius * np.cos(angle1),
                    config.offspring_radius * np.sin(angle1),
                    0.0
                ])
                child1_pos = parent1_pos + child1_offset
                
                # Child 2: random point on circle of radius offspring_radius around parent2
                angle2 = np.random.uniform(0, 2 * np.pi)
                child2_offset = np.array([
                    config.offspring_radius * np.cos(angle2),
                    config.offspring_radius * np.sin(angle2),
                    0.0
                ])
                child2_pos = parent2_pos + child2_offset
                
                # Clamp to world bounds
                child1_pos[0] = np.clip(child1_pos[0], 0, config.world_size[0])
                child1_pos[1] = np.clip(child1_pos[1], 0, config.world_size[1])
                child2_pos[0] = np.clip(child2_pos[0], 0, config.world_size[0])
                child2_pos[1] = np.clip(child2_pos[1], 0, config.world_size[1])
                
                pair_positions.append((child1_pos, child2_pos))
        
        print(f"  Created {len(pairs)} pairs from {self.population_size} robots")
        print(f"  Unpaired robots: {self.population_size - len(paired_indices)}")
        
        # Create offspring from pairs
        for pair_idx, (parent1_idx, parent2_idx) in enumerate(pairs):
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover
            if np.random.random() < config.crossover_rate:
                child1, child2 = self.crossover(parent1, parent2)
            else:
                # No crossover - clone parents with new unique IDs
                child1 = SpatialIndividual(unique_id=self.next_unique_id)
                self.next_unique_id += 1
                child1.genotype = parent1.genotype.copy()
                child1.parent_ids = [parent1.unique_id]
                
                child2 = SpatialIndividual(unique_id=self.next_unique_id)
                self.next_unique_id += 1
                child2.genotype = parent2.genotype.copy()
                child2.parent_ids = [parent2.unique_id]
            
            # Mutation
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            
            new_population.append(child1)
            new_positions.append(pair_positions[pair_idx][0])
            
            new_population.append(child2)
            new_positions.append(pair_positions[pair_idx][1])
        
        # # Fill remaining slots with mutations of best individuals
        # while len(new_population) < self.population_size:
        #     parent = np.random.choice(sorted_pop[:5])  # Top 5
        #     parent_idx = self.population.index(parent)
        #     child = self.mutate(parent)  # mutate() already creates new individual with unique_id
        #     new_population.append(child)
            
    #         # Inherit parent's position with small random offset to avoid exact overlap
    #         offset = np.random.normal(0, 0.1, size=3)
    #         offset[2] = 0  # Keep z coordinate unchanged
    #         new_pos = self.current_positions[parent_idx].copy() + offset
    #         # Clamp to world bounds
    #         new_pos[0] = np.clip(new_pos[0], 0, config.world_size[0])
    #         new_pos[1] = np.clip(new_pos[1], 0, config.world_size[1])
    #         new_positions.append(new_pos)
        
        # Extend population with offspring
        self.population.extend(new_population)
        self.current_positions.extend(new_positions)
        
        # Verify consistency
        if len(self.population) != len(self.current_positions):
            print(f"  ERROR: Population size ({len(self.population)}) != Position count ({len(self.current_positions)})")
            raise RuntimeError("Population and position arrays out of sync!")
        
        # Update population size
        old_size = self.population_size
        self.population_size = len(self.population)
        
        print(f"  Population extended from {old_size} to {self.population_size} individuals")
        print(f"  Added {len(new_population)} offspring")
        print(f"  Position tracking updated: {len(self.current_positions)} positions")
        print(f"  Paired individuals: {len(paired_indices)} out of {old_size}")

    
    def run_evolution(self, record_generation_videos: bool = False):
        print("=" * 60)
        print("SPATIAL EVOLUTIONARY ALGORITHM")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        if record_generation_videos:
            print(f"Video recording: ENABLED (one video per generation)")
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
                self.create_next_generation(record_video=record_generation_videos)
            else:
                # Run mating movement to capture final positions
                self.mating_movement_phase(
                    duration=config.simulation_time, 
                    save_trajectories=True,
                    record_video=record_generation_videos
                )

        
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
        
        # Diagnostic: Check model consistency
        print(f"\nDiagnostics:")
        print(f"  Evolution num_joints (self.num_joints): {self.num_joints}")
        print(f"  Demo model actuators (demo_model.nu): {demo_model.nu}")
        print(f"  Best genotype length: {len(best.genotype)}")
        print(f"  Expected genotype length: {self.num_joints * 3}")
        
        # Controller for single robot
        def demo_controller(model, data):
            genotype = best.genotype
            # Use demo_model.nu instead of self.num_joints to match actual model
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
    
    # Run evolution with optional video recording for each generation
    # Set to True to record video for each generation's mating movement phase
    spatial_ea.run_evolution(record_generation_videos=False)
    
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
