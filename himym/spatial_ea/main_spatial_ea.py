import mujoco
import numpy as np

from ea_config import config
from spatial_individual import SpatialIndividual
from genetic_operators import (
    create_initial_hyperneat_genome,
    crossover_hyperneat,
    mutate_hyperneat,
    clone_individual
)
from selection import apply_selection
from evaluation import evaluate_population
from visualization import save_mating_trajectories
from parent_selection import find_pairs, calculate_offspring_positions, generate_random_zone_centers
from evolution_data_collector import EvolutionDataCollector
from simulation_utils import (
    generate_spawn_positions,
    spawn_population_in_world,
    get_tracked_geoms,
    create_mating_controller,
    track_trajectories,
    update_trajectories
)
from visualize_experiment import ExperimentVisualizer
from incubation import IncubationEvolution, seed_spatial_population_from_incubation
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from periodic_boundary_utils import apply_periodic_boundaries_to_simulation


class SpatialEA:
    """Spatial Evolutionary Algorithm using proximity-based mating in physical simulation."""
    
    def __init__(
        self, 
        population_size: int | None = None, 
        num_generations: int | None = None, 
        num_joints: int | None = None
    ):
        self.population_size = population_size or config.population_size
        self.num_generations = num_generations or config.num_generations
        self.num_joints = num_joints
        
        self.population: list[SpatialIndividual] = []
        self.generation = 0
        self.fitness_history: list[float] = []
        self.best_individual_history: list[SpatialIndividual] = []
        self.data_collector = EvolutionDataCollector(config=config)
        
        self.world = None
        self.model = None
        self.data = None
        self.robots = []
        self.tracked_geoms = []
        self.current_positions: list[np.ndarray] = []
        self.current_orientations: list[float] = []
        self.next_unique_id = 0
        
        self.current_zone_centers: list[tuple[float, float]] = []
        self.assigned_zones: dict[int, int] = {}
        self._initialize_mating_zones()
    
    def _initialize_mating_zones(self) -> None:
        """Initialize mating zone positions."""
        if config.pairing_method != "mating_zone" and config.movement_bias != "nearest_zone":
            return
        
        if config.num_mating_zones == 1:
            # Use configured center for single zone
            self.current_zone_centers = [tuple(config.mating_zone_center)]
        else:
            # Generate random positions for multiple zones
            self.current_zone_centers = generate_random_zone_centers(
                num_zones=config.num_mating_zones,
                world_size=(config.world_size[0], config.world_size[1]),
                zone_radius=config.mating_zone_radius,
                min_zone_distance=config.min_zone_distance
            )
        
        print(f"  Initialized {len(self.current_zone_centers)} mating zone(s)")
        if config.pairing_method == "mating_zone":
            print(f"  Zones used for: pairing")
        if config.movement_bias == "nearest_zone":
            print(f"  Zones used for: movement bias")
        
        # Print zone relocation strategy
        strategy = config.zone_relocation_strategy
        if strategy == "static":
            print(f"  Zone relocation: static (zones never move)")
        elif strategy == "generation_interval":
            print(f"  Zone relocation: every {config.zone_change_interval} generations")
        elif strategy == "event_driven":
            print(f"  Zone relocation: event-driven (zones relocate after matings)")
        else:
            print(f"  Zone relocation: {strategy}")
    
    def cleanup_simulation_resources(self) -> None:
        """
        Explicitly clean up simulation resources to prevent memory leaks.
        Call this after each generation or when done with simulation.
        """
        import gc
        
        # Clear MuJoCo model and data
        if self.data is not None:
            del self.data
            self.data = None
        
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear world
        if self.world is not None:
            del self.world
            self.world = None
        
        # Clear robot references
        self.robots = []
        self.tracked_geoms = []
        
        # Force garbage collection
        gc.collect()
    
    def _assign_zones_to_population(self) -> None:
        """Assign each individual to a random mating zone."""
        if not self.current_zone_centers or len(self.current_zone_centers) == 0:
            return
        
        # Assign each individual a random zone
        for individual in self.population:
            zone_idx = np.random.randint(0, len(self.current_zone_centers))
            self.assigned_zones[individual.unique_id] = zone_idx
    
    def _update_mating_zones(self) -> None:
        """Update mating zone positions if dynamic zones are enabled."""
        
        # Only update zones if they're being used
        if config.pairing_method != "mating_zone" and config.movement_bias not in ["nearest_zone", "assigned_zone"]:
            return
        
        strategy = config.zone_relocation_strategy
        
        if strategy == "static":
            # Never move zones
            return
        
        if strategy == "generation_interval":
            # Check if it's time to change zones
            if self.generation > 0 and self.generation % config.zone_change_interval == 0:
                print(f"\n  UPDATING MATING ZONES (Generation {self.generation})")
                self.current_zone_centers = generate_random_zone_centers(
                    num_zones=config.num_mating_zones,
                    world_size=(config.world_size[0], config.world_size[1]),
                    zone_radius=config.mating_zone_radius,
                    min_zone_distance=config.min_zone_distance
                )
                
                print(f"  New zone centers: {self.current_zone_centers}")
                
                # Reassign zones when they change (for assigned_zone movement bias)
                if config.movement_bias == "assigned_zone":
                    self._assign_zones_to_population()
                    print(f"  Reassigned zones to all {len(self.population)} individuals")
        
        # Note: event_driven relocation is handled in create_next_generation after pairing
    
    def _relocate_zones(self, zone_indices: set[int]) -> None:
        """
        Relocate specific mating zones to new random positions.
        
        Args:
            zone_indices: Set of zone indices to relocate
        """
        if not zone_indices or len(self.current_zone_centers) == 0:
            return
        
        print(f"\n  EVENT-DRIVEN ZONE RELOCATION")
        print(f"  Relocating {len(zone_indices)} zones that had matings: {sorted(zone_indices)}")
        
        # For each zone to relocate, generate a new position
        for zone_idx in zone_indices:
            if zone_idx < len(self.current_zone_centers):
                # Generate new position that doesn't overlap with existing zones
                max_attempts = 100
                for attempt in range(max_attempts):
                    # Generate random position
                    margin = 1.0
                    zone_radius = config.mating_zone_radius
                    x_min = margin + zone_radius
                    x_max = config.world_size[0] - margin - zone_radius
                    y_min = margin + zone_radius
                    y_max = config.world_size[1] - margin - zone_radius
                    
                    new_x = np.random.uniform(x_min, x_max)
                    new_y = np.random.uniform(y_min, y_max)
                    new_center = (new_x, new_y)
                    
                    min_distance = zone_radius * config.min_zone_distance
                    too_close = False
                    for other_idx, other_center in enumerate(self.current_zone_centers):
                        if other_idx == zone_idx:
                            continue
                        distance = np.sqrt((new_x - other_center[0])**2 + (new_y - other_center[1])**2)
                        if distance < min_distance:
                            too_close = True
                            break
                    
                    if not too_close:
                        old_center = self.current_zone_centers[zone_idx]
                        self.current_zone_centers[zone_idx] = new_center
                        print(f"    Zone {zone_idx}: {old_center} -> {new_center}")
                        break
                else:
                    print(f"    Warning: Could not find non-overlapping position for zone {zone_idx}")
        
        if config.movement_bias == "assigned_zone":
            reassigned_count = 0
            for individual in self.population:
                if individual.unique_id in self.assigned_zones:
                    old_zone = self.assigned_zones[individual.unique_id]
                    if old_zone in zone_indices:
                        new_zone = np.random.randint(0, len(self.current_zone_centers))
                        self.assigned_zones[individual.unique_id] = new_zone
                        reassigned_count += 1
            
            print(f"  Reassigned {reassigned_count} individuals from relocated zones")
    
    
    def create_individual(self) -> SpatialIndividual:
        """Create a new individual with random genotype."""
        individual = SpatialIndividual(
            unique_id=self.next_unique_id, 
            generation=self.generation,
            initial_energy=config.initial_energy
        )
        self.next_unique_id += 1
        individual.genotype = create_initial_hyperneat_genome(
            num_inputs=4,
            num_outputs=1,
            activation='sine'
        )
        return individual
    
    def initialize_population(self) -> None:
        """Initialize population with random individuals or from incubation."""
        if config.incubation_enabled:
            # Run incubation phase
            print("\n" + "="*60)
            print("STARTING INCUBATION PHASE")
            print("="*60)
            
            incubator = IncubationEvolution(
                population_size=config.incubation_population_size,
                num_generations=config.incubation_num_generations,
                num_joints=self.num_joints,
                world_size=config.world_size,
                simulation_time=config.simulation_time,
                control_clip_min=config.control_clip_min,
                control_clip_max=config.control_clip_max,
                mutation_rate=config.incubation_mutation_rate,
                mutation_power=config.incubation_mutation_power,
                add_connection_rate=config.incubation_add_connection_rate,
                add_node_rate=config.incubation_add_node_rate,
                crossover_rate=config.incubation_crossover_rate,
                tournament_size=config.incubation_tournament_size,
                elitism_count=config.incubation_elitism_count,
                use_directional_fitness=config.incubation_use_directional_fitness,
                target_distance_min=config.incubation_target_distance_min,
                target_distance_max=config.incubation_target_distance_max,
                progress_weight=config.incubation_progress_weight,
                distance_weight=config.incubation_distance_weight
            )
            
            # Run incubation evolution
            incubation_population = incubator.run()
            
            # Demonstrate best incubation individual
            incubator.demonstrate_best(duration=15.0)
            
            # Update next_unique_id to continue from incubation
            self.next_unique_id = incubator.next_unique_id
            
            # Seed spatial population from incubation results
            self.population, self.next_unique_id = seed_spatial_population_from_incubation(
                incubation_population=incubation_population,
                target_population_size=self.population_size,
                starting_generation=0,
                next_unique_id=self.next_unique_id,
                world_size=config.world_size,
                spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
                spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
                min_spawn_distance=config.min_spawn_distance,
                initial_energy=config.initial_energy
            )
            
            # Store initial positions from seeding
            self.current_positions = [np.array([ind.x, ind.y, config.spawn_z]) for ind in self.population]
            
            # Initialize orientations as empty (will be generated randomly at first spawn)
            self.current_orientations = []
            
            # Assign zones if using assigned_zone movement bias
            if config.movement_bias == "assigned_zone":
                self._assign_zones_to_population()
                print(f"  Assigned zones to {len(self.population)} individuals from incubation")
            
            print("\nTransition to spatial evolution complete")
            print("="*60)
        else:
            # Standard random initialization
            print(f"Initializing population of {self.population_size} individuals")
            self.population = [self.create_individual() for _ in range(self.population_size)]
            
            if config.movement_bias == "assigned_zone":
                self._assign_zones_to_population()
    
    def spawn_population(self) -> None:
        """Spawn population in simulation world."""
        # Verify positions match population size
        if len(self.current_positions) > 0 and len(self.current_positions) != self.population_size:
            self.current_positions = []
            self.current_orientations = []
        
        # Determine spawn positions
        if len(self.current_positions) == self.population_size:
            positions = [pos.copy() for pos in self.current_positions]
        else:
            positions = generate_spawn_positions(
                population_size=self.population_size,
                spawn_x_range=(config.spawn_x_min, config.spawn_x_max),
                spawn_y_range=(config.spawn_y_min, config.spawn_y_max),
                spawn_z=config.spawn_z,
                min_spawn_distance=config.min_spawn_distance
            )
        
        # Determine spawn orientations
        if len(self.current_orientations) == self.population_size:
            orientations = self.current_orientations.copy()
        else:
            # Random yaw angles between 0 and 2*pi
            orientations = [np.random.uniform(0, 2 * np.pi) for _ in range(self.population_size)]
        
        # Spawn robots
        self.world, self.model, self.data, self.robots = spawn_population_in_world(
            population=self.population,
            positions=positions,
            world_size=config.world_size,
            orientations=orientations
        )
        
        # Store the orientations that were used for spawning
        self.current_orientations = orientations
        
        # Track robot geoms
        self.tracked_geoms = get_tracked_geoms(
            world=self.world,
            data=self.data,
            population_size=self.population_size
        )
        
        print(f"Spawned {len(self.population)} robots successfully")
    
    def evaluate_population_fitness(self) -> list[float]:
        """Evaluate fitness for new individuals using evaluate-once strategy."""
        unevaluated = [ind for ind in self.population if not ind.evaluated]
        
        if unevaluated:
            print(f"  Evaluating {len(unevaluated)} new individuals")
            
            fitness_values = evaluate_population(
                population=unevaluated,
                world_size=config.world_size,
                simulation_time=config.simulation_time,
                control_clip_min=config.control_clip_min,
                control_clip_max=config.control_clip_max,
                use_directional_fitness=config.use_directional_fitness,
                target_distance_min=config.target_distance_min,
                target_distance_max=config.target_distance_max,
                progress_weight=config.progress_weight
            )
            
            for ind in unevaluated:
                ind.evaluated = True
        
        return [ind.fitness for ind in self.population]
    
    def mating_movement_phase(
        self, 
        duration: float = 60.0, 
        save_trajectories: bool = True
    ) -> None:
        """Run mating movement phase where robots move towards attractive neighbors."""
        fitness_values = [ind.fitness for ind in self.population]
        num_spawned = len(self.tracked_geoms)
        
        sample_interval = max(1, int(duration / self.model.opt.timestep) // 100)
        trajectories = track_trajectories(self.tracked_geoms, sample_interval)
        
        controller = create_mating_controller(
            population=self.population,
            tracked_geoms=self.tracked_geoms,
            num_joints=self.num_joints,
            control_clip_min=config.control_clip_min,
            control_clip_max=config.control_clip_max,
            movement_bias=config.movement_bias,
            world_size=config.world_size,
            use_periodic_boundaries=config.use_periodic_boundaries,
            mating_zone_centers=self.current_zone_centers if self.current_zone_centers else None,
            mating_zone_radius=config.mating_zone_radius,
            assigned_zones=self.assigned_zones if config.movement_bias == "assigned_zone" else None
        )
        
        # Set up video recording if requested
        video_recorder = None
        renderer = None
        snapshot_saved = False
        
        if config.record_generation_videos or config.save_generation_snapshots:
            scene_option = mujoco.MjvOption()
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
            
            # Create renderer for video and/or snapshots
            # Note: Resolution limited by model's offwidth/offheight (1280x960 in SimpleFlatWorld)
            # Snapshots use max resolution, video uses smaller for file size
            render_width = 1280 if config.save_generation_snapshots else 1280
            render_height = 960 if config.save_generation_snapshots else 720
            
            renderer = mujoco.Renderer(
                self.model,
                width=render_width,
                height=render_height
            )
            
        if config.record_generation_videos:
            video_name = f"generation_{self.generation + 1:03d}_mating_movement"
            video_recorder = VideoRecorder(
                file_name=video_name,
                output_folder=config.video_folder
            )
            
            # Override renderer size for video
            renderer = mujoco.Renderer(
                self.model,
                width=video_recorder.width,
                height=video_recorder.height
            )
            
            steps_per_frame = max(
                1, 
                int(self.model.opt.timestep * duration * video_recorder.fps / duration)
            )
            print(f"  Recording video: {video_name}")
            print(f"  Steps per frame: {steps_per_frame}")
        
        # Run mating movement simulation
        mujoco.set_mjcb_control(controller)
        sim_steps = int(duration / self.model.opt.timestep)
        snapshot_step = sim_steps // 2  # Capture snapshot at midpoint
        
        for step in range(sim_steps):
            mujoco.mj_step(self.model, self.data)
            
            # PERIODIC BOUNDARY FOR MOVEMENT
            if config.use_periodic_boundaries:
                
                apply_periodic_boundaries_to_simulation(
                    self.tracked_geoms, 
                    (config.world_size[0], config.world_size[1])
                )
            
            # Sample positions periodically
            update_trajectories(trajectories, self.tracked_geoms, step, sample_interval)
            
            print(f"    Mating step {step + 1}/{sim_steps}", end='\r')
            
            # Save snapshot at midpoint (if enabled and not already saved)
            if (config.save_generation_snapshots and renderer is not None and 
                not snapshot_saved and step == snapshot_step):
                scene_option = mujoco.MjvOption()
                scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                renderer.update_scene(self.data, scene_option=scene_option)
                snapshot_frame = renderer.render()
                
                # Save snapshot as PNG
                import imageio
                snapshot_path = f"{config.figures_folder}/generation_{self.generation + 1:03d}_snapshot.png"
                imageio.imwrite(snapshot_path, snapshot_frame)
                snapshot_saved = True
                print(f"\n  Saved snapshot: generation_{self.generation + 1:03d}_snapshot.png")
            
            # Record video frame if enabled
            if config.record_generation_videos and renderer is not None and video_recorder is not None:
                scene_option = mujoco.MjvOption()
                scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                if step % steps_per_frame == 0:
                    renderer.update_scene(self.data, scene_option=scene_option)
                    video_recorder.write(frame=renderer.render())
        
        # Record final positions
        for i in range(num_spawned):
            pos = self.tracked_geoms[i].xpos.copy()
            trajectories[i].append(pos[:2])
        
        print(f"  MATING MOVEMENT COMPLETE")
        
        # Clean up video recording
        if config.record_generation_videos and video_recorder is not None:
            video_recorder.release()
            print(f"  Video saved: {video_recorder.frame_count} frames")
            if renderer is not None:
                renderer.close()
        elif config.save_generation_snapshots and renderer is not None:
            # Close renderer if it was only used for snapshots
            renderer.close()
        
        # Update positions for next generation using robot_index mapping
        pop_size = len(self.population)
        old_positions = self.current_positions.copy() if self.current_positions else []
        old_orientations = self.current_orientations.copy() if self.current_orientations else []
        
        self.current_positions = []
        self.current_orientations = []
        
        for i in range(pop_size):
            robot_idx = self.population[i].robot_index
            
            # Use old position if robot_index is invalid (offspring or edge cases)
            if robot_idx is None or robot_idx >= num_spawned:
                if i < len(old_positions):
                    self.current_positions.append(old_positions[i].copy())
                    self.current_orientations.append(old_orientations[i] if i < len(old_orientations) else 0.0)
                else:
                    self.current_positions.append(np.array([
                        np.random.uniform(config.spawn_x_min, config.spawn_x_max),
                        np.random.uniform(config.spawn_y_min, config.spawn_y_max),
                        config.spawn_z
                    ]))
                    self.current_orientations.append(np.random.uniform(0, 2 * np.pi))
                continue
            
            # Read current position from simulation using robot_index
            pos = self.tracked_geoms[robot_idx].xpos.copy()
            self.current_positions.append(pos)
            
            # Extract orientation (yaw) from quaternion
            joint_name = f"robot-{robot_idx}"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                qw = self.data.qpos[qpos_addr + 3]
                qx = self.data.qpos[qpos_addr + 4]
                qy = self.data.qpos[qpos_addr + 5]
                qz = self.data.qpos[qpos_addr + 6]
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
                self.current_orientations.append(yaw)
            else:
                self.current_orientations.append(0.0)
        
        # Save trajectory visualization
        if save_trajectories and pop_size > 0:
            save_path = f"{config.figures_folder}/mating_trajectories_gen_{self.generation + 1:03d}.png"
            # Handle case where tracked_geoms may be larger than current population
            # (e.g., after selection reduced population but geoms already spawned)
            pop_size = len(self.population)
            vis_population = self.population[:pop_size]
            vis_fitness = fitness_values[:pop_size]
            
            # Reorder trajectories to match population order using robot_index
            vis_trajectories = []
            for ind in vis_population:
                if ind.robot_index is not None and ind.robot_index < len(trajectories):
                    vis_trajectories.append(trajectories[ind.robot_index])
                else:
                    vis_trajectories.append([np.array([0.0, 0.0])])
            
            save_mating_trajectories(
                trajectories=vis_trajectories,
                population=vis_population,
                fitness_values=vis_fitness,
                generation=self.generation + 1,
                population_size=pop_size,
                simulation_time=config.simulation_time,
                world_size=config.world_size,
                robot_size=config.robot_size,
                save_path=save_path,
                use_periodic_boundaries=config.use_periodic_boundaries,
                mating_zone_centers=self.current_zone_centers if config.pairing_method == "mating_zone" else None,
                mating_zone_radius=config.mating_zone_radius if config.pairing_method == "mating_zone" else None,
                pairing_method=config.pairing_method
            )
    
    def create_next_generation(self) -> None:
        """
        Create next generation through movement-based pairing and reproduction.
        """
        print(f"  Creating next generation with movement-based selection...")
        
        # Check if population is already extinct
        if len(self.population) == 0:
            print(f"  ⚠️  WARNING: Population is extinct! Cannot create next generation.")
            print(f"  💡 TIP: If using energy_based selection, check your energy config:")
            print(f"      - initial_energy: {config.initial_energy}")
            print(f"      - energy_depletion_rate: {config.energy_depletion_rate}")
            print(f"      - Generations until extinction: ~{int(config.initial_energy / config.energy_depletion_rate)}")
            return
        
        # Deplete energy for all individuals (time-based passive depletion)
        if config.enable_energy:
            for ind in self.population:
                ind.energy -= config.energy_depletion_rate
            
            if self.population:
                energy_values = [ind.energy for ind in self.population]
                print(f"  Energy after depletion: min={min(energy_values):.1f}, max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
        
        self.mating_movement_phase(duration=config.simulation_time, save_trajectories=True)
        
        new_population: list[SpatialIndividual] = []
        new_positions: list[np.ndarray] = []
        new_orientations: list[float] = []
        
        # Find pairs based on selected pairing method
        pairs, paired_indices, zones_with_matings = find_pairs(
            population=self.population,
            tracked_geoms=self.tracked_geoms,
            method=config.pairing_method,
            pairing_radius=config.pairing_radius,
            world_size=(config.world_size[0], config.world_size[1]),
            use_periodic_boundaries=config.use_periodic_boundaries,
            mating_zone_centers=self.current_zone_centers if self.current_zone_centers else [tuple(config.mating_zone_center)],
            mating_zone_radius=config.mating_zone_radius
        )
        
        print(f"  Created {len(pairs)} pairs from {self.population_size} robots")
        print(f"  Pairing method: {config.pairing_method}")
        print(f"  Unpaired robots: {self.population_size - len(paired_indices)}")
        
        # Event-driven zone relocation: relocate zones that had matings
        if config.zone_relocation_strategy == "event_driven" and zones_with_matings:
            self._relocate_zones(zones_with_matings)
        
        # Record mating statistics
        self.data_collector.record_mating_stats(
            num_pairs=len(pairs),
            num_unpaired=self.population_size - len(paired_indices),
            population_size=self.population_size
        )
        
        # Calculate offspring positions for all pairs
        pair_positions = calculate_offspring_positions(
            pairs=pairs,
            current_positions=self.current_positions,
            offspring_radius=config.offspring_radius,
            world_size=(config.world_size[0], config.world_size[1]),
            use_periodic_boundaries=config.use_periodic_boundaries
        )
        
        # Create offspring from pairs
        for pair_idx, (parent1_idx, parent2_idx) in enumerate(pairs):
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Apply mating energy effects
            if config.enable_energy:
                if config.mating_energy_effect == "restore":
                    parent1.energy = config.initial_energy
                    parent2.energy = config.initial_energy
                elif config.mating_energy_effect == "cost":
                    parent1.energy -= config.mating_energy_amount
                    parent2.energy -= config.mating_energy_amount
            
            # Crossover
            if np.random.random() < config.crossover_rate:
                child1, child2, self.next_unique_id = crossover_hyperneat(
                    parent1, parent2, self.next_unique_id, self.generation + 1
                )
            else:
                child1, self.next_unique_id = clone_individual(
                    parent1, self.next_unique_id, self.generation + 1
                )
                child2, self.next_unique_id = clone_individual(
                    parent2, self.next_unique_id, self.generation + 1
                )
            
            # Mutation
            child1, self.next_unique_id = mutate_hyperneat(
                child1,
                next_unique_id=self.next_unique_id,
                weight_mutation_rate=config.mutation_rate,
                weight_mutation_power=config.mutation_strength,
                add_connection_rate=config.mutation_add_connection_rate,
                add_node_rate=config.mutation_add_node_rate
            )
            
            child2, self.next_unique_id = mutate_hyperneat(
                child2,
                next_unique_id=self.next_unique_id,
                weight_mutation_rate=config.mutation_rate,
                weight_mutation_power=config.mutation_strength,
                add_connection_rate=config.mutation_add_connection_rate,
                add_node_rate=config.mutation_add_node_rate
            )
            
            new_population.append(child1)
            new_positions.append(pair_positions[pair_idx][0])
            new_orientations.append(np.random.uniform(0, 2 * np.pi))
            
            new_population.append(child2)
            new_positions.append(pair_positions[pair_idx][1])
            new_orientations.append(np.random.uniform(0, 2 * np.pi))
        
        # Extend population with offspring
        self.population.extend(new_population)
        self.current_positions.extend(new_positions)
        self.current_orientations.extend(new_orientations)
        
        # Assign zones to new offspring if using assigned_zone movement bias
        if config.movement_bias == "assigned_zone" and len(new_population) > 0:
            for individual in new_population:
                zone_idx = np.random.randint(0, len(self.current_zone_centers))
                self.assigned_zones[individual.unique_id] = zone_idx
        
        # Verify consistency
        if len(self.population) != len(self.current_positions):
            raise RuntimeError("Population and position arrays out of sync!")
        if len(self.population) != len(self.current_orientations):
            raise RuntimeError("Population and orientation arrays out of sync!")
        
        # Update population size
        old_size = self.population_size
        self.population_size = len(self.population)
        
        print(f"  Population: {old_size} → {self.population_size} (+{len(new_population)} offspring, {len(paired_indices)} paired)")
        
        # Record reproduction statistics
        self.data_collector.record_reproduction(
            num_offspring=len(new_population),
            population_before=old_size
        )
        
        # Report energy effects if enabled
        if config.enable_energy and self.population:
            energy_values = [ind.energy for ind in self.population]
            print(f"  Energy: min={min(energy_values):.1f}, max={max(energy_values):.1f}, avg={np.mean(energy_values):.1f}")
            if config.mating_energy_effect == "restore":
                print(f"  Mating effect: Energy restored to {config.initial_energy} for {len(pairs)} mating pairs")
            elif config.mating_energy_effect == "cost":
                print(f"  Mating effect: Energy cost of {config.mating_energy_amount} for {len(pairs)} mating pairs")
            
            # Record energy stats after mating
            self.data_collector.record_energy_stats(self.population, "after_mating")
        
    
    def run_evolution(self) -> SpatialIndividual | None:
        """
        Run the evolutionary algorithm.
            
        Returns:
            Best individual found (or None if population extinct)
        """
        print("=" * 60)
        print("SPATIAL EVOLUTIONARY ALGORITHM")
        print("=" * 60)
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.num_generations}")
        print(f"Robot joints: {self.num_joints}")
        print(f"Max population limit: {config.max_population_limit}")
        print(f"Min population limit: {config.min_population_limit}")
        if config.stop_on_limits:
            print(f"Early stopping: ENABLED (stops if limits reached)")
        if config.record_generation_videos:
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
            
            # Check population limits BEFORE processing generation
            if config.stop_on_limits:
                if self.population_size >= config.max_population_limit:
                    print(f"\n{'!'*60}")
                    print(f"POPULATION LIMIT REACHED!")
                    print(f"{'!'*60}")
                    print(f"Population size ({self.population_size}) reached or exceeded maximum limit ({config.max_population_limit})")
                    print(f"Stopping evolution at generation {gen}")
                    print(f"{'!'*60}")
                    
                    # Record the current generation so it appears in plots
                    # Use regular recording since population still exists (just too big)
                    self.data_collector.record_generation_start(gen, len(self.population))
                    
                    self.data_collector.record_early_stop(
                        gen, 
                        f"Population reached maximum limit ({config.max_population_limit})",
                        self.num_generations
                    )
                    break
                
                if self.population_size < config.min_population_limit:
                    print(f"\n{'!'*60}")
                    print(f"POPULATION EXTINCTION!")
                    print(f"{'!'*60}")
                    print(f"Population size ({self.population_size}) dropped below minimum limit ({config.min_population_limit})")
                    print(f"Stopping evolution at generation {gen}")
                    print(f"{'!'*60}")
                    
                    # Record the final generation with 0 population using special method
                    self.data_collector.record_extinct_generation(gen)
                    
                    self.data_collector.record_early_stop(
                        gen, 
                        f"Population extinction (below {config.min_population_limit})",
                        self.num_generations
                    )
                    break
            
            # Record generation start
            self.data_collector.record_generation_start(gen, len(self.population))
            
            # Update mating zones if dynamic
            self._update_mating_zones()
            
            # Spawn population in simulation
            self.spawn_population()
            
            # Evaluate fitness
            fitness_values = self.evaluate_population_fitness()
            
            # Record fitness statistics
            self.data_collector.record_fitness_stats(self.population, gen)
            
            # Record age statistics
            self.data_collector.record_age_stats(self.population, gen)
            
            # Record genotype diversity
            self.data_collector.record_genotype_diversity(self.population)
            
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
            
            # Apply selection AFTER fitness evaluation (not before!)
            # This ensures offspring from previous generation are evaluated before being selected
            if gen > 0:  # Skip selection in first generation (no offspring yet)
                print(f"\n  Applying selection...")
                print(f"  Pre-selection population: {len(self.population)}")
                
                # Show fitness statistics before selection
                if self.population:
                    fitness_values_pre_selection = [ind.fitness for ind in self.population]
                    print(f"  Pre-selection fitness range: {min(fitness_values_pre_selection):.4f} to {max(fitness_values_pre_selection):.4f}")
                
                # Record population size before selection
                population_before_selection = len(self.population)
                
                # Apply selection to manage population size
                self.population, self.current_positions, self.population_size, self.current_orientations = apply_selection(
                    population=self.population,
                    current_positions=self.current_positions,
                    method=config.selection_method,
                    target_size=config.target_population_size,
                    current_generation=self.generation,
                    current_orientations=self.current_orientations,
                    paired_indices=set(),  # No paired indices at this point
                    max_age=config.max_age,
                    # Density-based selection parameters
                    world_size=(config.world_size[0], config.world_size[1]),
                    use_periodic_boundaries=config.use_periodic_boundaries,
                    locality_radius=config.locality_radius,
                    critical_density=config.critical_density,
                    base_death_prob=config.base_death_prob,
                    max_density_death_prob=config.max_density_death_prob,
                    density_fitness_protection=config.density_fitness_protection
                )
                
                print(f"  Post-selection population: {len(self.population)}")
                
                # Record selection statistics
                self.data_collector.record_selection(
                    population_before=population_before_selection,
                    population_after=len(self.population)
                )
            
            # Create next generation (except for last generation)
            if gen < self.num_generations - 1:
                self.create_next_generation()
                
                # Check population limits AFTER creating next generation
                if config.stop_on_limits:
                    if self.population_size >= config.max_population_limit:
                        print(f"POPULATION LIMIT REACHED AFTER REPRODUCTION!")
                        print(f"Population size ({self.population_size}) reached or exceeded maximum limit ({config.max_population_limit})")
                        print(f"Stopping evolution after generation {gen + 1}")
                        
                        # Record the final generation with stats
                        self.data_collector.record_generation_start(gen + 1, len(self.population))
                        
                        # Record fitness and age stats for the final generation
                        self.data_collector.record_fitness_stats(self.population, gen + 1)
                        self.data_collector.record_age_stats(self.population, gen + 1)
                        self.data_collector.record_genotype_diversity(self.population)
                        
                        self.data_collector.record_early_stop(
                            gen + 1, 
                            f"Population reached maximum limit after reproduction ({config.max_population_limit})",
                            self.num_generations
                        )
                        # Run final mating movement to capture positions
                        self.mating_movement_phase(
                            duration=config.simulation_time, 
                            save_trajectories=True
                        )
                        # Clean up resources before breaking
                        self.cleanup_simulation_resources()
                        break
                    
                    if self.population_size < config.min_population_limit:
                        print(f"Population size ({self.population_size}) dropped below minimum limit ({config.min_population_limit})")
                        
                        # Record the final generation with 0 population using special method
                        self.data_collector.record_extinct_generation(gen + 1)
                        
                        self.data_collector.record_early_stop(
                            gen + 1, 
                            f"Population extinction after selection (below {config.min_population_limit})",
                            self.num_generations
                        )
                        # Clean up resources before breaking
                        self.cleanup_simulation_resources()
                        break
            else:
                # Run mating movement to capture final positions
                self.mating_movement_phase(
                    duration=config.simulation_time, 
                    save_trajectories=True
                )
            
            # Clean up simulation resources after each generation to manage memory
            self.cleanup_simulation_resources()
        
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        # Save collected data
        print(f"\nSaving evolution data...")
        summary = self.data_collector.get_summary_stats()
        print(f"\nEvolution Summary:")
        print(f"  Total generations: {summary['total_generations']}")
        print(f"  Population: {summary['population']['initial']} → {summary['population']['final']}")
        print(f"  Best fitness ever: {summary['fitness']['best_ever']:.4f}")
        print(f"  Total births: {summary['total_births']}")
        print(f"  Total deaths: {summary['total_deaths']}")
        print(f"  Avg mating success: {summary['avg_mating_success_rate']:.1f}%")
        
        self.data_collector.save_to_csv(config.results_folder)
        self.data_collector.save_to_npz(config.results_folder)
        self.data_collector.plot_evolution_statistics(config.figures_folder)

        return self.get_best_individual()
    
    def _genotype_to_dict(self, genotype: dict) -> dict:
        """Convert HyperNEAT genotype to JSON-serializable dictionary."""
        if not isinstance(genotype, dict):
            return genotype  # Not a HyperNEAT genotype, return as-is
        
        serializable = {}
        
        # Convert nodes
        if 'nodes' in genotype:
            serializable['nodes'] = [
                {
                    'id': node.id,
                    'type': node.type,
                    'activation': node.activation,
                    'layer': node.layer
                }
                for node in genotype['nodes']
            ]
        
        # Convert connections
        if 'connections' in genotype:
            serializable['connections'] = [
                {
                    'in_node': conn.in_node,
                    'out_node': conn.out_node,
                    'weight': conn.weight,
                    'enabled': conn.enabled,
                    'innovation': conn.innovation
                }
                for conn in genotype['connections']
            ]
        
        # Copy other fields
        for key in ['input_size', 'output_size', 'next_node_id', 'next_innovation']:
            if key in genotype:
                serializable[key] = genotype[key]
        
        return serializable
    
    def save_final_controllers(self) -> None:
        """
        Save the final population's controllers (genotypes) to files.
        
        Saves in multiple formats:
        - JSON: Human-readable with individual metadata
        - NPZ: NumPy format for fast loading (HyperNEAT genotypes saved as pickled objects)
        - TXT: Simple readable format for best individual
        """
        from datetime import datetime
        from pathlib import Path
        import json
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_folder = Path(config.results_folder)
        results_folder.mkdir(parents=True, exist_ok=True)
        
        # 1. Save all controllers as JSON with metadata
        json_path = results_folder / f"final_controllers_{timestamp}.json"
        controllers_data = {
            'timestamp': timestamp,
            'generation': self.generation,
            'num_joints': self.num_joints,
            'population_size': len(self.population),
            'config': {
                'selection_method': config.selection_method,
                'enable_energy': config.enable_energy,
                'mating_energy_effect': config.mating_energy_effect if config.enable_energy else None,
            },
            'controllers': []
        }
        
        for ind in self.population:
            controller = {
                'unique_id': ind.unique_id,
                'generation_born': ind.generation,
                'age': self.generation - ind.generation,
                'fitness': ind.fitness,
                'energy': ind.energy if config.enable_energy else None,
                'genotype': self._genotype_to_dict(ind.genotype),
                'parent_ids': ind.parent_ids,
                'position': ind.spawn_position.tolist() if ind.spawn_position is not None else None,
            }
            controllers_data['controllers'].append(controller)
        
        # Sort by fitness (best first)
        controllers_data['controllers'].sort(key=lambda x: x['fitness'], reverse=True)
        
        with open(json_path, 'w') as f:
            json.dump(controllers_data, f, indent=2)
        print(f"  Final controllers saved to JSON: {json_path}")
        
        # 2. Save all genotypes as NPZ (using pickle for complex HyperNEAT objects)
        npz_path = results_folder / f"final_genotypes_{timestamp}.npz"
        
        # For HyperNEAT genotypes, we need to use object arrays with pickle
        genotype_array = np.array([ind.genotype for ind in self.population], dtype=object)
        fitness_array = np.array([ind.fitness for ind in self.population])
        energy_array = np.array([ind.energy for ind in self.population]) if config.enable_energy else None
        age_array = np.array([self.generation - ind.generation for ind in self.population])
        id_array = np.array([ind.unique_id for ind in self.population])
        
        npz_data = {
            'genotypes': genotype_array,
            'fitness': fitness_array,
            'ages': age_array,
            'ids': id_array,
            'num_joints': self.num_joints,
            'generation': self.generation,
        }
        if energy_array is not None:
            npz_data['energy'] = energy_array
        
        np.savez(npz_path, **npz_data)
        print(f"  Final genotypes saved to NPZ: {npz_path}")
        
        # 3. Save best controller in human-readable format
        best = self.get_best_individual()
        if not best:
            print(f"  No best individual to save (population extinct)")
            return
        
        txt_path = results_folder / f"best_controller_{timestamp}.txt"
        
        with open(txt_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("BEST CONTROLLER\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Individual ID: {best.unique_id}\n")
            f.write(f"Born in Generation: {best.generation}\n")
            f.write(f"Age: {self.generation - best.generation} generations\n")
            f.write(f"Fitness: {best.fitness:.6f}\n")
            if config.enable_energy:
                f.write(f"Energy: {best.energy:.2f}\n")
            f.write(f"Parent IDs: {best.parent_ids}\n")
            
            # Write HyperNEAT network structure
            if isinstance(best.genotype, dict):
                f.write(f"\nHyperNEAT Network Structure:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Nodes: {len(best.genotype.get('nodes', []))}\n")
                f.write(f"Connections: {len(best.genotype.get('connections', []))}\n")
                f.write(f"Input size: {best.genotype.get('input_size', 'N/A')}\n")
                f.write(f"Output size: {best.genotype.get('output_size', 'N/A')}\n")
                
                f.write(f"\nNetwork Nodes:\n")
                for node in best.genotype.get('nodes', []):
                    f.write(f"  Node {node.id}: type={node.type}, activation={node.activation}, layer={node.layer}\n")
                
                f.write(f"\nNetwork Connections:\n")
                for conn in best.genotype.get('connections', []):
                    status = "enabled" if conn.enabled else "disabled"
                    f.write(f"  {conn.in_node} → {conn.out_node}: weight={conn.weight:.6f} ({status})\n")
            else:
                # Fallback for other genotype formats
                f.write(f"\nGenotype:\n")
                f.write("-" * 60 + "\n")
                f.write(f"{best.genotype}\n")
            
            f.write("\n" + "-" * 60 + "\n")
            f.write("Raw genotype array:\n")
            f.write(str(best.genotype) + "\n")
        
        print(f"  Best controller saved to TXT: {txt_path}")
        
        # Summary
        print(f"\n  Saved {len(self.population)} controllers from final population")
        print(f"  Best fitness: {best.fitness:.6f} (ID: {best.unique_id})")
    
    @staticmethod
    def load_controllers_from_json(json_path: str) -> dict:
        """
        Load controllers from a saved JSON file.
        
        Args:
            json_path: Path to the JSON file
            
        Returns:
            Dictionary containing all controller data
        """
        import json
        with open(json_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def load_genotypes_from_npz(npz_path: str) -> dict:
        """
        Load genotypes from a saved NPZ file.
        
        Args:
            npz_path: Path to the NPZ file
            
        Returns:
            Dictionary with arrays: genotypes, fitness, ages, ids
        """
        data = np.load(npz_path)
        return {
            'genotypes': data['genotypes'],
            'fitness': data['fitness'],
            'ages': data['ages'],
            'ids': data['ids'],
            'num_joints': int(data['num_joints']),
            'generation': int(data['generation']),
            'energy': data['energy'] if 'energy' in data else None,
        }
    
    def get_best_individual(self) -> SpatialIndividual | None:
        """Get the best individual from current population."""
        if not self.population:
            return None
        return max(self.population, key=lambda ind: ind.fitness)

def main():
    """Main entry point for the spatial EA."""
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
    
    # Run evolution
    spatial_ea.run_evolution()
    
    # Plot results
    viz = ExperimentVisualizer('path/to/results.csv')
    viz.generate_report()
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__": 
    main()
