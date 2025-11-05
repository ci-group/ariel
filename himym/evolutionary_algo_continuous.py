#!/usr/bin/env python3
import traceback
import argparse
import numpy as np
import mujoco
from pathlib import Path
import time
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

from ea_config import config


class ContinuousIndividual:
    """Individual in continuous evolution (no generation tracking)"""
    
    def __init__(self, genotype: np.ndarray, robot_index: int):
        self.genotype = genotype
        self.robot_index = robot_index
        self.fitness: float = 0.0
        self.fitness_timestamp: float = 0.0  # When fitness was last evaluated
        self.spawn_position: Optional[np.ndarray] = None
        self.birth_time: float = 0.0  # When this genotype was created
        self.parent_ids: Tuple[int, int] = (-1, -1)  # Track lineage
        self.num_matings: int = 0
        self.total_displacement: float = 0.0  # Running total for online fitness


class ContinuousContactEA:
    """
    Continuous evolution with contact-based mating.
    
    Evolution occurs in real-time:
    1. All robots move simultaneously in the arena
        Works
    2. When two robots collide, they may mate (if cooldown allows)
        Needs testing
    3. Mating produces 2 offspring via crossover + mutation
        Needs testing, but should work
    4. Each parent has a chance to be replaced: P = 1 - (fitness/max_fitness)
        Needs testing, but should work
    5. Replaced robots adopt offspring genotype immediately
        Needs testing
    6. Process continues for specified duration
        Works
    """
    
    def __init__(
        self,
        population_size: int = 25,
        num_joints: int = 8,
        mating_cooldown: float = 30.0,
        fitness_update_interval: float = 30.0,
        checkpoint_interval: float = 300.0,
        elite_archive_size: int = 10
    ):
        self.population_size = population_size
        self.num_joints = num_joints
        self.mating_cooldown_duration = mating_cooldown
        self.fitness_update_interval = fitness_update_interval
        self.checkpoint_interval = checkpoint_interval
        self.elite_archive_size = elite_archive_size
        
        # Population
        self.population: List[ContinuousIndividual] = []
        
        # MuJoCo simulation
        self.world: Optional[SimpleFlatWorld] = None
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        
        # Tracking
        self.geom_to_robot: Dict[int, int] = {}  # geom_id -> robot_index
        self.tracked_geoms: List = []  # Geom objects for position tracking
        self.mating_cooldown: Dict[int, float] = {}  # robot_id -> time when can mate again
        self.last_fitness_update: float = 0.0
        self.last_checkpoint: float = 0.0
        self.checkpoint_count: int = 0
        
        # Statistics
        self.mating_events: List[Dict] = []  # Log of all matings
        self.elite_archive: List[Tuple[np.ndarray, float]] = []  # (genotype, fitness)
        self.checkpoints: List[Dict] = []  # Population snapshots
        
        # Position tracking for fitness
        self.last_positions: Dict[int, np.ndarray] = {}
        
        # Trajectory visualization
        self.trajectories: Dict[int, List[Tuple[float, float, float]]] = {}  # robot_id -> [(x,y,time)]
        self.mating_locations: List[Tuple[float, float, float, int, int]] = []  # [(x,y,time,r1,r2)]
        self.trajectory_sample_interval: float = 10.0  # Sample every 10 seconds
        self.last_trajectory_sample: float = 0.0
        
    def initialize_population(self) -> List[ContinuousIndividual]:
        population = []
        
        for i in range(self.population_size):
            # Random genotype: [amplitude, frequency, phase] for each joint
            genotype = np.zeros(self.num_joints * 3)
            
            for j in range(self.num_joints):
                genotype[j * 3] = np.random.uniform(
                    config.amplitude_min, 
                    config.amplitude_max
                )
                genotype[j * 3 + 1] = np.random.uniform(
                    config.frequency_min, 
                    config.frequency_max
                )
                genotype[j * 3 + 2] = np.random.uniform(
                    0, 
                    config.phase_max
                )
            
            individual = ContinuousIndividual(genotype, robot_index=i)
            individual.birth_time = 0.0
            population.append(individual)
        
        return population
    
    def initialize_from_genotypes(self, genotypes: List[np.ndarray]) -> List[ContinuousIndividual]:
        if len(genotypes) != self.population_size:
            print(f"Warning: Provided {len(genotypes)} genotypes but population_size={self.population_size}")
            print(f"  Adjusting population_size to {len(genotypes)}")
            self.population_size = len(genotypes)
        
        population = []
        for i, genotype in enumerate(genotypes):
            individual = ContinuousIndividual(genotype.copy(), robot_index=i)
            individual.birth_time = 0.0
            population.append(individual)
        
        print(f"Initialized population from {len(genotypes)} provided genotypes")
        return population
    
    def spawn_population(self):
        print(f"Spawning population of {self.population_size} robots...")
        
        # Clear any existing MuJoCo control callback before creating world
        mujoco.set_mjcb_control(None)
        
        # Create world (use list, not tuple - MuJoCo requires list for size parameter)
        self.world = SimpleFlatWorld(config.world_size)
        
        # Generate non-overlapping spawn positions
        positions = []
        max_attempts = 1000
        min_distance = config.min_spawn_distance
        
        # Randomly place robots with collision avoidance
        for i in range(self.population_size):
            attempts = 0
            while attempts < max_attempts:
                # Random position in world
                x = np.random.uniform(0.5, config.world_size[0] - 0.5)
                y = np.random.uniform(0.5, config.world_size[1] - 0.5)
                pos = np.array([x, y, 0.2])
                
                # Check distance to existing positions
                if len(positions) == 0:
                    positions.append(pos)
                    break
                
                distances = [np.linalg.norm(pos[:2] - p[:2]) for p in positions]
                if min(distances) >= min_distance:
                    positions.append(pos)
                    break
                
                attempts += 1
            
            if attempts >= max_attempts:
                # Fallback to grid if random placement fails
                print(f"  Warning: Using grid fallback for robot {i}")
                grid_size = int(np.ceil(np.sqrt(self.population_size)))
                spacing = min(config.world_size[0], config.world_size[1]) / (grid_size + 1)
                grid_x = (i % grid_size) * spacing + spacing/2
                grid_y = (i // grid_size) * spacing + spacing/2
                pos = np.array([grid_x, grid_y, 0.2])
                positions.append(pos)
        
        # Spawn robots
        print(f"  Spawning {self.population_size} robots...")
        for i, individual in enumerate(self.population):
            robot = gecko()
            pos = positions[i]
            individual.spawn_position = pos.copy()
            # Don't correct for bounding box when spawning multiple robots - causes compilation errors
            self.world.spawn(robot.spec, spawn_position=pos, prefix_id=i, correct_for_bounding_box=False)
            if (i + 1) % 5 == 0 or i == self.population_size - 1:
                print(f"    Spawned {i + 1}/{self.population_size} robots")
        
        # Compile world
        print(f"  Compiling world with {self.population_size} robots...")
        try:
            self.model = self.world.spec.compile()
            print(f"  World compiled successfully!")
        except Exception as e:
            print(f"  ERROR during compilation: {e}")
            traceback.print_exc()
            raise
        self.data = mujoco.MjData(self.model)
        
        mujoco.mj_forward(self.model, self.data)  # Forward simulate to initialize positions
        
        # Track all robot core geoms
        all_geoms = self.world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        self.tracked_geoms = []
        for i in range(self.population_size):
            # Match exactly "robot-{i}core" - the main core of robot i
            core_name = f"robot-{i}core"
            for geom in all_geoms:
                if geom.name == core_name:
                    bound_geom = self.data.bind(geom)
                    self.tracked_geoms.append(bound_geom)
                    # Map geom id to robot index using the id field
                    geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, core_name)
                    self.geom_to_robot[geom_id] = i
                    break
        
        # Initialize position tracking
        for i in range(self.population_size):
            self.last_positions[i] = self.tracked_geoms[i].xpos.copy()
        
        print(f"  Successfully spawned {self.population_size} robots")
        print(f"  Tracking {len(self.tracked_geoms)} core geoms")
    
    def evaluate_fitness(self, robot_index: int) -> float:
        current_pos = self.tracked_geoms[robot_index].xpos.copy()
        last_pos = self.last_positions.get(robot_index, current_pos)
        
        displacement = np.linalg.norm(current_pos[:2] - last_pos[:2])
        
        # Update tracking
        self.last_positions[robot_index] = current_pos.copy()
        self.population[robot_index].total_displacement += displacement
        
        # Fitness = total displacement (simple metric)
        fitness = self.population[robot_index].total_displacement
        
        return fitness
    
    def update_all_fitness(self):
        for i in range(self.population_size):
            fitness = self.evaluate_fitness(i)
            self.population[i].fitness = fitness
            self.population[i].fitness_timestamp = self.data.time
    
    def crossover(
        self, 
        parent1: ContinuousIndividual, 
        parent2: ContinuousIndividual
    ) -> Tuple[np.ndarray, np.ndarray]:
        crossover_point = np.random.randint(1, len(parent1.genotype))
        
        child1_genotype = np.concatenate([
            parent1.genotype[:crossover_point],
            parent2.genotype[crossover_point:]
        ])
        
        child2_genotype = np.concatenate([
            parent2.genotype[:crossover_point],
            parent1.genotype[crossover_point:]
        ])
        
        return child1_genotype, child2_genotype
    
    def mutate(self, genotype: np.ndarray) -> np.ndarray:
        mutated = genotype.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < config.mutation_rate:
                # Add Gaussian noise
                mutated[i] += np.random.normal(0, config.mutation_strength)
                
                # Clamp values
                param_type = i % 3
                if param_type == 0:  # amplitude
                    mutated[i] = np.clip(
                        mutated[i], 
                        config.amplitude_min, 
                        config.amplitude_max
                    )
                elif param_type == 1:  # frequency
                    mutated[i] = np.clip(
                        mutated[i], 
                        config.frequency_min, 
                        config.frequency_max
                    )
                else:  # phase
                    mutated[i] = mutated[i] % config.phase_max
        
        return mutated
    
    def handle_contact(self, robot1_id: int, robot2_id: int):
        current_time = self.data.time
        
        # Check cooldown
        if current_time < self.mating_cooldown.get(robot1_id, 0):
            return
        if current_time < self.mating_cooldown.get(robot2_id, 0):
            return
        
        parent1 = self.population[robot1_id]
        parent2 = self.population[robot2_id]
        
        # Perform crossover
        child1_genotype, child2_genotype = self.crossover(parent1, parent2)
        
        # Mutate offspring
        child1_genotype = self.mutate(child1_genotype)
        child2_genotype = self.mutate(child2_genotype)
        
        # Fitness-weighted replacement decision
        max_fitness = max([ind.fitness for ind in self.population])
        if max_fitness == 0:
            max_fitness = 1.0  # Avoid division by zero
        
        # Parent 1: probability of being replaced = 1 - (fitness / max_fitness)
        replace_prob_1 = 1.0 - (parent1.fitness / max_fitness)
        if np.random.random() < replace_prob_1:
            # Replace parent 1 with child 1
            parent1.genotype = child1_genotype
            parent1.birth_time = current_time
            parent1.parent_ids = (robot1_id, robot2_id)
            parent1.fitness = 0.0  # Reset fitness
            parent1.fitness_timestamp = current_time
            parent1.total_displacement = 0.0
            self.last_positions[robot1_id] = self.tracked_geoms[robot1_id].xpos.copy()
            replaced_1 = True
        else:
            replaced_1 = False
        
        # Parent 2: probability of being replaced = 1 - (fitness / max_fitness)
        replace_prob_2 = 1.0 - (parent2.fitness / max_fitness)
        if np.random.random() < replace_prob_2:
            # Replace parent 2 with child 2
            parent2.genotype = child2_genotype
            parent2.birth_time = current_time
            parent2.parent_ids = (robot1_id, robot2_id)
            parent2.fitness = 0.0  # Reset fitness
            parent2.fitness_timestamp = current_time
            parent2.total_displacement = 0.0
            self.last_positions[robot2_id] = self.tracked_geoms[robot2_id].xpos.copy()
            replaced_2 = True
        else:
            replaced_2 = False
        
        # Update mating counts
        parent1.num_matings += 1
        parent2.num_matings += 1
        
        # Set cooldown
        self.mating_cooldown[robot1_id] = current_time + self.mating_cooldown_duration
        self.mating_cooldown[robot2_id] = current_time + self.mating_cooldown_duration
        
        # Record mating location (midpoint between robots)
        pos1 = self.tracked_geoms[robot1_id].xpos.copy()
        pos2 = self.tracked_geoms[robot2_id].xpos.copy()
        mating_x = (pos1[0] + pos2[0]) / 2
        mating_y = (pos1[1] + pos2[1]) / 2
        self.mating_locations.append((mating_x, mating_y, current_time, robot1_id, robot2_id))
        
        # Log mating event
        event = {
            'time': current_time,
            'robot1': robot1_id,
            'robot2': robot2_id,
            'fitness1': parent1.fitness,
            'fitness2': parent2.fitness,
            'replaced1': replaced_1,
            'replaced2': replaced_2,
            'replace_prob1': replace_prob_1,
            'replace_prob2': replace_prob_2
        }
        self.mating_events.append(event)
        
        # Print mating notification
        status1 = "REPLACED" if replaced_1 else "survived"
        status2 = "REPLACED" if replaced_2 else "survived"
        print(f"  [t={current_time:.1f}s] Mating: R{robot1_id} ({status1}) × R{robot2_id} ({status2})")
    
    def check_contacts(self):
        """Check for contacts between robots and trigger mating"""
        # Iterate through all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2
            
            # Check if both geoms belong to robots
            if geom1 in self.geom_to_robot and geom2 in self.geom_to_robot:
                robot1_id = self.geom_to_robot[geom1]
                robot2_id = self.geom_to_robot[geom2]
                
                # Handle contact (will check cooldown internally)
                self.handle_contact(robot1_id, robot2_id)
    
    def update_elite_archive(self):
        for individual in self.population:
            if individual.fitness > 0:
                self.elite_archive.append((individual.genotype.copy(), individual.fitness))
        
        # Sort by fitness and keep top N
        self.elite_archive.sort(key=lambda x: x[1], reverse=True)
        self.elite_archive = self.elite_archive[:self.elite_archive_size]
    
    def create_checkpoint(self):
        fitness_values = [ind.fitness for ind in self.population]
        
        checkpoint = {
            'time': self.data.time,
            'checkpoint_num': self.checkpoint_count,
            'population_size': self.population_size,
            'fitness_mean': float(np.mean(fitness_values)),
            'fitness_std': float(np.std(fitness_values)),
            'fitness_max': float(np.max(fitness_values)),
            'fitness_min': float(np.min(fitness_values)),
            'total_matings': len(self.mating_events),
            'elite_archive_best': float(self.elite_archive[0][1]) if self.elite_archive else 0.0
        }
        
        self.checkpoints.append(checkpoint)
        self.checkpoint_count += 1
        
        print(f"\n{'='*60}")
        print(f"CHECKPOINT {self.checkpoint_count} @ t={self.data.time:.1f}s")
        print(f"{'='*60}")
        print(f"Fitness: mean={checkpoint['fitness_mean']:.3f}, "
              f"max={checkpoint['fitness_max']:.3f}, "
              f"std={checkpoint['fitness_std']:.3f}")
        print(f"Total matings: {checkpoint['total_matings']}")
        print(f"Best ever: {checkpoint['elite_archive_best']:.3f}")
        print(f"{'='*60}\n")
    
    def sample_trajectories(self):
        """Sample current positions of all robots for trajectory visualization."""
        current_time = self.data.time
        
        for i in range(self.population_size):
            pos = self.tracked_geoms[i].xpos.copy()
            if i not in self.trajectories:
                self.trajectories[i] = []
            self.trajectories[i].append((pos[0], pos[1], current_time))
    
    def plot_trajectories(self, save_path: Optional[str] = None):
        print(f"\nPlotting trajectories:")
        print(f"  Total robots tracked: {len(self.trajectories)}")
        if self.trajectories:
            sample_points = [len(traj) for traj in self.trajectories.values()]
            print(f"  Trajectory points per robot: min={min(sample_points)}, max={max(sample_points)}, avg={sum(sample_points)/len(sample_points):.1f}")
        print(f"  Total mating events: {len(self.mating_events)}")
        print(f"  Total mating locations: {len(self.mating_locations)}")
        
        fig, ax = plt.subplots(figsize=(12, 12))
        
        colors = plt.cm.tab20(np.linspace(0, 1, self.population_size))
        
        # Plot trajectories - plot even single points
        trajectories_plotted = 0
        for robot_id, trajectory in self.trajectories.items():
            if len(trajectory) > 0:
                xs = [point[0] for point in trajectory]
                ys = [point[1] for point in trajectory]
                
                # Plot path if more than one point
                if len(trajectory) > 1:
                    ax.plot(xs, ys, '-', color=colors[robot_id], 
                           alpha=0.6, linewidth=1.5, label=f'Robot {robot_id}')
                    trajectories_plotted += 1
                
                # Mark start position
                ax.plot(xs[0], ys[0], 'o', color=colors[robot_id], 
                       markersize=8, markeredgecolor='black', markeredgewidth=1)
                
                # Mark end position
                if len(trajectory) > 1:
                    ax.plot(xs[-1], ys[-1], 's', color=colors[robot_id], 
                           markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        print(f"  Trajectories plotted: {trajectories_plotted}")
        
        # Plot mating locations
        if self.mating_locations:
            mating_xs = [loc[0] for loc in self.mating_locations]
            mating_ys = [loc[1] for loc in self.mating_locations]
            
            # Plot mating points with red X markers
            ax.scatter(mating_xs, mating_ys, marker='x', s=200, 
                      c='red', linewidths=3, label='Mating Events', zorder=10)
            
            # Add circles around mating points
            for x, y in zip(mating_xs, mating_ys):
                circle = patches.Circle((x, y), 0.15, fill=False, 
                                       edgecolor='red', linewidth=2, alpha=0.7)
                ax.add_patch(circle)
        
        # Configure plot
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title(f'Robot Trajectories and Mating Locations\n'
                    f'{len(self.mating_events)} mating events over simulation',
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Set arena bounds
        world_size = config.world_size
        ax.set_xlim(0, world_size[0])
        ax.set_ylim(0, world_size[1])
        
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 11:  # 10 robots + mating events
            ax.legend(handles[:10] + [handles[-1]], labels[:10] + [labels[-1]], 
                     loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trajectory plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def controller(self, model, data):
        """CPG controller for all robots"""
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
    
    # def run_evolution(self, duration: float = 1800.0):
    #     print(f"\n{'='*60}")
    #     print(f"CONTINUOUS CONTACT-BASED EVOLUTION")
    #     print(f"{'='*60}")
    #     print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
    #     print(f"Population size: {self.population_size}")
    #     print(f"Mating cooldown: {self.mating_cooldown_duration}s")
    #     print(f"Fitness update interval: {self.fitness_update_interval}s")
    #     print(f"Checkpoint interval: {self.checkpoint_interval}s")
    #     print(f"{'='*60}\n")
        
    #     # Initialize population
    #     self.population = self.initialize_population()
        
    #     # Spawn robots
    #     self.spawn_population()
        
    #     # Initial fitness evaluation
    #     self.update_all_fitness()
    #     self.update_elite_archive()
        
    #     # Set controller
    #     mujoco.set_mjcb_control(lambda m, d: self.controller(m, d))
        
    #     # Initial checkpoint
    #     self.create_checkpoint()
        
    #     # Main evolution loop
    #     print("Starting continuous evolution...\n")
    #     start_time = time.time()
        
    #     sim_steps = int(duration / self.model.opt.timestep)
    #     contact_check_interval = int(0.1 / self.model.opt.timestep)  # Check contacts every 0.1s
    #     trajectory_sample_steps = int(self.trajectory_sample_interval / self.model.opt.timestep)
        
    #     for step in range(sim_steps):
    #         mujoco.mj_step(self.model, self.data)
            
    #         # Check for contacts periodically
    #         if step % contact_check_interval == 0:
    #             self.check_contacts()
            
    #         # Sample trajectories periodically
    #         if step % trajectory_sample_steps == 0:
    #             self.sample_trajectories()
            
    #         # Periodic fitness updates
    #         if self.data.time - self.last_fitness_update >= self.fitness_update_interval:
    #             self.update_all_fitness()
    #             self.update_elite_archive()
    #             self.last_fitness_update = self.data.time
            
    #         # Periodic checkpoints
    #         if self.data.time - self.last_checkpoint >= self.checkpoint_interval:
    #             self.create_checkpoint()
    #             self.last_checkpoint = self.data.time
        
    #     # Final checkpoint
    #     self.update_all_fitness()
    #     self.update_elite_archive()
    #     self.create_checkpoint()
        
    #     elapsed_time = time.time() - start_time
        
    #     # Final statistics
    #     print(f"\n{'='*60}")
    #     print(f"EVOLUTION COMPLETE")
    #     print(f"{'='*60}")
    #     print(f"Simulated time: {duration}s ({duration/60:.1f} minutes)")
    #     print(f"Real time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} minutes)")
    #     print(f"Speedup: {duration/elapsed_time:.2f}x")
    #     print(f"Total matings: {len(self.mating_events)}")
    #     print(f"Matings per minute: {len(self.mating_events)/(duration/60):.1f}")
    #     print(f"Best fitness: {self.elite_archive[0][1]:.3f}" if self.elite_archive else "N/A")
    #     print(f"{'='*60}\n")
        
    #     # Save results
    #     self.save_results()
    
    def save_results(self):
        """Save evolution results to files"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Save checkpoints
        checkpoint_file = f"{config.results_folder}/continuous_checkpoints_{timestamp}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(self.checkpoints, f, indent=2)
        print(f"Saved checkpoints to {checkpoint_file}")
        
        # Save mating events
        mating_file = f"{config.results_folder}/continuous_matings_{timestamp}.json"
        with open(mating_file, 'w') as f:
            json.dump(self.mating_events, f, indent=2)
        print(f"Saved mating events to {mating_file}")
        
        # Save elite archive
        elite_file = f"{config.results_folder}/continuous_elite_{timestamp}.npz"
        elite_genotypes = [g for g, f in self.elite_archive]
        elite_fitness = [f for g, f in self.elite_archive]
        np.savez(
            elite_file,
            genotypes=np.array(elite_genotypes),
            fitness=np.array(elite_fitness)
        )
        print(f"Saved elite archive to {elite_file}")
        
        # Save final population
        pop_file = f"{config.results_folder}/continuous_population_{timestamp}.npz"
        genotypes = [ind.genotype for ind in self.population]
        fitness = [ind.fitness for ind in self.population]
        np.savez(
            pop_file,
            genotypes=np.array(genotypes),
            fitness=np.array(fitness)
        )
        print(f"Saved final population to {pop_file}")
        
        # Plot and save trajectories
        trajectory_plot_file = f"{config.results_folder}/continuous_trajectories_{timestamp}.png"
        self.plot_trajectories(save_path=trajectory_plot_file)


# def main():
#     parser = argparse.ArgumentParser(
#         description="Continuous Contact-Based Spatial EA"
#     )
#     parser.add_argument(
#         '-d', '--duration',
#         type=float,
#         default=1800.0,
#         help='Simulation duration in seconds (default: 1800 = 30 min)'
#     )
#     parser.add_argument(
#         '-p', '--population',
#         type=int,
#         default=25,
#         help='Population size (default: 25)'
#     )
#     parser.add_argument(
#         '-c', '--cooldown',
#         type=float,
#         default=30.0,
#         help='Mating cooldown in seconds (default: 30)'
#     )
#     parser.add_argument(
#         '-f', '--fitness-interval',
#         type=float,
#         default=30.0,
#         help='Fitness update interval in seconds (default: 30)'
#     )
#     parser.add_argument(
#         '-k', '--checkpoint-interval',
#         type=float,
#         default=300.0,
#         help='Checkpoint interval in seconds (default: 300 = 5 min)'
#     )
    
#     args = parser.parse_args()
    
#     # Create EA
#     ea = ContinuousContactEA(
#         population_size=args.population,
#         num_joints=8,
#         mating_cooldown=args.cooldown,
#         fitness_update_interval=args.fitness_interval,
#         checkpoint_interval=args.checkpoint_interval,
#         elite_archive_size=10
#     )
    
#     # Run evolution
#     ea.run_evolution(duration=args.duration)


# if __name__ == "__main__":
#     main()
