# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mujoco import viewer

import concurrent.futures

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.ec.a001 import Individual
from ariel.ec.a005 import Crossover
from ariel.ec.a000 import IntegerMutator

# Keep track of data / history
HISTORY = []

# Global variables for EA
CURRENT_INDIVIDUAL = None
POPULATION = []
GENERATION = 0
POPULATION_SIZE = 25
NUM_GENERATIONS = 400
SIMULATION_TIME = 50.0  # seconds per evaluation


def create_individual(num_joints):
    """Create an individual with sinusoidal movement parameters."""
    individual = Individual()

    genotype = []
    for _ in range(num_joints):
        amplitude = np.random.uniform(0.1, np.pi/3)  # Joint amplitude
        frequency = np.random.uniform(0.5, 3.0)     # Oscillation frequency
        phase = np.random.uniform(0, 2*np.pi)       # Phase offset
        genotype.extend([amplitude, frequency, phase])
    
    individual.genotype = genotype
    return individual


def sinusoidal_controller(model, data, to_track):
    """Movement controller based on current individual's genotype."""
    global CURRENT_INDIVIDUAL
    
    if CURRENT_INDIVIDUAL is None:
        return
    
    num_joints = model.nu
    genotype = CURRENT_INDIVIDUAL.genotype
    
    # Apply sinusoidal control based on genotype
    for i in range(num_joints):
        if i * 3 + 2 < len(genotype):  # Ensure we have all 3 parameters
            amplitude = genotype[i * 3]
            frequency = genotype[i * 3 + 1] 
            phase = genotype[i * 3 + 2]
            
            # Generate sinusoidal control signal
            control_value = amplitude * np.sin(frequency * data.time + phase)
            data.ctrl[i] = np.clip(control_value, -np.pi/4, np.pi/4)
    
    # Save movement to history
    HISTORY.append(to_track[0].xpos.copy())


def evaluate_fitness(individual, start_pos, end_pos):
    """Calculate fitness based on distance traveled."""
    distance = np.linalg.norm(np.array(end_pos) - np.array(start_pos))
    individual.fitness = distance
    return distance


def evaluate_individual(individual, model, world):
    """Evaluate a single individual - designed for parallel execution."""
    # Reset simulation
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    
    # Create local history for this individual
    local_history = []
    
    def local_sinusoidal_controller(model, data, to_track):
        num_joints = model.nu
        genotype = individual.genotype
        
        # Apply sinusoidal control based on genotype
        for i in range(num_joints):
            if i * 3 + 2 < len(genotype):  # Ensure we have all 3 parameters
                amplitude = genotype[i * 3]
                frequency = genotype[i * 3 + 1] 
                phase = genotype[i * 3 + 2]
                
                # Generate sinusoidal control signal
                control_value = amplitude * np.sin(frequency * data.time + phase)
                data.ctrl[i] = np.clip(control_value, -np.pi/2, np.pi/2)
        
        # Save movement to local history
        local_history.append(to_track[0].xpos.copy())
    
    # Setup simulation
    mujoco.set_mjcb_control(lambda m, d: local_sinusoidal_controller(m, d, to_track))
    
    start_pos = to_track[0].xpos.copy()
    
    # Run simulation
    sim_steps = int(SIMULATION_TIME / model.opt.timestep)
    for _ in range(sim_steps):
        mujoco.mj_step(model, data)
    
    end_pos = to_track[0].xpos.copy()
    
    # Calculate fitness
    fitness = evaluate_fitness(individual, start_pos, end_pos)
    
    return individual, fitness, local_history


def tournament_selection(population, tournament_size=3):
    """Select parents using tournament selection."""
    selected = []
    for _ in range(len(population)):
        tournament = np.random.choice(population, tournament_size, replace=False)
        winner = max(tournament, key=lambda ind: ind.fitness)
        selected.append(winner)
    return selected


def crossover(parent1, parent2):
    """Simple one-point crossover."""
    child1 = Individual()
    child2 = Individual()
    
    crossover_point = np.random.randint(1, len(parent1.genotype))
    
    child1.genotype = (parent1.genotype[:crossover_point] + 
                      parent2.genotype[crossover_point:])
    child2.genotype = (parent2.genotype[:crossover_point] + 
                      parent1.genotype[crossover_point:])
    
    return child1, child2


def mutate(individual, mutation_rate=0.1, mutation_strength=0.1):
    mutated = Individual()
    mutated.genotype = individual.genotype.copy()
    
    for i in range(len(mutated.genotype)):
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            mutated.genotype[i] += np.random.normal(0, mutation_strength)
            
            # Clamp values to reasonable ranges
            param_type = i % 3
            if param_type == 0:  # amplitude
                mutated.genotype[i] = np.clip(mutated.genotype[i], 0.01, np.pi/2)
            elif param_type == 1:  # frequency
                mutated.genotype[i] = np.clip(mutated.genotype[i], 0.1, 5.0)
            else:  # phase
                mutated.genotype[i] = mutated.genotype[i] % (2 * np.pi)
    
    return mutated


def show_qpos_history(history):
    """Plot the robot's movement trajectory."""
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    max_range = max(
        abs(pos_data).max(), 0.3
    )  # At least 0.3 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.show()
    PATH_TO_FIGS = "./__figures__"
    plt.savefig(f"{PATH_TO_FIGS}/robot_trajectory.png")


def show_fitness_evolution(fitness_history):
    """Plot the evolution of fitness over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(fitness_history) + 1), fitness_history, 'b-o', linewidth=2, markersize=6)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness (Distance Traveled)")
    plt.title("Evolution of Robot Movement")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    """Main function to run evolutionary algorithm for robot movement."""
    global CURRENT_INDIVIDUAL, POPULATION, GENERATION, HISTORY
    
    print("Starting Evolutionary Algorithm for Robot Movement Learning")
    print(f"Population size: {POPULATION_SIZE}, Generations: {NUM_GENERATIONS}")
    
    # Initialize world and robot
    mujoco.set_mjcb_control(None)
    # Initialize world and robot - SINGLE ROBOT FOR EA TRAINING
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()

    # Get number of joints for genotype creation
    num_joints = model.nu
    print(f"Robot has {num_joints} joints")
    print("Training with SINGLE robot for optimal EA performance...")
    
    # Create initial population
    POPULATION = [create_individual(num_joints) for _ in range(POPULATION_SIZE)]
    
    best_fitness_history = []
    
    # Evolution loop
    for gen in range(NUM_GENERATIONS):
        GENERATION = gen
        print(f"\n=== Generation {gen + 1}/{NUM_GENERATIONS} ===")
        
        # Evaluate each individual
        generation_fitness = []
        for i, individual in enumerate(POPULATION):
            # Reset simulation
            data = mujoco.MjData(model)
            geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
            to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
            
            # Set current individual and reset history
            CURRENT_INDIVIDUAL = individual
            HISTORY = []
            
            # SETUP SIMULATION
            mujoco.set_mjcb_control(lambda m, d: sinusoidal_controller(m, d, to_track))
            
            start_pos = to_track[0].xpos.copy()
            
            # RUN SIMULATION
            sim_steps = int(SIMULATION_TIME / model.opt.timestep)
            for _ in range(sim_steps):
                mujoco.mj_step(model, data)
            
            end_pos = to_track[0].xpos.copy()
            
            # FITNESS EVALUATION
            fitness = evaluate_fitness(individual, start_pos, end_pos)
            generation_fitness.append(fitness)
            
            print(f"  Individual {i+1}: fitness = {fitness:.4f}")
        
        # Track best fitness
        best_fitness = max(generation_fitness)
        best_fitness_history.append(best_fitness)
        best_individual = max(POPULATION, key=lambda ind: ind.fitness)
        
        print(f"  Best fitness this generation: {best_fitness:.4f}")
        print(f"  Average fitness: {np.mean(generation_fitness):.4f}")
        
        # Create next generation (except for last generation)
        if gen < NUM_GENERATIONS - 1:
            # SELECTION
            parents = tournament_selection(POPULATION)
            
            new_population = []
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    # CROSSOVER
                    child1, child2 = crossover(parents[i], parents[i + 1])
                    # MUTATION
                    new_population.extend([mutate(child1), mutate(child2)])
                else:
                    new_population.append(mutate(parents[i]))
            
            # Keep best individual (elitism)
            new_population[0] = best_individual
            POPULATION = new_population[:POPULATION_SIZE]
    
    # Final evaluation with best individual
    print(f"\n~~~ Final Evaluation ~~~")
    best_individual = max(POPULATION, key=lambda ind: ind.fitness)
    CURRENT_INDIVIDUAL = best_individual
    
    print(f"Best evolved fitness: {best_individual.fitness:.4f}")
    print("Best genotype (amplitude, frequency, phase for each joint):")
    genotype = best_individual.genotype
    for j in range(num_joints):
        if j * 3 + 2 < len(genotype):
            amp, freq, phase = genotype[j*3:(j+1)*3]
            print(f"  Joint {j}: amp={amp:.3f}, freq={freq:.3f}, phase={phase:.3f}")
    
    # Run final simulation with video recording
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    HISTORY = []

    

    mujoco.set_mjcb_control(lambda m, d: sinusoidal_controller(m, d, to_track))
    
    # Record video of best individual
    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)
    
    video_renderer(
        model,
        data,
        duration=15,  # Longer video to see the evolved behavior
        video_recorder=video_recorder,
    )
    
    # Show trajectory and fitness evolution
    show_qpos_history(HISTORY)
    show_fitness_evolution(best_fitness_history)



def multi_robot_controller(model, data, individuals, to_track):
    num_joints_per_robot = len(individuals[0].genotype) // 3
    for robot_idx, individual in enumerate(individuals):
        genotype = individual.genotype

        for j in range(num_joints_per_robot):
            ctrl_idx = robot_idx * num_joints_per_robot + j
            if ctrl_idx < model.nu and j * 3 + 2 < len(genotype):
                amplitude = genotype[j * 3]
                frequency = genotype[j * 3 + 1]
                phase = genotype[j * 3 + 2]
                control_value = amplitude * np.sin(frequency * data.time + phase)
                data.ctrl[ctrl_idx] = np.clip(control_value, -np.pi/2, np.pi/2)

def demonstrate_multiple_robots(individuals):
    num_robots = len(individuals)
    print(f"\n~~~ Multi-Robot Demonstration ~~~")
    print(f"Deploying {num_robots} robots with evolved genotypes...")

    # Create world for multiple robots
    mujoco.set_mjcb_control(None)
    multi_world = SimpleFlatWorld([5, 5, 0.1])

    # Spawn multiple robots at random positions
    robots = []
    positions = []
    for _ in range(num_robots):
        x = np.random.uniform(0.1, 4.9)
        y = np.random.uniform(0.1, 4.9)
        z = 0.05
        positions.append([x, y, z])

    for i in range(num_robots):
        robot = gecko()
        robots.append(robot)
        pos = positions[i]
        multi_world.spawn(robot.spec, spawn_position=pos, prefix_id=i)

    multi_model = multi_world.spec.compile()
    multi_data = mujoco.MjData(multi_model)

    # Forward simulate once (so positions are valid)
    mujoco.mj_forward(multi_model, multi_data)

    # Find all robots to track
    multi_geoms = multi_world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    multi_to_track = [multi_data.bind(geom) for geom in multi_geoms if "core" in geom.name]

    # Set controller
    mujoco.set_mjcb_control(lambda m, d: multi_robot_controller(m, d, individuals, multi_to_track))

    # Record video of multiple robots
    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    print(f"Recording {num_robots} robots performing evolved behavior...")
    video_renderer(
        multi_model,
        multi_data,
        duration=20,  # Longer video to see multiple robots
        video_recorder=video_recorder,
    )

    print(f"Multi-robot demonstration complete!")


if __name__ == "__main__":
    main()
    
    num_robots = 5  # Or any number <= POPULATION_SIZE
    top_best = sorted(POPULATION, key=lambda ind: ind.fitness, reverse=True)[:num_robots]
    print(f"Top {len(top_best)} individuals selected for multi-robot demonstration.")
    print("Genotypes of top individuals:")
    for idx, ind in enumerate(top_best):
        print(f" Individual {idx+1}: {ind.genotype}")

    demonstrate_multiple_robots(top_best)