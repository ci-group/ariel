import numpy as np
import mujoco
from cmaes import CMA
import matplotlib.pyplot as plt

# Ariel imports (honestly I still don't fully understand the package structure here)
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from mujoco import viewer


# ==== Global variables ====
HISTORY = []
TARGET_POSITION = np.array([1.0, 0.0])  
TIMESTEPS = 1500
NUM_JOINTS = 8  
NUM_GENERATIONS = 50
POPULATION_SIZE = 60
ALGORITHM_NAME = 'CMAES_ALGO'


# Hopf Oscillator CPG (with coupling)
class HopfCPGNetwork:
    """
    Implements a network of coupled Hopf oscillators used to generate 
    rhythmic control signals for the robot joints.
    """
    def __init__(self, num_oscillators, params):
        """
        Initializes the CPG network parameters from the genome (params vector).
        
        The genome is structured as: 
        [freqs (N), amps (N), weights (N*N), phases (N*N)]
        """
        self.num = num_oscillators

        # Split params based on expected sizes
        split1 = num_oscillators       # frequencies (N)
        split2 = 2 * num_oscillators    # amplitudes (N)
        split3 = split2 + num_oscillators**2 # weights (N*N)
        split4 = split3 + num_oscillators**2 # phases (N*N)

        # Ensure frequencies and amplitudes are positive
        self.freqs = np.abs(params[0:split1])      
        self.amps = np.abs(params[split1:split2])      
        
        self.weights = params[split2:split3].reshape((num_oscillators, num_oscillators))
        self.phases = params[split3:split4].reshape((num_oscillators, num_oscillators))

        # States (r=amplitude, theta=phase)
        self.r = np.ones(num_oscillators) * 0.1
        self.theta = np.random.rand(num_oscillators) * 2 * np.pi

    def step(self, dt=0.01):
        """
        Updates the state of the oscillators based on Hopf dynamics and coupling.
        """
        # Amplitude dynamics (dampens towards target amplitude self.amps)
        dr = 1.0 * (self.amps - self.r) * self.r
        
        # Phase dynamics (natural frequency + coupling)
        dtheta = self.freqs.copy()

        # Coupling effect
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    # Influence from j to i
                    coupling_term = self.weights[i, j] * np.sin(self.theta[j] - self.theta[i] - self.phases[i, j])
                    dtheta[i] += coupling_term

        # Euler integration
        self.r += dr * dt
        self.theta += dtheta * dt

        # Output is the sinusoidal component, driving the joint position
        return self.r * np.sin(self.theta) 

def generate_cpg_output(model, data, cpg, to_track):
    """
    MuJoCo control callback. Calculates CPG output and sets motor torques.
    """
    out = cpg.step()
    # Clip CPG output and apply it as the desired joint position/torque
    # Scaling by (np.pi/2) ensures joint limits are respected
    data.ctrl[:] = np.clip(out * (np.pi/2), -np.pi/2, np.pi/2) 
    
    # Track the core position for fitness evaluation
    if len(to_track) > 0:
        HISTORY.append(to_track[0].xpos.copy())

def controller(genome, visualize=False):
    """
    Runs the MuJoCo simulation for one trial using the given CPG genome.
    """
    HISTORY.clear()

    # --- Build world and robot ---
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Find the core geometry to track its position
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    cpg = HopfCPGNetwork(NUM_JOINTS, genome)

    # Set the custom control callback
    mujoco.set_mjcb_control(lambda m, d: generate_cpg_output(m, d, cpg, to_track))

    if visualize:
        # Launch the MuJoCo viewer for visualization
        viewer.launch(model, data)
    else:
        # Run the simulation steps
        for _ in range(TIMESTEPS):
            mujoco.mj_step(model, data)
            if len(HISTORY) > 0:
                current_pos = HISTORY[-1][:2]
                # Early termination if target is reached
                if np.linalg.norm(current_pos - TARGET_POSITION) < 0.05:
                    break

    # Unset the control callback
    mujoco.set_mjcb_control(None)
    
    return HISTORY.copy()

def eval_fitness_controller(genome):
    """
    Evaluates the fitness of a genome.
    MAXIMIZE X-progress and MINIMIZE Y-drift.
    Since CMA-ES minimizes, we use: Fitness = -X_final + 5 * |Y_final|
    Lower fitness is better (best possible score is -1.0), though I haven't reached that value in testing YET.
    """
    hist = controller(genome, visualize=False)
    if len(hist) == 0:
        # If no history, assign a high positive penalty
        return 10.0
        
    final_pos = np.array(hist[-1][:2])
    x_final = final_pos[0]
    y_final = final_pos[1]
    
    # eward Forward Progress (Minimize -X_final)
    x_progress_term = -x_final 
    
    # Penalize Lateral Drift (Minimize 5 * |Y_final|)
    y_penalty_term = 5.0 * np.abs(y_final) 
    
    # Final Fitness (to be minimized)
    fitness = x_progress_term + y_penalty_term
    
    # Euclidean distance fitness is also an option, though it felt very limited.

    return fitness

def plot_fitness_over_generations(all_fitness_histories, algorithm, window=5, save_path="fitness_plot.png"):
    """
    Plots the mean and standard deviation of fitness across all runs.
    """
    # Convert list of lists (fitness per generation per run) to numpy array
    # Shape = (num_runs, num_generations, pop_size)
    fitness_array = np.array(all_fitness_histories, dtype=object)
    
    # Convert inner lists to arrays for correct mean calculation
    max_gen = max(len(h) for h in all_fitness_histories)
    processed_fitness = []
    for run_hist in all_fitness_histories:
        run_data = []
        for gen_fitness in run_hist:
             run_data.append(np.mean(gen_fitness))
        # Pad with the last value if runs ended early (lazy I know, but I'm tired and it works)
        while len(run_data) < max_gen:
            run_data.append(run_data[-1] if run_data else np.inf)
        processed_fitness.append(run_data)
        
    processed_fitness = np.array(processed_fitness) # Shape (num_runs, num_generations)
    
    # Mean and std across runs
    mean_fitness = processed_fitness.mean(axis=0)
    std_fitness = processed_fitness.std(axis=0)

    # Moving average
    if window > 1 and len(mean_fitness) >= window:
        ma_fitness = np.convolve(mean_fitness, np.ones(window)/window, mode='valid')
        ma_std = np.convolve(std_fitness, np.ones(window)/window, mode='valid')
        ma_generations = np.arange(window-1, len(mean_fitness))
    else:
        ma_fitness = mean_fitness
        ma_std = std_fitness
        ma_generations = np.arange(len(mean_fitness))

    generations = np.arange(len(mean_fitness))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_fitness, 'b-', alpha=0.5, label=f'Mean Fitness (Raw)')
    plt.plot(ma_generations, ma_fitness, 'r-', linewidth=2, label=f'Moving Avg ({window} gen)')
    
    # Plot standard deviation range
    plt.fill_between(ma_generations, ma_fitness - ma_std, ma_fitness + ma_std,
                     color='gray', alpha=0.3, label='+/- Std Dev')
                     
    plt.xlabel('Generation')
    plt.ylabel('Fitness (Minimize: -X + 5|Y|)')
    plt.title(f'Fitness Evolution ({algorithm})')
    plt.legend()
    plt.grid(True)
    plt.savefig(algorithm + "_" + save_path, dpi=300)
    plt.close()

def plot_best_trajectory(history, algorithm, target=None, save_path="best_trajectory.png"):
    """
    Plots the XY trajectory of the robot core position over time.
    """
    pos_data = np.array(history)
    if pos_data.ndim == 1:
        print("Warning: Trajectory history is too short to plot.")
        return

    plt.figure(figsize=(8,6))
    
    # Path
    plt.plot(pos_data[:,0], pos_data[:,1], 'b-', label='Path')
    plt.plot(pos_data[0,0], pos_data[0,1], 'go', markersize=10, label='Start')
    plt.plot(pos_data[-1,0], pos_data[-1,1], 'ro', markersize=10, label='End')

    # Target
    if target is not None:
        plt.scatter([target[0]], [target[1]], c='red', marker='*', s=300, label='Target')

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Best Individual Trajectory ({algorithm})")
    plt.legend()
    plt.grid(True)

    # Set axis limits with padding
    x_coords = np.append(pos_data[:,0], target[0]) if target is not None else pos_data[:,0]
    y_coords = np.append(pos_data[:,1], target[1]) if target is not None else pos_data[:,1]
    
    # Calculate a sensible square plot area
    all_coords = np.concatenate([x_coords, y_coords])
    min_coord = all_coords.min() - 0.5
    max_coord = all_coords.max() + 0.5
    
    plt.xlim(min_coord, max_coord)
    plt.ylim(min_coord, max_coord)

    plt.axis('equal')
    plt.savefig(algorithm + "_" + save_path, dpi=300)
    plt.close()


def cmaes_algorithm(x0, sigma0, bounds):
    """
    Runs the CMA-ES optimization loop.
    """
    cma = CMA(mean=x0, sigma=sigma0, bounds=bounds, population_size=POPULATION_SIZE)

    fitness_history = []
    best_genome_overall = None
    best_fitness_overall = float('inf')

    for generation in range(NUM_GENERATIONS):
        solutions = []
        gen_fitness = []
        
        # Ask for new solutions
        for _ in range(cma.population_size):
            x = cma.ask()
            value = eval_fitness_controller(x)
            solutions.append((x, value))
            gen_fitness.append(value)
        
        # Tell the algorithm the results
        cma.tell(solutions)
        fitness_history.append(gen_fitness)

        # Track the best solution found in this generation
        current_best_genome, current_best_fitness = min(solutions, key=lambda s: s[1])
        
        if current_best_fitness < best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_genome_overall = current_best_genome
            
        print(f"Gen {generation+1}/{NUM_GENERATIONS} | Best fitness (Gen): {current_best_fitness:.3f} | Best fitness (Overall): {best_fitness_overall:.3f}")

    # Return the best genome found across all generations of this run(3 runs is plenty)
    return fitness_history, best_genome_overall


if __name__ == "__main__":
    
    # genome size = freqs + amps + weights + phases
    genome_size = (NUM_JOINTS * 2) + (NUM_JOINTS**2) * 2
    
    # Initial guess and search space definition
    x0 = np.random.uniform(0.1, 1.0, genome_size)
    sigma0 = 0.5 # Initial standard deviation for the search
    bounds = np.tile([[-2.0, 2.0]], (genome_size, 1))

    all_fitness_histories = []
    best_genomes = []
    NUM_RUNS = 1

    print(f"RUNNING {ALGORITHM_NAME} for {NUM_RUNS} runs...")

    for run in range(NUM_RUNS):
        print(f"\n--- STARTING RUN {run+1}/{NUM_RUNS} ---")
        
        # Run CMA-ES optimization
        fitness_history, best_genome = cmaes_algorithm(x0, sigma0, bounds)
        
        all_fitness_histories.append(fitness_history)
        best_genomes.append(best_genome)

    # --- Analysis and Visualization ---

    # Visualize the trajectory of the best genome from the final run
    print("\nVisualizing best genome from last run...")
    controller(best_genomes[-1], visualize=True)

    # Plot averaged results across all runs
    plot_fitness_over_generations(all_fitness_histories, algorithm=ALGORITHM_NAME, window=5)

    # Plot trajectory of the best genome from the final run
    best_history = controller(best_genomes[-1], visualize=False)
    plot_best_trajectory(best_history, ALGORITHM_NAME, target=TARGET_POSITION)

    #Just save the plot if it doesn't open up automatically, or we can write a generalized function later.
    plt.show()

    # Save results
    np.savez("experiment_results_cmaes.npz", 
             all_fitness_histories=np.array(all_fitness_histories, dtype=object), 
             best_genomes=np.array(best_genomes, dtype=object))

    print("\nOptimization complete. Results saved and plots generated.")
