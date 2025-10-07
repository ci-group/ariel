import numpy as np
import mujoco
from cmaes import CMA
import matplotlib.pyplot as plt

# Ariel imports (honestly I still don't fully understand the package structure here)
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from mujoco import viewer


# ==== Global variables ====from pyexpat import model
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time, copy, random
from deap import base, creator, tools
from functools import partial
import logging

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments import SimpleFlatWorld, TiltedFlatWorld, BoxyRugged, RuggedTerrainWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.utils.runners import simple_runner
from ariel.utils.renderers import single_frame_renderer

import json
import os
from datetime import datetime

# -----------------------------
# Utility functions
# -----------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def weights_from_list_to_matrix(individual, input_size, hidden_size, output_size):
    """
    Convert a flat genome list into the weight matrices W1, W2, W3.
    W1 shape: (input_size + 1, hidden_size)
    W2 shape: (hidden_size, hidden_size)
    W3 shape: (hidden_size, output_size)
    """
    index = 0
    input_size_bias = input_size + 1
    W1_size = input_size_bias * hidden_size
    W2_size = hidden_size * hidden_size
    W3_size = hidden_size * output_size

    W1 = np.array(individual[index: index + W1_size]).reshape((input_size_bias, hidden_size))
    index += W1_size
    W2 = np.array(individual[index: index + W2_size]).reshape((hidden_size, hidden_size))
    index += W2_size
    W3 = np.array(individual[index: index + W3_size]).reshape((hidden_size, output_size))

    return {"W1": W1, "W2": W2, "W3": W3}


# -----------------------------
# Controller
# -----------------------------

def persistent_controller(to_track, weights, history_controller, actuator_limit=None):
    """Return a controller closure that uses the provided weights.

    actuator_limit: a scalar or array-like of shape (model.nu,) that gives safe limits.
    If None, the controller will clip outputs to +/- pi/2 as a reasonable default.
    """

    def controller(model, data):
        # inputs: qpos plus bias
        inputs = np.append(data.qpos, 1.0)

        # forward pass
        layer1 = np.tanh(np.dot(inputs, weights['W1']))
        layer2 = np.tanh(np.dot(layer1, weights['W2']))
        outputs = np.sin(np.dot(layer2, weights['W3']))  # in [-1,1]

        # scale outputs to actuator ranges
        if actuator_limit is None:
            scaled = outputs * (np.pi / 2)
            clip_min, clip_max = -np.pi / 2, np.pi / 2
        else:
            # if actuator_limit is a scalar assume symmetric
            try:
                arr = np.array(actuator_limit)
                if arr.shape == ():
                    scaled = outputs * float(arr)
                    clip_min, clip_max = -float(arr), float(arr)
                else:
                    scaled = outputs * arr
                    clip_min, clip_max = np.min(-arr), np.max(arr)
            except Exception:
                scaled = outputs * (np.pi / 2)
                clip_min, clip_max = -np.pi / 2, np.pi / 2

        # assign safely to data.ctrl view
        try:
            data.ctrl[:] = np.clip(scaled, clip_min, clip_max)
        except Exception:
            # fallback assignment
            data.ctrl = np.clip(scaled, clip_min, clip_max)

        # Save core position to history if available
        if len(to_track) > 0:
            try:
                # ensure copy so later modifications don't mutate history
                history_controller.append(to_track[0].xpos.copy())
            except Exception:
                # if binding fails silently continue
                pass

    return controller


# -----------------------------
# Visualization helpers
# -----------------------------

def plot(best_history, NUM_GENERATION):
    best_history = np.array(best_history)
    generation = list(range(len(best_history)))
    plt.figure()
    plt.plot(generation, best_history)
    plt.xlabel("generation")
    plt.ylabel("best_fitness")
    plt.tight_layout()
    plt.show()


def show_qpos_history(history: list):
    if len(history) == 0:
        print("No history to plot")
        return

    pos_data = np.array(history)
    plt.figure(figsize=(8, 6))
    plt.plot(pos_data[:, 0], pos_data[:, 1], label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    max_range = max(0.3, np.abs(pos_data).max())
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    plt.tight_layout()
    plt.show()


# -----------------------------
# JSON saving
# -----------------------------

def save_results_json(output_folder, filename_prefix,
                      fitness_history, best_individual,
                      input_size, hidden_size, output_size,
                      final_population=None):

    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    out_path = os.path.join(output_folder, filename)

    generations_obj = {f"gen{i}": float(v) for i, v in enumerate(fitness_history)}

    total_generations = len(fitness_history)
    final_best = float(fitness_history[-1]) if len(fitness_history) > 0 else 0.0

    if final_population is not None and len(final_population) > 0:
        pop_vals = [float(ind.fitness.values[0]) for ind in final_population]
        final_best = float(max(pop_vals))
        final_mean = float(np.mean(pop_vals))
        final_worst = float(min(pop_vals))
    else:
        final_mean = final_best
        final_worst = final_best

    metadata = {
        "total_generations": int(total_generations),
        "final_best": final_best,
        "final_mean": final_mean,
        "final_worst": final_worst
    }

    best_genome = [float(x) for x in best_individual]
    weights = weights_from_list_to_matrix(best_genome, input_size, hidden_size, output_size)
    weights_lists = {k: v.tolist() for k, v in weights.items()}

    out_data = {
        "generations": generations_obj,
        "metadata": metadata,
        "weights": weights_lists
    }

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"Saved JSON to {out_path}")
    return out_path


# -----------------------------
# Fitness evaluation
# -----------------------------

def fitness_eval_ind(ind, duration = 20):
    mujoco.set_mjcb_control(None)
    world = RuggedTerrainWorld()#BoxyRugged()#TiltedFlatWorld() #SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]


    input_size = len(data.qpos)
    hidden_size = 12
    output_size = model.nu


    weights = weights_from_list_to_matrix(ind, input_size, hidden_size, output_size)
    HISTORY = []
    controller = persistent_controller(to_track,weights, HISTORY)
    mujoco.set_mjcb_control(lambda m,d: controller(m, d))


    simple_runner(model=model,data=data,duration=duration)
    mujoco.set_mjcb_control(None)
    if len(HISTORY) < 2:
        return (0.0,)
    positions = np.array(HISTORY)


    dy = positions[-1,1] - positions[0,1] # forward progress
    dx = abs(positions[-1,0] - positions[0,0]) # sideways drift
    penalty_coeff = 1


    # ✅ reward forward movement, penalize sideways
    fitness = max(0, dy - (dx * penalty_coeff))


    return (fitness,)


# -----------------------------
# DEAP setup (safe)
# -----------------------------
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
except Exception:
    # already created in this session
    pass

try:
    creator.create("Individual", list, fitness=creator.FitnessMax)
except Exception:
    pass


# -----------------------------
# Main GA loop
# -----------------------------

def main():
    # Logging
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

    # RNG seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    # GA hyperparameters
    NUM_GENERATIONS = 10
    POP_SIZE = 50
    MATE_CHANCE = 0.7
    MUTATE_CHANCE = 0.3
    ELITES = 1
    MUT_SIGMA = 0.3
    MAX_MUT_SIGMA = 3
    NO_IMPROVE = 0
    STAGNATION_WINDOW = 5
    BEST_FOR_NOW = -np.inf
    MAX_STAGNATION = 7

    # Debug flags
    DEBUG = False
    DEBUG_DURATION = 2
    DEFAULT_DURATION = 20
    eval_duration = DEBUG_DURATION if DEBUG else DEFAULT_DURATION

    # Create a small dummy world to compute sizes (safe pre-check)
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    input_size = len(data.qpos)
    input_size_bias = input_size + 1

    hidden_size = 12
    output_size = model.nu

    genome_length = (input_size_bias * hidden_size) + (hidden_size * hidden_size) + (hidden_size * output_size)
    logging.info(f"Genome length computed = {genome_length}")

    # DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_param", lambda: random.uniform(-1, 1))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_param, n=genome_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register fitness with a duration parameter closure
    toolbox.register("evaluate_fitness", lambda ind: fitness_eval_ind(ind, duration=eval_duration))

    toolbox.register("mate", tools.cxUniform, indpb=0.5)

    # changing_sigma as wrapper mutate function
    def changing_sigma_wrapper(individual):
        nonlocal MUT_SIGMA
        return tools.mutGaussian(individual, mu=0.0, sigma=MUT_SIGMA, indpb=0.1)

    toolbox.register("mutate", changing_sigma_wrapper)
    toolbox.register("select", tools.selTournament, tournsize=6)

    # initialize population
    pop = toolbox.population(n=POP_SIZE)

    # initial evaluation
    logging.info("Evaluating initial population...")
    for i, ind in enumerate(pop):
        ind.fitness.values = toolbox.evaluate_fitness(ind)
        logging.debug(f"Init individual {i} fitness = {ind.fitness.values[0]}")

    best_individual_ever = tools.HallOfFame(1)
    fitness_history = []

    for gen in range(NUM_GENERATIONS):
        best = tools.selBest(pop, 1)[0]

        if best.fitness.values[0] <= BEST_FOR_NOW + 1e-8:
            NO_IMPROVE += 1
        else:
            NO_IMPROVE = 0
            BEST_FOR_NOW = best.fitness.values[0]

        if NO_IMPROVE >= STAGNATION_WINDOW and MUT_SIGMA < MAX_MUT_SIGMA:
            MUT_SIGMA *= 1.3
            logging.info(f"increasing sigma to get out of local minimum to {MUT_SIGMA}")
        else:
            MUT_SIGMA = max(0.05, MUT_SIGMA * 0.995)

        if NO_IMPROVE >= MAX_STAGNATION:
            logging.info(f"No improvement for {MAX_STAGNATION} generations. Stopping at generation {gen}.")
            break

        # selection and variation
        parents = toolbox.select(pop, POP_SIZE - ELITES)
        parents = list(map(copy.deepcopy, parents))
        random.shuffle(parents)

        new_children = []
        for parent1, parent2 in zip(parents[::2], parents[1::2]):
            child1, child2 = parent1, parent2
            if random.random() < MATE_CHANCE:
                toolbox.mate(child1, child2)
                if hasattr(child1, 'fitness'):
                    del child1.fitness.values
                if hasattr(child2, 'fitness'):
                    del child2.fitness.values
            if random.random() < MUTATE_CHANCE:
                toolbox.mutate(child1)
                toolbox.mutate(child2)
                if hasattr(child1, 'fitness'):
                    del child1.fitness.values
                if hasattr(child2, 'fitness'):
                    del child2.fitness.values
            new_children.extend([child1, child2])

        # evaluate invalid offspring
        invalid = [ind for ind in new_children if not hasattr(ind, 'fitness') or not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate_fitness(ind)

        elites = tools.selBest(pop, ELITES)
        total_population = pop + new_children
        survivors = toolbox.select(total_population, POP_SIZE - ELITES)
        pop[:] = survivors + elites

        best = tools.selBest(pop, 1)[0]
        fitness_history.append(best.fitness.values[0])
        best_individual_ever.update(pop)

        logging.info(f"generation {gen}: best fitness = {best.fitness.values[0]:.6f}")

    # end GA
    best_ind = best_individual_ever[0]

    out_path = save_results_json(
        output_folder="./results",
        filename_prefix="fitness_statistics",
        fitness_history=fitness_history,
        best_individual=best_ind,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        final_population=pop
    )

    # visualize best agent
    HISTORY_BEST = []
    weights = weights_from_list_to_matrix(best_ind, input_size, hidden_size, output_size)

    # Spawn a fresh world for visualization
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    actuator_limit = None
    try:
        if hasattr(model, 'actuator_ctrlrange'):
            arr = np.array(model.actuator_ctrlrange)
            actuator_limit = np.max(np.abs(arr), axis=1)
            if np.allclose(actuator_limit, actuator_limit[0]):
                actuator_limit = float(actuator_limit[0])
    except Exception:
        actuator_limit = None

    global_controller = persistent_controller(to_track, weights, HISTORY_BEST, actuator_limit=actuator_limit)
    mujoco.set_mjcb_control(lambda m, d: global_controller(m, d))

    # Launch viewer to inspect behaviour
    viewer.launch(model, data)

    # After closing viewer, plot and save results
    show_qpos_history(HISTORY_BEST)
    plot(fitness_history, len(fitness_history))


if __name__ == "__main__":
    main()
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
