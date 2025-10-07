import numpy as np
import mujoco
from cmaes import CMA

# Local libraries
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

import matplotlib.pyplot as plt
import numpy as np
from mujoco import viewer



# ==== Global variables ====
HISTORY = []
TARGET_POSITION = np.array([1.0, 0.0])   
TIMESTEPS = 7500
NUM_JOINTS = 8   
NUM_GENERATIONS = 10
POPULATION_SIZE = 64

CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.1

ALGORITHM_TO_RUN = 'GENETIC_ALGO'
# ALGORITHM_TO_RUN = 'CMA_ALGO'


# Hopf Oscillator CPG (with coupling)
class HopfCPGNetwork:
    def __init__(self, num_oscillators, params):
        """
        params vector contains:
        - freq[i], amp[i] for each oscillator
        - coupling weights matrix (flattened)
        - phase biases matrix (flattened)
        """
        self.num = num_oscillators

        # Split params
        split1 = num_oscillators        # frequencies
        split2 = 2 * num_oscillators    # amplitudes
        split3 = split2 + num_oscillators**2
        split4 = split3 + num_oscillators**2

        self.freqs = np.abs(params[0:split1])           # positive frequencies
        self.amps = np.abs(params[split1:split2])       # positive amplitudes
        self.weights = params[split2:split3].reshape((num_oscillators, num_oscillators))
        self.phases = params[split3:split4].reshape((num_oscillators, num_oscillators))

        # States
        self.r = np.ones(num_oscillators) * 0.1
        self.theta = np.random.rand(num_oscillators) * 2 * np.pi

    def step(self, dt=0.01):
        dr = 1.0 * (self.amps - self.r) * self.r
        dtheta = self.freqs.copy()

        # Coupling effect
        for i in range(self.num):
            for j in range(self.num):
                if i != j:
                    dtheta[i] += self.weights[i, j] * np.sin(self.theta[j] - self.theta[i] - self.phases[i, j])

        self.r += dr * dt
        self.theta += dtheta * dt

        return self.r * np.sin(self.theta)   # oscillator outputs

def generate_cpg_output(model, data, cpg, to_track):
    out = cpg.step()
    data.ctrl[:] = np.clip(out * (np.pi/2), -np.pi/2, np.pi/2) # bound to valid torque range
    HISTORY.append(to_track[0].xpos.copy())

def controller(genome, visualize=False):
    HISTORY.clear()

    # --- Build world and robot ---
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]


    cpg = HopfCPGNetwork(NUM_JOINTS, genome)

    mujoco.set_mjcb_control(lambda m, d: generate_cpg_output(m, d, cpg, to_track))

    if visualize:
        viewer.launch(model, data)
    else:
        for _ in range(TIMESTEPS):
            mujoco.mj_step(model, data)
            if len(HISTORY) > 0:
                current_pos = HISTORY[-1][:2]
                if np.linalg.norm(current_pos - TARGET_POSITION) < 0.05:
                    break

    mujoco.set_mjcb_control(None)
    return HISTORY.copy()

def eval_fitness_controller(genome):
    hist = controller(genome, visualize=False)
    if len(hist) == 0:
        return float("inf")
    final_pos = np.array(hist[-1][:2])
    dist = np.linalg.norm(final_pos - TARGET_POSITION)
    return dist

def moving_average(arr, w):
    return np.convolve(arr, np.ones(w)/w, mode='valid')

def plot_fitness_over_generations(fitness_history, algorithm, window=5, save_path="fitness_plot.png"):

    fitness_array = np.array(fitness_history)
    mean_fitness = fitness_array.mean(axis=1)
    std_fitness = fitness_array.std(axis=1)

    # Moving average
    if window > 1 and len(mean_fitness) >= window:
        ma_fitness = np.convolve(mean_fitness, np.ones(window)/window, mode='valid')
        ma_generations = np.arange(window-1, len(mean_fitness))
    else:
        ma_fitness = mean_fitness
        ma_generations = np.arange(len(mean_fitness))

    generations = np.arange(len(mean_fitness))

    plt.figure(figsize=(10, 6))
    plt.plot(generations, mean_fitness, 'b-', label='Mean Fitness')
    plt.plot(ma_generations, ma_fitness, 'r-', label=f'Moving Avg (window={window})')
    plt.fill_between(generations, mean_fitness - std_fitness, mean_fitness + std_fitness,
                     color='gray', alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Evolution Over Generations')
    plt.legend()
    plt.grid(True)
    plt.savefig(algorithm+save_path, dpi=300)
    plt.close()

def plot_best_trajectory(history, algorithm, target=None):
    """
    Plots the XY trajectory of the robot over time.

    Parameters
    ----------
    history : list of np.ndarray
        Robot core positions recorded over time [(x0,y0,z0), ...]
    target : np.ndarray, optional
        Target position [x, y] to plot
    """
    pos_data = np.array(history)
    plt.figure(figsize=(8,6))
    
    # Path
    plt.plot(pos_data[:,0], pos_data[:,1], 'b-', label='Path')
    plt.plot(pos_data[0,0], pos_data[0,1], 'go', label='Start')
    plt.plot(pos_data[-1,0], pos_data[-1,1], 'ro', label='End')

    # Target
    if target is not None:
        plt.scatter([target[0]], [target[1]], c='red', marker='*', s=200, label='Target')

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Best Individual Trajectory")
    plt.legend()
    plt.grid(True)

    # Set axis limits with padding
    x_coords = np.append(pos_data[:,0], target[0]) if target is not None else pos_data[:,0]
    y_coords = np.append(pos_data[:,1], target[1]) if target is not None else pos_data[:,1]
    plt.xlim(x_coords.min()-0.5, x_coords.max()+0.5)
    plt.ylim(y_coords.min()-0.5, y_coords.max()+0.5)

    plt.axis('equal')
    plt.show()

    plt.savefig(algorithm+'best_trajectory.png', dpi=300)


if __name__ == "__main__":
    # genome size = freqs + amps + weights + phases
    genome_size = (NUM_JOINTS * 2) + (NUM_JOINTS**2) * 2
    x0 = np.random.uniform(0.1, 1.0, genome_size)
    sigma0 = 0.5

    bounds = np.tile([[-2.0, 2.0]], (genome_size, 1))

    from scipy.optimize import differential_evolution


    if ALGORITHM_TO_RUN == 'GENETIC_ALGO':
        print(f"RUNNING {ALGORITHM_TO_RUN}")
        pop = np.random.uniform(bounds[:,0], bounds[:,1], (POPULATION_SIZE, genome_size))
        fitness = np.array([eval_fitness_controller(ind) for ind in pop])
        fitness_history = [fitness.copy()]


        for gen in range(NUM_GENERATIONS):
            new_pop = []
            while len(new_pop) < POPULATION_SIZE:
                # --- Selection: tournament ---
                idx1, idx2 = np.random.choice(POPULATION_SIZE, 2, replace=False)
                parent1 = pop[idx1] if fitness[idx1] < fitness[idx2] else pop[idx2]

                idx3, idx4 = np.random.choice(POPULATION_SIZE, 2, replace=False)
                parent2 = pop[idx3] if fitness[idx3] < fitness[idx4] else pop[idx4]

                # --- Crossover ---
                if np.random.rand() < CROSSOVER_RATE:
                    cross_point = np.random.randint(1, genome_size)
                    child1 = np.concatenate([parent1[:cross_point], parent2[cross_point:]])
                    child2 = np.concatenate([parent2[:cross_point], parent1[cross_point:]])
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # --- Mutation ---
                for child in [child1, child2]:
                    for i in range(genome_size):
                        if np.random.rand() < MUTATION_RATE:
                            child[i] += np.random.normal(0, 0.1)
                    # Clip to bounds
                    child[:] = np.clip(child, bounds[:,0], bounds[:,1])
                    new_pop.append(child)
                    if len(new_pop) >= POPULATION_SIZE:
                        break

            # Evaluate new population
            pop = np.array(new_pop)
            fitness = np.array([eval_fitness_controller(ind) for ind in pop])
            fitness_history.append(fitness.copy())

            # Best individual
            best_idx = np.argmin(fitness)
            print(f"Gen {gen} | Best fitness: {fitness[best_idx]:.3f}")

        # Extract best genome
        best_idx = np.argmin(fitness)
        best_genome = pop[best_idx]
        best_fitness = fitness[best_idx]
        print(f"Best fitness overall: {best_fitness:.3f}")
    elif ALGORITHM_TO_RUN == 'CMA_ALGO':
        print(f"RUNNING {ALGORITHM_TO_RUN}")
        cma = CMA(mean=x0, sigma=sigma0, bounds=bounds, population_size=32)

        fitness_history = []  # each element = list of fitness values in that generation

        for generation in range(NUM_GENERATIONS):
            solutions = []
            gen_fitness = []
            for _ in range(POPULATION_SIZE):
                x = cma.ask()
                value = eval_fitness_controller(x)
                solutions.append((x, value))
                gen_fitness.append(value)
            cma.tell(solutions)
            fitness_history.append(gen_fitness)


            best_genome, best_fitness = min(solutions, key=lambda s: s[1])
            print(f"Gen {generation} | Best fitness: {best_fitness:.3f}")

    # Visualize best controller
    print("Visualizing best genome...")
    controller(best_genome, visualize=True)

    # Example usage after CMA loop
    plot_fitness_over_generations(fitness_history, algorithm=ALGORITHM_TO_RUN, window=5)

    best_genome, _ = min(solutions, key=lambda s: s[1])
    best_history = controller(best_genome, visualize=False)
    plot_best_trajectory(best_history, ALGORITHM_TO_RUN, target=TARGET_POSITION)