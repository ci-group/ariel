import numpy as np
import itertools, time
import matplotlib.pyplot as plt
from collections import defaultdict

from ariel.ec.drone.inspection.animate_evolution import plot_population
from ariel.ec.drone.inspection.drone_visualizer import DroneVisualizer

def get_first_x_primes(x):
    if x < 1:
        return []

    # Estimate upper bound for nth prime using the prime number theorem
    # This is a rough estimate and can be adjusted for better performance
    n = int(x * (np.log(x) + np.log(np.log(x)))) if x > 5 else 15

    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(np.sqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i * i, n, i):
                is_prime[j] = False

    primes = [i for i, prime in enumerate(is_prime) if prime]
    return primes[:x]

class SearchSpace:
    def __init__(self, bounds, num_bins):
        self.bounds = bounds
        self.num_bins = num_bins
        self.visits = defaultdict(int)
        self.points_in_cubes = defaultdict(list)
        self.individuals_in_cubes = defaultdict(list)
        self.gen_num_in_cubes = defaultdict(list)
        self.bin_sizes = [(b[1] - b[0]) / num_bins if b[0] != b[1] else None for b in bounds]

    def visit(self, point, individual, generation_num):
        cube = self._get_cube(point)
        cube_key = self._hash_cube(cube)
        self.visits[cube_key] += 1
        self.points_in_cubes[cube_key].append(point)
        self.individuals_in_cubes[cube_key].append(individual)
        self.gen_num_in_cubes[cube_key].append(generation_num)

    def _get_cube(self, point):
        cube = []
        for p, b, bin_size in zip(point, self.bounds, self.bin_sizes):
            if bin_size is None:
                continue  # Skip this dimension if bounds are equal
            cube_index = int((p - b[0]) / bin_size)
            cube.append(cube_index)
        return tuple(cube)
    
    def get_cube_bounds(self, cube):
        bounds = []
        for index, (b, bin_size) in zip(cube, zip(self.bounds, self.bin_sizes)):
            if bin_size is None:
                bounds.append((b[0], b[1]))  # If bin_size is None, use the original bounds
            else:
                lower_bound = b[0] + index * bin_size
                upper_bound = lower_bound + bin_size
                bounds.append([lower_bound, upper_bound])
        return np.array(bounds)
    
    def _hash_cube(self, cube):
        return hash(cube)
    
    def percentage_of_sampled_cubes(self):
        return len(self.visits) / (self.num_bins ** len(self.bounds))

    def spread_of_sampled_cubes(self):
        points = list(self.visits.keys())
        distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in itertools.combinations(points, 2)]
        return np.mean(distances), np.std(distances)
    
    def spread_of_points(self):
        points = list(itertools.chain(*self.points_in_cubes.values()))
        distances = [np.linalg.norm(np.array(p1) - np.array(p2)) for p1, p2 in itertools.combinations(points, 2)]
        return np.mean(distances), np.std(distances)
    
    def most_popular_cube(self):
        return max(self.visits, key=self.visits.get), self.visits[max(self.visits, key=self.visits.get)]

    def num_visited_cubes(self):
        """
        Return the number of unique cubes visited.
        
        :return: Number of visited cubes.
        """
        return len(self.visits)

def get_search_space_info(point_data, population_data, bounds, num_bins=100):
    point_num_gen, point_pop_size, point_nparams = point_data.shape
    num_gen, pop_size, narms, nparams = population_data.shape
    assert point_num_gen == num_gen
    assert point_pop_size == pop_size

    search_space = SearchSpace(bounds, num_bins)
    number_of_sampled_cubes = np.empty(num_gen)
    rate_of_sampled_cubes = np.empty(num_gen)
    mean_distance,std_distance = np.empty(num_gen), np.empty(num_gen)
    prev_num_sampled_cubes = 0
    start = time.time()
    for g in range(num_gen):
        for i in range(pop_size):
            point = point_data[g, i]
            individual = population_data[g, i]
            
            search_space.visit(point, individual, g)

        num_samp_cubes = search_space.num_visited_cubes()
        number_of_sampled_cubes[g] = num_samp_cubes
        rate_of_sampled_cubes[g] = num_samp_cubes - prev_num_sampled_cubes
        prev_num_sampled_cubes = num_samp_cubes
        mean_distance[g], std_distance[g] = search_space.spread_of_sampled_cubes()
        print(f"Generation {g}, time: {time.time() - start}")
        start = time.time()
    
    return search_space, number_of_sampled_cubes, rate_of_sampled_cubes, mean_distance, std_distance

def plot_num_sampled_cubes_info(number_of_sampled_cubes, rate_of_sampled_cubes, mean_distance, std_distance):
    generations = np.arange(0,len(number_of_sampled_cubes))

    fig, axs = plt.subplots(1,3)
    axs[0].plot(generations, number_of_sampled_cubes, label="Num Sampled Cubes")
    axs[0].set_xlabel('Generation')
    axs[0].set_ylabel('Num of Sampled Cubes')
    axs[0].grid()
    axs[1].plot(generations, rate_of_sampled_cubes, label="Density Sampled Cubes")
    axs[1].set_xlabel('Generation')
    axs[1].set_ylabel('Number of Sampled Cubes Gained')
    axs[1].grid()
    axs[2].plot(generations, mean_distance, label="Mean Pairwise Distances")
    axs[2].fill_between(generations, mean_distance-std_distance, mean_distance+std_distance, alpha=0.2, label='Std Dev Pairwise Distances')
    axs[2].set_xlabel('Generation')
    axs[2].set_ylabel('Distance between points')
    axs[2].legend()
    axs[2].grid()

def plot_most_sampled_cubes(search_space, save_name=None, twod=True):
    # Plot a bar chart of the top 10 most sampled cubes
    top_10_cubes = sorted(search_space.visits.items(), key=lambda item: item[1], reverse=True)[:10]

    labels = []
    sizes = []
    for i, (hashed_cube, count) in enumerate(top_10_cubes):

        points_in_most_popular_cube = search_space.points_in_cubes[hashed_cube]
        individuals_in_most_popular_cube = np.array(search_space.individuals_in_cubes[hashed_cube])
        gen_nums = np.array(search_space.gen_num_in_cubes[hashed_cube])

        _, indices = np.unique(np.nan_to_num(individuals_in_most_popular_cube), return_index=True, axis=0)
        unique_individuals = individuals_in_most_popular_cube[indices]
        unique_gen_nums = gen_nums[indices]

        num_unique_individuals = unique_individuals.shape[0]

        cube = search_space._get_cube(points_in_most_popular_cube[0])
        cube_bounds = np.round(search_space.get_cube_bounds(cube), decimals=3)
        
        if num_unique_individuals > 1:
            plot_population(unique_individuals, fitnesses=unique_gen_nums, title=f"Cube {i}, Bounds: {cube_bounds}", include_motor_orientation=1, twod=twod)
            if save_name is not None:
                plt.savefig(f"{save_name}_cube{i}.png")
        elif num_unique_individuals == 1:
            visualizer = DroneVisualizer()
            if twod:
                fig, ax = visualizer.plot_2d(unique_individuals[0], title=f"Cube {i}, Gen {unique_gen_nums[0]}, Bounds: {cube_bounds}")
            else:
                fig, ax = visualizer.plot_3d(unique_individuals[0], title=f"Cube {i}, Gen {unique_gen_nums[0]}, Bounds: {cube_bounds}")
            if save_name is not None:
                plt.savefig(f"{save_name}_cube{i}.png")

        labels.append(f"Ucount: {num_unique_individuals}\nCount: {count}")
        sizes.append(count)
    plt.figure(figsize=(10, 7))
    plt.bar(labels, sizes)
    plt.grid()
    plt.xlabel('Cube')
    plt.ylabel('Number of Points')
    plt.title('Top 10 Most Sampled Cubes')
    plt.xticks(rotation=45)
    if save_name is not None:
        plt.savefig(f"{save_name}_bar.png")
        
def make_params_unique(self, individual):
    sorted_individual = sorted(individual, key=lambda x: self.arm_distance(x, self.arm_min_vals))
    normalized_individual = np.array([self.normalize(arm) for arm in sorted_individual])
    
    unique_individual = np.empty(6)

    for param_idx in range(6):
        exponents = np.arange(6) * 2
        unique_individual[param_idx] = np.sum(normalized_individual[:,0] / (10 ** exponents))

    return unique_individual

def get_unique_individuals(population_data, search_space):
    ngens, pop_size, narms, nparams = population_data.shape
    unique_individuals = np.empty((ngens, pop_size, nparams))
    for i, gen in enumerate(population_data):
        for j, individual in enumerate(gen):
            unique_individuals[i, j] = make_params_unique(search_space, individual)
    
    return unique_individuals

