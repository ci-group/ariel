import numpy as np
import itertools

def print_statistics(population):
    """Print fitness statistics for a population. Requires a population of genotypes with the attribute .fitness.values present!

    Args:
        population (list): Population of individuals with fitness values
    """
    # Extract fitness values
    pop_fitness = [ind.fitness.values[0] for ind in population]
    
    # Calculate statistics
    avg_fitness = np.mean(pop_fitness)
    std_fitness = np.std(pop_fitness)
    best_fitness = np.max(pop_fitness)
    
       # Cast individuals as numpy
    arrays = [np.array(x, dtype = float) for x in pop_fitness]
    
    # Calculate distances
    dists = [np.linalg.norm(a - b) for a, b, in itertools.combinations(arrays, 2)]
    av_dists, min_dists, max_dists = float(np.mean(dists)), float(np.min(dists)), float(np.max(dists))
    
    # Print
    print(f'{"Av. fitness":<12} {"std":<12} {"Best fitness":<12} {"Av. distance":<12} {"Min. distance":<12} {"Max. distance":<12}')
    print(f"{avg_fitness:<12.2f} {std_fitness:<12.2f} {best_fitness:<12.2f} {av_dists:<12.2f} {min_dists:<12.2f} {max_dists:<12.2f}")
