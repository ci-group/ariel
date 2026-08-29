import numpy as np
import scipy

import ariel.ec.drone.inspection.utils as u

def compute_volume_for_individual(individual):
    max_narms, _ = individual.shape
    points = u.get_points(individual)
    if max_narms+1 - np.sum((points[:,:] == 0.0).all(axis=1)) < 3:
        return 0
    elif (points[:,2] == 0.0).all():
        points = points[:,:2]
    
    try:
        convex_hull = scipy.spatial.ConvexHull(points)
    except:
        return 0
    coverage = convex_hull.volume

    return coverage

def compute_volume_for_population(population, max_workers=4):
    coverage = np.empty(len(population))
    for i, individual in enumerate(population):
        coverage[i] = compute_volume_for_individual(individual)

    return coverage

def compute_volume(individual_or_population, max_workers=4):
    num_dims = len(individual_or_population.shape)

    if num_dims == 2:
        narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((1,nparams)), axis=0) # Add core as point

        coverage = compute_volume_for_individual(individual_or_population)
    elif num_dims == 3:
        pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((pop_size,1,nparams)), axis=1) # Add core as point

        coverage = compute_volume_for_population(individual_or_population, max_workers=max_workers)
    elif num_dims == 4:
        ngens, pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((ngens, pop_size,1,nparams)), axis=2) # Add core as point

        coverage = np.empty((ngens, pop_size))
        for gen in range(ngens):
            coverage[gen] = compute_volume_for_population(individual_or_population[gen], max_workers=max_workers)
    else:
        raise Exception(f"Error compute_volume, shape did not match: {individual_or_population.shape}") 
    
    return coverage