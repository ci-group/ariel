import numpy as np
import scipy

import ariel.ec.drone.inspection.utils as u

def compute_area_for_individual(individual, perimeter=False):
    
    points = u.get_points(individual)
    points = points[:,:2]
    
    try:
        convex_hull = scipy.spatial.ConvexHull(points)
    except: # Then points are coplanar
        return 0

    if perimeter:
        area = convex_hull.area
    else:
        area = convex_hull.volume
        
    return area

def compute_area_for_population(population, perimeter=False):
    areas = np.empty(len(population))
    for i, individual in enumerate(population):
        areas[i] = compute_area_for_individual(individual, perimeter=perimeter)
    
    return areas

def compute_area(individual_or_population, perimeter=False):
    num_dims = len(individual_or_population.shape)
    if num_dims == 2:
        narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((1,nparams)), axis=0)
        
        area = compute_area_for_individual(individual_or_population, perimeter=perimeter)
    elif num_dims == 3:
        pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((pop_size,1,nparams)), axis=1)

        area = compute_area_for_population(individual_or_population, perimeter=perimeter)
    elif num_dims == 4:
        ngens, pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((ngens, pop_size,1,nparams)), axis=2)

        ngens, pop_size, narms, nparms = individual_or_population.shape
        area = np.empty((ngens, pop_size))
        for gen in range(ngens):
            area[gen] = compute_area_for_population(individual_or_population[gen], perimeter=perimeter)
    else:
        raise Exception(f"Error area, shape did not match: {individual_or_population.shape}") 

    return area