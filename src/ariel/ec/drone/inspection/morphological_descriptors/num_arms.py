import numpy as np

def compute_num_arms(individual_or_population):
    num_dims = len(individual_or_population.shape)
    num_arms = np.sum(~np.isnan(individual_or_population).all(axis=num_dims-1), axis=num_dims-2)

    return num_arms