
import numpy as np

def compute_avr_arm_length(individual_or_population):
    num_dims = len(individual_or_population.shape)
    var = np.nanmean(individual_or_population[...,0], axis=num_dims-2)
    var = np.nan_to_num(var) # if any arms are empty
    return var