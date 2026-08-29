import numpy as np

def compute_total_mass(individual_or_population, core_mass=0.25, motor_mass=0.05, arm_mass_coeff=0.01):   
    num_dims = len(individual_or_population.shape)

    num_arms = np.sum(~np.isnan(individual_or_population).all(axis=num_dims-1), axis=num_dims-2)
    arm_lengths = individual_or_population[...,0]
    mass = core_mass + num_arms*motor_mass + arm_mass_coeff*np.nansum(arm_lengths, axis=num_dims-2)

    return mass