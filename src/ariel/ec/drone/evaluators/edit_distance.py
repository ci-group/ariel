"""Evaluator functions for edit distance calculations."""

import numpy as np
import scipy

from ariel.ec.drone.genome_handlers.base import GenomeHandler

def normalize(individual_or_population, min_vals, max_vals):
    """
    Normalize the parameters of each motor in each individual in the population.

    Args:
    - individual_or_population (numpy.ndarray): The population array of shape (num_individuals, num_motors, num_parameters),
      or individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.

    Returns:
    - numpy.ndarray: The normalized population or individual array.
    """
    ranges = max_vals - min_vals
    return (individual_or_population - min_vals) / ranges


def compute_euclidean_distance(ind1, ind2, min_vals, max_vals):
    """
    Compute the Euclidean distance between two individuals.

    Args:
    - ind1 (numpy.ndarray): First individual array of shape (num_motors, num_parameters).
    - ind2 (numpy.ndarray): Second individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.

    Returns:
    - float: The minimum matching cost based on Euclidean distance.
    """
    ind1 = normalize(ind1, min_vals, max_vals)
    ind2 = normalize(ind2, min_vals, max_vals)

    valid_mask_ind1 = ~np.isnan(ind1).all(axis=1)
    valid_mask_ind2 = ~np.isnan(ind2).all(axis=1)

    valid_ind1 = ind1[valid_mask_ind1]
    valid_ind2 = ind2[valid_mask_ind2]

    diff = valid_ind1[:, np.newaxis, :] - valid_ind2[np.newaxis, :, :]
    cost_matrix = np.sqrt(np.nansum(diff ** 2, axis=2))

    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    return cost_matrix[row_ind, col_ind].sum()


def compute_euclidean_distance_for_pop(population, target, min_vals, max_vals):
    """
    Compute the Euclidean distance between a target individual and a population.

    Args:
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - target (numpy.ndarray): Target individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.

    Returns:
    - numpy.ndarray: Array of distances for each individual in the population.
    """
    pop_size = len(population)
    dists = np.empty(pop_size)
    for i in range(pop_size):
        dists[i] = compute_euclidean_distance(target, population[i], min_vals, max_vals)
    return dists


def compute_arm_cost(ind1, ind2):
    """
    Compute the arm cost difference between two individuals.

    Args:
    - ind1 (numpy.ndarray): First individual array of shape (num_motors, num_parameters).
    - ind2 (numpy.ndarray): Second individual array of shape (num_motors, num_parameters).

    Returns:
    - int: The absolute difference in the number of valid arms.
    """
    ind1_num_arms = np.sum(~np.isnan(ind1).all(axis=1))
    ind2_num_arms = np.sum(~np.isnan(ind2).all(axis=1))
    return np.abs(ind1_num_arms - ind2_num_arms)


def compute_arm_cost_for_population(population, target):
    """
    Compute the arm cost difference between a target individual and a population.

    Args:
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - target (numpy.ndarray): Target individual array of shape (num_motors, num_parameters).

    Returns:
    - numpy.ndarray: Array of arm cost differences for each individual in the population.
    """
    target_num_arms = np.sum(~np.isnan(target).all(axis=1))
    num_arms_per_individual = np.sum(~np.isnan(population).all(axis=2), axis=1)
    return np.abs(num_arms_per_individual - target_num_arms)



def compute_edit_distance(ind1, ind2, min_vals, max_vals, ins_del_cost=1.0):
    """
    Compute the edit distance between two individuals.

    Args:
    - ind1 (numpy.ndarray): First individual array of shape (num_motors, num_parameters).
    - ind2 (numpy.ndarray): Second individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.
    - ins_del_cost (float): Cost of insertion or deletion.

    Returns:
    - float: The edit distance.
    """
    distance_cost = compute_euclidean_distance(ind1, ind2, min_vals, max_vals)
    arm_cost = compute_arm_cost(ind1, ind2) * ins_del_cost
    return distance_cost + arm_cost



def compute_edit_distances_for_population(population, target, min_vals, max_vals, ins_del_cost=1.0):
    """
    Compute the edit distances between a target individual and a population.

    Args:
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - target (numpy.ndarray): Target individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.
    - ins_del_cost (float): Cost of insertion or deletion.

    Returns:
    - numpy.ndarray: Array of edit distances for each individual in the population.
    """
    distance_cost = compute_euclidean_distance_for_pop(population, target, min_vals, max_vals)
    arm_cost = compute_arm_cost_for_population(population, target) * ins_del_cost
    return distance_cost + arm_cost

def compute_population_edit_distances(population, min_vals, max_vals, ins_del_cost=1.0):
    """
    Compute the average edit distances for each individual in a population.

    Args:
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.
    - ins_del_cost (float): Cost of insertion or deletion.

    Returns:
    - numpy.ndarray: Array of average edit distances for each individual in the population.
    """
    pop_size = len(population)
    dists = np.empty(pop_size)

    for i, ind1 in enumerate(population):
        ind1_dists = np.empty(pop_size)
        for j, ind2 in enumerate(population):
            ind1_dists[j] = compute_edit_distance(ind1, ind2, min_vals, max_vals, ins_del_cost)
        dists[i] = np.mean(ind1_dists)

    return dists


def compute_individual_population_edit_distance(individual, population, min_vals, max_vals):
    """
    Compute the average edit distance between an individual and a population.

    Args:
    - individual (numpy.ndarray): Individual array of shape (num_motors, num_parameters).
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.

    Returns:
    - float: The average edit distance.
    """
    pop_size = len(population)
    distance_cost = np.empty(pop_size)
    for i in range(pop_size):
        distance_cost[i] = compute_euclidean_distance(individual, population[i], min_vals, max_vals)

    target_num_arms = np.sum(~np.isnan(individual).all(axis=1))
    num_arms_per_individual = np.sum(~np.isnan(population).all(axis=2), axis=1)
    arm_cost = np.abs(num_arms_per_individual - target_num_arms)

    return np.mean(distance_cost + arm_cost)


def evaluate_individual(individual : GenomeHandler, log_dir, target, min_vals, max_vals):
    """
    Evaluate the fitness of an individual based on its edit distance to the target.

    Args:
    - individual (numpy.ndarray): Individual array of shape (num_motors, num_parameters).
    - target (numpy.ndarray): Target individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.

    Returns:
    - float: The fitness value (negative edit distance).
    """
    return -compute_edit_distance(target, individual.body, min_vals, max_vals)

def evaluate_population(population, target, min_vals, max_vals, ins_del_cost=1.0):
    """
    Evaluate the fitness of a population based on their edit distances to the target.

    Args:
    - population (numpy.ndarray): Population array of shape (num_individuals, num_motors, num_parameters).
    - target (numpy.ndarray): Target individual array of shape (num_motors, num_parameters).
    - min_vals (numpy.ndarray): Minimum values for normalization.
    - max_vals (numpy.ndarray): Maximum values for normalization.
    - ins_del_cost (float): Cost of insertion or deletion.

    Returns:
    - numpy.ndarray: Array of fitness values (negative edit distances) for the population.
    """
    return -compute_edit_distances_for_population(population, target, min_vals, max_vals, ins_del_cost)