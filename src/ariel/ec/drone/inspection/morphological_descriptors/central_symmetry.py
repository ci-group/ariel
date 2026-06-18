import numpy as np

def compute_symmetry(data):
    """
    Calculate the symmetry score for each individual in the given data.
    
    Parameters:
    data (numpy.ndarray): Input array of shape (generations, population_size, max_arms, 6), 
                          (population_size, max_arms, 6), or (max_arms, 6)
    
    Returns:
    numpy.ndarray: Symmetry scores for each individual
    """
    # Determine the shape of the input data
    if data.ndim == 4:
        generations, population_size, max_arms, _ = data.shape
        has_generations = True
    elif data.ndim == 3:
        population_size, max_arms, _ = data.shape
        generations = 1
        has_generations = False
    elif data.ndim == 2:
        max_arms, _ = data.shape
        population_size = 1
        generations = 1
        has_generations = False
    else:
        raise ValueError("Input data must have shape (generations, population_size, max_arms, 6), (population_size, max_arms, 6), or (max_arms, 6)")
    
    # Extract the relevant parameters (arm length, arm yaw, arm pitch)
    arm_lengths = data[..., 0]
    arm_yaws = data[..., 1]
    arm_pitches = data[..., 2]
    
    # Convert arms to 3D points
    x = arm_lengths * np.cos(arm_yaws) * np.cos(arm_pitches)
    y = arm_lengths * np.sin(arm_yaws) * np.cos(arm_pitches)
    z = arm_lengths * np.sin(arm_pitches)
    points = np.stack((x, y, z), axis=-1)
    
    # Mirror the arms by negating the 3D coordinates
    mirrored_points = -points
    
    # Calculate the symmetry score for each individual
    symmetry_scores = np.zeros((generations, population_size))
    
    for gen in range(generations):
        for ind in range(population_size):
            if has_generations:
                individual_points = points[gen, ind]
                mirrored_individual_points = mirrored_points[gen, ind]
            else:
                individual_points = points[ind] if population_size > 1 else points
                mirrored_individual_points = mirrored_points[ind] if population_size > 1 else mirrored_points
            
            # Filter out NaN arms
            valid_arms = ~np.isnan(individual_points).any(axis=1)
            valid_mirrored_arms = ~np.isnan(mirrored_individual_points).any(axis=1)
            individual_points = individual_points[valid_arms]
            mirrored_individual_points = mirrored_individual_points[valid_mirrored_arms]
            
            if individual_points.size == 0 or mirrored_individual_points.size == 0:
                symmetry_scores[gen, ind] = np.nan
                continue
            
            # Calculate the distance between each mirrored point and all actual points
            distances = np.linalg.norm(individual_points[:, np.newaxis, :] - mirrored_individual_points[np.newaxis, :, :], axis=2)
            
            # Find the minimum distance for each mirrored point
            min_distances = np.min(distances, axis=1)
            
            # Average the minimum distances to get the symmetry score
            symmetry_scores[gen, ind] = np.mean(min_distances)
    
    # If there were no generations, return a 1D array
    if not has_generations:
        return symmetry_scores[0] if population_size > 1 else symmetry_scores[0, 0]
    
    return symmetry_scores

# # Example usage:
# data_with_generations = np.random.rand(10, 5, 4, 6)  # Example data with shape (generations, population_size, max_arms, 6)
# data_without_generations = np.random.rand(5, 4, 6)  # Example data with shape (population_size, max_arms, 6)
# data_single_individual = np.random.rand(4, 6)  # Example data with shape (max_arms, 6)
# scores_with_generations = calculate_symmetry_score(data_with_generations)
# scores_without_generations = calculate_symmetry_score(data_without_generations)
# score_single_individual = calculate_symmetry_score(data_single_individual)
# print(scores_with_generations)
# print(scores_without_generations)
# print(score_single_individual)