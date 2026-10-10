import numpy as np

import ariel.ec.drone.inspection.utils as u

def reflect_points(points, plane):
    # Unpack the plane coefficients
    a, b, c, d = plane
    
    # Convert points to a NumPy array if not already
    points = np.array(points)
    
    # Calculate the numerator for the scale factor for all points at once
    scale_numerator = 2 * (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)
    
    # Calculate the denominator for the scale factor (it's a constant)
    scale_denominator = a**2 + b**2 + c**2
    
    # Compute the scale factor for all points
    scale = scale_numerator / scale_denominator
    
    # Calculate the reflected points using broadcasting
    reflected_points = points - scale[:, np.newaxis] * np.array([a, b, c])
    
    # Round the result to 15 decimal places and return
    return reflected_points.round(8)

def sum_closest_distance(list1, list2):
    # Convert lists to numpy arrays
    array1 = np.array(list1)
    array2 = np.array(list2)
    
    # Calculate the squared differences and sum them along the last axis
    distances = np.sqrt(((array1[:, np.newaxis, :] - array2[np.newaxis, :, :]) ** 2).sum(axis=2))
    
    # Find the minimum distance for each point in array1
    min_distances = np.min(distances, axis=1)
    
    # Calculate the mean of these minimum distances
    mean_distance = np.sum(min_distances)
    
    return mean_distance

def find_best_reflection_plane(points, coarse_step=20, fine_step=5, fixed_plane=None):
    best_score = float('inf')
    best_plane = None
    centroid = np.mean(points, axis=0)
    if fixed_plane is not None:
        # Use the fixed plane directly
        reflected_points = reflect_points(points, fixed_plane)
        best_score = sum_closest_distance(points, reflected_points)
        return fixed_plane, best_score

    def search_planes(theta_range, phi_range):
        nonlocal best_score, best_plane
        for theta in theta_range:
            for phi in phi_range:
                a = np.sin(theta) * np.cos(phi)
                b = np.sin(theta) * np.sin(phi)
                c = np.cos(theta)
                d = -(a * centroid[0] + b * centroid[1] + c * centroid[2])
                plane = np.array((a, b, c, d)).round(15)
                reflected_points = reflect_points(points, plane)
                if (reflected_points == points.astype(np.float64)).all():
                    break
                score = sum_closest_distance(points, reflected_points)

                if score < best_score:
                    best_score = score
                    best_plane = plane
                    if score < 1e-15: # perfect symmetry, don't continue
                        return

    # Coarse search
    coarse_theta_range = np.linspace(0, np.pi, coarse_step)
    coarse_phi_range = np.linspace(0, 2 * np.pi, coarse_step)
    search_planes(coarse_theta_range, coarse_phi_range)

    # Fine search around the best coarse result
    if not isinstance(best_plane, np.ndarray):
        print(f"Warning, symmetry not found for shape: {points}, coarse_step={coarse_step}, fine_step={fine_step}")
        
    best_theta = np.arccos(best_plane[2])
    best_phi = np.arctan2(best_plane[1], best_plane[0])
    fine_theta_range = np.linspace(best_theta - np.pi / coarse_step, best_theta + np.pi / coarse_step, fine_step)
    fine_phi_range = np.linspace(best_phi - 2 * np.pi / coarse_step, best_phi + 2 * np.pi / coarse_step, fine_step)
    search_planes(fine_theta_range, fine_phi_range)
    
    return best_plane, best_score

def compute_symmetry_for_individual(individual, fixed_plane=None):
    num_dims = len(individual.shape)
    num_arms = np.sum(~np.isnan(individual).all(axis=num_dims-1), axis=num_dims-2)
    if num_arms == 0:
        return 0
    points = u.get_points(individual)
    points = np.append(points, [[0,0,0]], axis=0)
    _, best_score = find_best_reflection_plane(points, coarse_step=20, fine_step=5, fixed_plane=fixed_plane)
    if best_score < 1e-6:
        return 0
    return best_score

def compute_symmetry_for_population(population, fixed_plane=None):
    best_score = np.empty(len(population))

    for ind, individual in enumerate(population):
        best_score[ind] = compute_symmetry_for_individual(individual, fixed_plane=fixed_plane)
    
    return best_score

def compute_symmetry(individual_or_population, max_workers=4, fixed_plane=None):
    num_dims = len(individual_or_population.shape)

    if num_dims == 2:
        best_score = compute_symmetry_for_individual(individual_or_population, fixed_plane=fixed_plane)
    elif num_dims == 3:
        best_score = compute_symmetry_for_population(individual_or_population, fixed_plane=fixed_plane)
    elif num_dims == 4:
        ngens, pop_size, narms, nparms = individual_or_population.shape
        best_score = np.empty((ngens, pop_size))
        for gen in range(ngens):
            best_score[gen] = compute_symmetry_for_population(individual_or_population[gen], fixed_plane=fixed_plane)
    else:
        raise Exception(f"Error symmetry, shape did not match: {individual_or_population.shape}") 
    
    return best_score