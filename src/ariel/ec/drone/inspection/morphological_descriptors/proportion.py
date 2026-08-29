import numpy as np
import ariel.ec.drone.inspection.utils as u

def calculate_proportionality(individual):
    # Remove rows with all NaNs
    valid_points = individual[~np.isnan(individual).all(axis=1)]
    
    if valid_points.shape[0] == 0:
        return np.nan  # No valid points, return NaN
    
    # Calculate the ranges in each dimension
    x_range = np.ptp(valid_points[:, 0])  # Range in x dimension
    y_range = np.ptp(valid_points[:, 1])  # Range in y dimension
    z_range = np.ptp(valid_points[:, 2])  # Range in z dimension
    
    # Determine the dimensionality
    unique_x = (valid_points[:, 0] == 0.0).all()
    unique_y = (valid_points[:, 1] == 0.0).all()
    unique_z = (valid_points[:, 2] == 0.0).all()
    if (unique_x and unique_y) or (unique_x and unique_z) or (unique_y and unique_z) :
        # 1D shape
        return 1.0
    elif not(unique_x)  and not(unique_y) and unique_z:
        # 2D shape in XY plane
        aspect_ratios = [x_range / y_range, y_range / x_range]
        return min(aspect_ratios)
    elif not(unique_x)  and not(unique_z) and unique_y:
        # 2D shape in XZ plane
        aspect_ratios = [x_range / z_range, z_range / x_range]
        return min(aspect_ratios)
    elif not(unique_y)  and not(unique_z) and unique_x:
        # 2D shape in YZ plane
        aspect_ratios = [y_range / z_range, z_range / y_range]
        return min(aspect_ratios)
    else:
        # 3D shape
        aspect_ratios = [
            min(x_range / y_range, y_range / x_range),
            min(x_range / z_range, z_range / x_range),
            min(y_range / z_range, z_range / y_range)
        ]
        return np.mean(aspect_ratios)

def compute_proportion(individual_or_population):
    # Handle cases where population could be (8, 3), (n, 8, 3), or (m, n, 8, 3)
    if individual_or_population.ndim == 2:
        narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((1,nparams)), axis=0) # Add core as point
        points = np.nan_to_num(u.get_points(individual_or_population))
        return calculate_proportionality(points)

    elif individual_or_population.ndim == 3:
        pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((pop_size,1,nparams)), axis=1) # Add core as point
        points = np.nan_to_num(u.get_points(individual_or_population))
        return np.array([calculate_proportionality(individual) for individual in points])

    elif individual_or_population.ndim == 4:
        ngens, pop_size, narms, nparams = individual_or_population.shape
        individual_or_population = np.append(individual_or_population, np.zeros((ngens, pop_size,1,nparams)), axis=2) # Add core as point
        points = np.nan_to_num(u.get_points(individual_or_population))
        return np.array([[calculate_proportionality(individual) for individual in gen] for gen in points])