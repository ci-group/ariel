import numpy as np
from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import orientation_to_unit_vector
"""
Computes how the thrust vectors of multirotor aligns (is parallel to) each other.
If the output is equal to the number of the number of thrust vectors, then they are all parallel to each other. 
"""
def unit_vector_to_yaw_pitch(unit_vector):
    """
    Converts a unit vector to yaw (rotation around z-axis) and pitch (rotation around y-axis).
    
    Args:
        unit_vector (np.ndarray): A 3D unit vector [x, y, z].
    
    Returns:
        tuple: (yaw, pitch) in radians.
    """
    x, y, z = unit_vector
    yaw = np.arctan2(y, x)  # Yaw: angle in the x-y plane
    pitch = np.arcsin(z)    # Pitch: angle from the x-y plane

    return yaw, pitch

def thrust_alignment(individual_or_population, normalize=False):
    if len(individual_or_population.shape) == 4:
        alignment_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                alignment = thrust_alignment(individual, normalize)
                gen_list.append(alignment)
            alignment_list.append(gen_list)
        return np.array(alignment_list)
    elif len(individual_or_population.shape) == 3:
        alignment_list = []
        for individual in individual_or_population:
            alignment = thrust_alignment(individual, normalize)
            alignment_list.append(alignment)
        return np.array(alignment_list)
    else:
        thrust_vectors = [orientation_to_unit_vector(0., arm[4], arm[3]) for arm in individual_or_population]
        thrust_vector_sum = np.sum(thrust_vectors, axis=0)
        alignment = np.linalg.norm(thrust_vector_sum)
        if normalize:
            num_arm = len(individual_or_population)
            alignment = alignment / num_arm
    return alignment

def thrust_unitvector_yaw(individual_or_population):
    if len(individual_or_population.shape) == 4:
        yaw_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                yaw = thrust_unitvector_yaw(individual)
                gen_list.append(yaw)
            yaw_list.append(gen_list)

        return np.array(yaw_list)
    elif len(individual_or_population.shape) == 3:
        yaw_list = []
        for individual in individual_or_population:
            yaw = thrust_unitvector_yaw(individual)
            yaw_list.append(yaw)
        return np.array(yaw_list)
    else:
        thrust_vectors = [orientation_to_unit_vector(0., arm[4], arm[3]) for arm in individual_or_population]
        thrust_vector_norm = np.sum(thrust_vectors, axis=0) / len(thrust_vectors)

        yaw, pitch = unit_vector_to_yaw_pitch(thrust_vector_norm)
    return yaw

def thrust_unitvector_pitch(individual_or_population):

    if len(individual_or_population.shape) == 4:
        pitch_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                pitch = thrust_unitvector_pitch(individual)
                gen_list.append(pitch)
            pitch_list.append(gen_list)
        return np.array(pitch_list)
    elif len(individual_or_population.shape) == 3:
        pitch_list = []
        for individual in individual_or_population:
            pitch = thrust_unitvector_pitch(individual)
            pitch_list.append(pitch)
        return np.array(pitch_list)
    else:
        thrust_vectors = [orientation_to_unit_vector(0., arm[4], arm[3]) for arm in individual_or_population]
        thrust_vector_norm = np.sum(thrust_vectors, axis=0) / len(thrust_vectors)

        yaw, pitch = unit_vector_to_yaw_pitch(thrust_vector_norm)
    
    return pitch