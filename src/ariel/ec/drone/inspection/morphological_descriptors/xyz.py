import numpy as np

from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import euler_to_rotation_matrix


def get_xyzs(individual):
    default_rotation = np.array([0, 0, -1])# Define the negative z-axis vector

    xs = np.zeros(len(individual))
    ys = np.zeros(len(individual))
    zs = np.zeros(len(individual))

    for i, (arm_length, arm_yaw, arm_pitch, motor_pitch, motor_yaw, direction) in enumerate(individual):
        # Get the rotation matrix from Euler angles
        R = euler_to_rotation_matrix(0, motor_pitch, motor_yaw)
        transform_from_ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        R = transform_from_ENU_to_NED @ R
        # Transform the negative z-axis using the rotation matrix
        transformed_vector = np.dot(R, default_rotation)
        # Normalize the resulting vector
        transformed_vector = transformed_vector / np.linalg.norm(transformed_vector)
        transformed_vector = arm_length * transformed_vector

        xs[i] = transformed_vector[0]
        ys[i] = transformed_vector[1]
        zs[i] = transformed_vector[2]

    return xs, ys, zs

def avr_x(individual_or_population):
    if len(individual_or_population.shape) == 4:
        xs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                xs = avr_x(individual)
                gen_list.append(xs)
            xs_list.append(gen_list)
        return np.array(xs_list)
    elif len(individual_or_population.shape) == 3:
        xs_list = []
        for individual in individual_or_population:
            xs = avr_x(individual)
            xs_list.append(xs)
        return np.array(xs_list)
    else:
        xs, _, _ = get_xyzs(individual_or_population)
        return np.mean(xs)

def avr_y(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ys_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                ys = avr_y(individual)
                gen_list.append(ys)
            ys_list.append(gen_list)
        return np.array(ys_list)
    elif len(individual_or_population.shape) == 3:
        ys_list = []
        for individual in individual_or_population:
            ys = avr_y(individual)
            ys_list.append(ys)
        return np.array(ys_list)
    else:
        _, ys, _ = get_xyzs(individual_or_population)
        return np.mean(ys)
    
def avr_z(individual_or_population):
    if len(individual_or_population.shape) == 4:
        zs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                zs = avr_z(individual)
                gen_list.append(zs)
            zs_list.append(gen_list)
        return np.array(zs_list)
    elif len(individual_or_population.shape) == 3:
        zs_list = []
        for individual in individual_or_population:
            zs = avr_z(individual)
            zs_list.append(zs)
        return np.array(zs_list)
    else:
        _, _, zs = get_xyzs(individual_or_population)
        return np.mean(zs)
    
def var_x(individual_or_population):
    if len(individual_or_population.shape) == 4:
        xs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                xs = avr_x(individual)
                gen_list.append(xs)
            xs_list.append(gen_list)
        return np.array(xs_list)
    elif len(individual_or_population.shape) == 3:
        xs_list = []
        for individual in individual_or_population:
            xs = avr_x(individual)
            xs_list.append(xs)
        return np.array(xs_list)
    else:
        xs, _, _ = get_xyzs(individual_or_population)
        return np.var(xs)

def var_y(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ys_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                ys = avr_y(individual)
                gen_list.append(ys)
            ys_list.append(gen_list)
        return np.array(ys_list)
    elif len(individual_or_population.shape) == 3:
        ys_list = []
        for individual in individual_or_population:
            ys = avr_y(individual)
            ys_list.append(ys)
        return np.array(ys_list)
    else:
        _, ys, _ = get_xyzs(individual_or_population)
        return np.var(ys)
    
def var_z(individual_or_population):
    if len(individual_or_population.shape) == 4:
        zs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                zs = avr_z(individual)
                gen_list.append(zs)
            zs_list.append(gen_list)
        return np.array(zs_list)
    elif len(individual_or_population.shape) == 3:
        zs_list = []
        for individual in individual_or_population:
            zs = avr_z(individual)
            zs_list.append(zs)
        return np.array(zs_list)
    else:
        _, _, zs = get_xyzs(individual_or_population)
        return np.var(zs)

def x_size(individual_or_population):
    if len(individual_or_population.shape) == 4:
        xs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                xs = avr_x(individual)
                gen_list.append(xs)
            xs_list.append(gen_list)
        return np.array(xs_list)
    elif len(individual_or_population.shape) == 3:
        xs_list = []
        for individual in individual_or_population:
            xs = avr_x(individual)
            xs_list.append(xs)
        return np.array(xs_list)
    else:
        xs, _, _ = get_xyzs(individual_or_population)
        return np.max(xs) - np.min(xs)
    
def y_size(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ys_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                ys = avr_y(individual)
                gen_list.append(ys)
            ys_list.append(gen_list)
        return np.array(ys_list)
    elif len(individual_or_population.shape) == 3:
        ys_list = []
        for individual in individual_or_population:
            ys = avr_y(individual)
            ys_list.append(ys)
        return np.array(ys_list)
    else:
        _, ys, _ = get_xyzs(individual_or_population)
        return np.max(ys) - np.min(ys)

def z_size(individual_or_population):
    if len(individual_or_population.shape) == 4:
        zs_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                zs = avr_z(individual)
                gen_list.append(zs)
            zs_list.append(gen_list)
        return np.array(zs_list)
    elif len(individual_or_population.shape) == 3:
        zs_list = []
        for individual in individual_or_population:
            zs = avr_z(individual)
            zs_list.append(zs)
        return np.array(zs_list)
    else:
        _, _, zs = get_xyzs(individual_or_population)
        return np.max(zs) - np.min(zs)

def avr_distance_between_points(individual_or_population):
    if len(individual_or_population.shape) == 4:
        distances_list = []
        for gen in individual_or_population:
            gen_list = []
            for individual in gen:
                distances = avr_distance_between_points(individual)
                gen_list.append(distances)
            distances_list.append(gen_list)
        return np.array(distances_list)
    elif len(individual_or_population.shape) == 3:
        distances_list = []
        for individual in individual_or_population:
            distances = avr_distance_between_points(individual)
            distances_list.append(distances)
        return np.array(distances_list)
    else:
        xs, ys, zs = get_xyzs(individual_or_population)
        distances = np.sqrt(np.sum(np.square(xs)) + np.sum(np.square(ys)) + np.sum(np.square(zs)))
        return distances

    
    
    
    
    
