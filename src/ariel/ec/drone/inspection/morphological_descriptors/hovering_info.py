import numpy as np
import copy

from dronehover.bodies.custom_bodies import Custombody
from dronehover.optimization import Hover
from ariel.ec.drone.genome_handlers.mounting_points import (
    generate_disc_mounting_points,
    assign_nearest_mounting_point
)

import ariel.ec.drone.inspection.utils as u
from ariel.ec.drone.inspection.morphological_descriptors.mass import compute_total_mass
from ariel.ec.drone.inspection.morphological_descriptors.centre_of_gravity import centre_of_gravity
from ariel.ec.drone.inspection.morphological_descriptors.inertia import inertia

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    The rotation matrix follows the z-y-x convention (yaw-pitch-roll).
    """
    # Compute individual rotation matrices
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0,             1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_x = np.array([
        [1, 0,            0           ],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    # Combined rotation matrix
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def orientation_to_unit_vector(roll, pitch, yaw):
    """
    Converts roll, pitch, and yaw to a unit vector where the negative z-axis is pointing up.
    """
    # Define the negative z-axis vector
    default_rotation = np.array([0, 0, -1])
    
    # Get the rotation matrix from Euler angles
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    transform_from_ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
    R = transform_from_ENU_to_NED @ R
    # Transform the negative z-axis using the rotation matrix
    transformed_vector = np.dot(R, default_rotation)
    
    # Normalize the resulting vector
    unit_vector = transformed_vector / np.linalg.norm(transformed_vector)
    
    return unit_vector

def get_sim(individual, motor_template = {"propsize": 2}):
    # remove rows with nan values
    individual = individual[~np.isnan(individual).any(axis=1)]

    mass = compute_total_mass(individual)
    cg = centre_of_gravity(individual)
    Ix, Iy, Iz, Ixy, Ixz, Iyz = inertia(individual)


    props = []
    propeller_positions = []
    mypypd = individual[:,:6] # [mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, dir]
    for mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, dir in mypypd:
        global_x,global_y,global_z = u.convert_to_cartesian(mag, arm_yaw, arm_pitch)
        x,y,z = u.ENU_to_NED(global_x,global_y,global_z)

        propeller_positions.append([float(x), float(y), float(z)])

        tmp = copy.deepcopy(motor_template)
        tmp.update({"loc": [float(x),float(y),float(z)]})
        if dir == 0:
            d = "ccw"
        else:
            d = "cw"

        ENU_unit_vector = orientation_to_unit_vector(0,mot_pitch,mot_yaw)
        unit_vector = ENU_unit_vector
        tmp.update({"dir": [float(unit_vector[0]),float(unit_vector[1]),float(unit_vector[2]), d]})

        props.append(tmp)

    # Generate 8 mounting points on 60mm diameter disc
    disc_mounting_points = generate_disc_mounting_points(num_points=8, diameter=0.060)

    # Assign each propeller to nearest mounting point
    mounting_points = assign_nearest_mounting_point(propeller_positions, disc_mounting_points)

    # Create drone with mounting points
    # Note: We override mass/inertia calculations since we use our own descriptors
    drone = Custombody(props, mountpoints=mounting_points, mass=mass, cg=cg, Ix=Ix, Iy=Iy, Iz=Iz, Ixy=Ixy, Ixz=Ixz, Iyz=Iyz)

    # Define hovering optimizer for drone
    try:
        sim = Hover(drone)
    except:
        sim = None

    return sim

def drone_info(individual):

    sim = get_sim(individual)
    if sim is None:
        return 0, np.nan, np.nan, np.nan, np.nan
    sim.compute_hover(verbose=False)
    sim.static(verbose=False,tol=1e-5)
    spinning_success = False
    
    if sim.static_success == False:
        sim.spinning(verbose=False,tol=1e-5)
        spinning_success = sim.spinning_success

    success = sim.static_success or spinning_success
    if sim.static_success:
        hover_type = 2
    elif spinning_success:
        hover_type = 1
    else:
        hover_type = 0

    if not(success):
        max_thrust_to_weight = np.nan
        input_cost = np.nan
        rank_controlability = np.nan
        controlability = np.nan
    else:
        max_thrust_to_weight = np.linalg.norm(sim.f_max)/9.81
        input_cost = sim.input_cost
        rank_controlability = np.linalg.matrix_rank(sim.Bm)
        #controlability = np.linalg.eig(sim.Bm @ sim.Bm.T).eigenvalues[-1]
        eigenvalues, eigenvectors = np.linalg.eig(sim.Bm @ sim.Bm.T)
        controlability = min(eigenvalues)
    
    return hover_type, max_thrust_to_weight, input_cost, rank_controlability, controlability

def max_thrust_to_weight(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ngens, pop_size, max_num_arms, nparams = individual_or_population.shape
        result = np.zeros((ngens, pop_size))
        for g in range(ngens):
            for i in range(pop_size):
                hover_type, result[g,i], input_cost, rank_controlability, controlability = drone_info(individual_or_population[g,i])
        return result
    if len(individual_or_population.shape) == 3:
        hover_type, result, input_cost, rank_controlability, controlability = drone_info(individual_or_population)
        return result
    if len(individual_or_population.shape) == 2:
        hover_type, result, input_cost, rank_controlability, controlability = drone_info(individual_or_population)
        return result
    return np.nan

def input_cost(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ngens, pop_size, max_num_arms, nparams = individual_or_population.shape
        input_cost = np.zeros((ngens, pop_size))
        for g in range(ngens):
            for i in range(pop_size):
                hover_type, max_thrust_to_weight, input_cost[g,i], rank_controlability, controlability = drone_info(individual_or_population[g,i])
    elif len(individual_or_population.shape) == 3:
        input_cost = np.zeros((individual_or_population.shape[0]))
        for i in range(individual_or_population.shape[0]):
            hover_type, max_thrust_to_weight, input_cost[i], rank_controlability, controlability = drone_info(individual_or_population[i])
    else:
        hover_type, max_thrust_to_weight, input_cost, rank_controlability, controlability = drone_info(individual_or_population)
    
    return input_cost

def rank_controlability(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ngens, pop_size, max_num_arms, nparams = individual_or_population.shape
        rank_controlability = np.zeros((ngens, pop_size))
        for g in range(ngens):
            for i in range(pop_size):
                hover_type, max_thrust_to_weight, input_cost, rank_controlability[g,i], controlability = drone_info(individual_or_population[g,i])
    elif len(individual_or_population.shape) == 3:
        rank_controlability = np.zeros((individual_or_population.shape[0]))
        for i in range(individual_or_population.shape[0]):
            hover_type, max_thrust_to_weight, input_cost, rank_controlability[i], controlability = drone_info(individual_or_population[i])

    elif len(individual_or_population.shape) == 2:
        hover_type, max_thrust_to_weight, input_cost, rank_controlability, controlability = drone_info(individual_or_population)
    
    return rank_controlability

def controlability(individual_or_population):
    if len(individual_or_population.shape) == 4:
        ngens, pop_size, max_num_arms, nparams = individual_or_population.shape
        result = np.zeros((ngens, pop_size))
        for g in range(ngens):
            for i in range(pop_size):
                hover_type, mtw, ic, rc, result[g,i] = drone_info(individual_or_population[g,i])
        return result
    if len(individual_or_population.shape) == 3:
        result = np.zeros((individual_or_population.shape[0]))
        for i in range(individual_or_population.shape[0]):
            hover_type, mtw, ic, rc, result[i] = drone_info(individual_or_population[i])
        return result
    if len(individual_or_population.shape) == 2:
        hover_type, mtw, ic, rc, result = drone_info(individual_or_population)
        return result
    return np.nan

def compute_hovering_info(individual_or_population):
    
    if len(individual_or_population.shape) == 4:
        ngens, pop_size, max_num_arms, nparams = individual_or_population.shape
        hovering_info = np.zeros((ngens, pop_size, 5))
        for g in range(ngens):
            for i in range(pop_size):
                success, max_thrust_to_weight, input_cost, rank_controlability, controlability = drone_info(individual_or_population[g,i])
                hovering_info[g,i] = [success, max_thrust_to_weight, input_cost, rank_controlability, controlability]

    if len(individual_or_population.shape) == 3:
        pop_size, max_num_arms, nparams = individual_or_population.shape
        hovering_info = np.zeros((pop_size, 5))
        for i in range(pop_size):
            success, max_thrust_to_weight, input_cost, rank_controlability, controlability = drone_info(individual_or_population[i])
            hovering_info[i] = [success, max_thrust_to_weight, input_cost, rank_controlability, controlability]
    else:
        hovering_info = np.zeros((5))
        success, max_thrust_to_weight, input_cost, rank_controlability, controlability = drone_info(individual_or_population)
        hovering_info = [success, max_thrust_to_weight, input_cost, rank_controlability, controlability]
    return hovering_info
