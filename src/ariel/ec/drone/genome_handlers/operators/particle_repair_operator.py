

import numpy as np
import itertools
from scipy.spatial.transform import Rotation
from typing import Any
import numpy.typing as npt
import fcl
import gc

from ariel.ec.drone.genome_handlers.conversions.arm_conversions import arms_to_cylinders_polar_angular, cylinders_to_arms_polar_angular

def particle_repair_individual(individual, propeller_radius, inner_boundary_radius, outer_boundary_radius,
                     max_iterations=25, step_size=1.0, propeller_tolerance=0.1, repair_along_fixed_axis=None,
                     arms_to_cylinders=arms_to_cylinders_polar_angular, cylinders_to_arms=cylinders_to_arms_polar_angular):
    """Repair collisions between cylindrical arms by iteratively adjusting positions.
    
    This function performs collision detection and resolution using cylindrical approximations
    of drone arms. It supports both free repair (all collision pairs) and symmetric repair
    (preserving bilateral symmetry).

    Parameters:
    -----------
    individual : array-like
        Genome representation of drone arms
    propeller_radius : float
        Radius of propellers for collision clearance calculation
    inner_boundary_radius : float
        Minimum distance from origin for arm positions
    outer_boundary_radius : float
        Maximum distance from origin for arm positions
    max_iterations : int, default=25
        Maximum number of repair iterations
    step_size : float, default=1.0
        Step size multiplier for collision resolution movements
    propeller_tolerance : float, default=0.1
        Additional clearance tolerance for propellers
    repair_along_fixed_axis : array-like or None, default=None
        Direction vector for symmetric collision resolution. When provided:
        - Assumes bilateral symmetry with arms split into two halves
        - Only resolves collisions between arms from different halves
        - Forces collision resolution along the specified axis direction
        - Preserves symmetry during repair process
    arms_to_cylinders : callable
        Function to convert arm parameters to cylinder representations
    cylinders_to_arms : callable
        Function to convert cylinder representations back to arm parameters
        
    Returns:
    --------
    array-like
        Repaired genome with collision-free arm positions
        
    Notes:
    ------
    For symmetric drones, the recommended repair workflow is:
    1. Apply free repair (repair_along_fixed_axis=None)
    2. Apply symmetry operators to restore bilateral symmetry
    3. Apply constrained repair (repair_along_fixed_axis=[x,y,z]) to maintain symmetry
    """
    # Initialize - filter valid arms and convert to cylinders
    valid_arms = ~np.isnan(individual.copy()).any(axis=-1)
    valid_arm_params = individual.copy()[valid_arms]
    cylinders = arms_to_cylinders(valid_arm_params)
    constraints_violated = are_distance_constraints_violated(cylinders, inner_boundary_radius, outer_boundary_radius)
    if not are_there_cylinder_collisions(cylinders) and not constraints_violated:
        return individual  # No repair needed if no collisions
    num_arms = len(cylinders)
    
    # Precompute constants
    required_clearance = (2 + propeller_tolerance) * propeller_radius
    
    for iteration in range(max_iterations):
        # Apply distance constraints
        constraints_violated = are_distance_constraints_violated(cylinders,
                                                                inner_boundary_radius=inner_boundary_radius, 
                                                                outer_boundary_radius=outer_boundary_radius)
        if constraints_violated:
            cylinders = enforce_distance_constraints(cylinders, inner_boundary_radius, outer_boundary_radius)
        
        # Check and resolve all collisions
        collision_pairs = get_combinations(num_arms, repair_along_fixed_axis)
        collisions_found = False
        for i, j in collision_pairs:
            if are_there_cylinder_collisions([cylinders[i], cylinders[j]]):
                collisions_found = True
                new_cyl_i, new_cyl_j = resolve_collision(cylinders[i], cylinders[j], required_clearance, 
                                                       step_size, repair_along_fixed_axis)

                cylinders[i] = new_cyl_i
                cylinders[j] = new_cyl_j

        # collisions_found = collisions_found or are_there_cylinder_collisions(cylinders)
        
        # Check for convergence
        if not (collisions_found or constraints_violated):
            break
    
    if iteration == max_iterations - 1:
        print(f"Warning: Particle repair reached maximum iterations ({max_iterations}) without convergence.")
    
    # Convert back and update individual
    repaired_arms = cylinders_to_arms(cylinders)
    new_individual = individual.copy()
    new_individual[valid_arms, :-1] = repaired_arms
    
    return new_individual


def get_combinations(num_arms, repair_along_fixed_axis):
    """Generate pairs of arms to check for collisions.
    
    For symmetric repair (when repair_along_fixed_axis is provided):
    - Assumes bilateral symmetry with arms split into two halves
    - Only checks collisions between arms from different halves
    - This preserves symmetry during collision resolution by ensuring
      symmetric arms don't collide with each other within the same half
    """
    if repair_along_fixed_axis is None:
        return itertools.combinations(range(num_arms), 2)
    else:
        list1 = range(0, num_arms // 2)
        list2 = range(num_arms // 2, num_arms)
        return itertools.product(list1, list2)

def resolve_collision(cyl1, cyl2, required_clearance, step_size, fixed_direction):
    """Resolve collision between two cylinders by moving them apart.
    
    Parameters:
    -----------
    cyl1, cyl2 : Cylinder
        Cylinder instances with position attributes
    required_clearance : float
        Minimum distance required between cylinder centers
    step_size : float
        Multiplier for the collision resolution movement
    fixed_direction : array-like or None
        If provided, collision resolution follows this direction instead of
        the natural separation direction. Used for symmetric repair.
        
    Returns:
    --------
    tuple : (new_cyl1, new_cyl2)
        New Cylinder instances with adjusted positions
    """
    p1, p2 = cyl1.position, cyl2.position
    min_dist = np.linalg.norm(p1 - p2)
    
    # Calculate separation direction
    if fixed_direction is None:
        direction = p2 - p1
    else:
        direction = np.array(fixed_direction, dtype=float)
    
    # Handle zero-length direction vectors
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:  # Very small or zero vector
        # Use a default separation direction (e.g., along x-axis)
        direction = np.array([1.0, 0.0, 0.0])
        direction_norm = 1.0
    
    direction = direction / direction_norm
    
    # Calculate required shift
    shift_distance = abs(required_clearance - min_dist) * step_size
    shift_distance = max(shift_distance, 0.01)  # Ensure non-negative shift
    # Limit shift to maximum possible
    if required_clearance/2 < shift_distance:
        shift_distance = required_clearance/2
    
    # Apply bidirectional shift and return new cylinders
    shift_vector = direction * shift_distance

    if fixed_direction is not None:
        axis_to_shift = np.where(np.abs(direction) > 1e-6)[0]
        if cyl2.position[axis_to_shift] < cyl1.position[axis_to_shift]:
            # cyl1 is closer to origin, move it outward
            shift_vector *= -1
    new_pos1 = cyl1.position - shift_vector / 2
    new_pos2 = cyl2.position + shift_vector / 2
    
    return cyl1.with_position(new_pos1), cyl2.with_position(new_pos2)

# Cylinder utilities
# ------------------

def are_distance_constraints_violated(cylinders, inner_boundary_radius, outer_boundary_radius):
    """
    Check if any cylinder is outside the specified inner or outer boundaries.
    
    Parameters:
    - cylinders: list of dicts, each with keys:
        - 'position': numpy array of shape (3,) for cylinder position
        - 'radius': float for cylinder radius
        - 'height': float for cylinder height
        - 'orientation': numpy array of shape (4,) for cylinder orientation as quaternion [w, x, y, z]
    - inner_boundary_radius: float, inner boundary radius
    - outer_boundary_radius: float, outer boundary radius
    
    Returns:
    - True if any cylinder is outside the boundaries, False otherwise
    """
    for cyl in cylinders:
        distance_to_origin = np.linalg.norm(cyl.position)
        
        # Check if cylinder is outside inner boundary
        if distance_to_origin < inner_boundary_radius:
            return True
        
        # Check if cylinder is outside outer boundary
        if distance_to_origin > outer_boundary_radius:
            return True
        
    return False

def enforce_distance_constraints(cylinders, inner_boundary_radius, outer_boundary_radius):
    """
    Enforce minimum and maximum distance constraints from origin.
    
    Parameters:
    - cylinders: list of Cylinder instances
    - inner_boundary_radius: float, minimum distance from origin
    - outer_boundary_radius: float, maximum distance from origin
    
    Returns:
    - List of new Cylinder instances with constrained positions
    """
    new_cylinders = []
    
    for cylinder in cylinders:
        position = cylinder.position
        distance_to_origin = np.linalg.norm(position)
        
        if distance_to_origin < inner_boundary_radius:
            # Too close to origin, move outward
            if np.isclose(distance_to_origin, 0):
                # If the cylinder is at the origin, move it to the inner boundary
                new_position = np.array([inner_boundary_radius, 0, 0])
            else:
                # Move outward along the direction from origin
                direction = position / distance_to_origin  # Normalize
                new_position = direction * inner_boundary_radius
            new_cylinders.append(cylinder.with_position(new_position))
            
        elif distance_to_origin > outer_boundary_radius:
            # Too far from origin, move inward
            direction = position / distance_to_origin  # Normalize
            new_position = direction * outer_boundary_radius
            new_cylinders.append(cylinder.with_position(new_position))
        else:
            # Position is within bounds, keep original
            new_cylinders.append(cylinder)

    return new_cylinders


def correct_quaternion_for_trimesh(quat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """
    Convert a quaternion from X-aligned cylinder (your convention)
    to Z-aligned cylinder (Trimesh convention).
    
    Args:
        quat: Quaternion representing orientation (X-aligned)
        
    Returns:
        Adjusted quaternion for Trimesh's Z-aligned cylinder
    """
    # Rotation to align X axis to Z axis: -90 degrees around Y
    correction = Rotation.from_euler('y', -90, degrees=True)
    
    # Original orientation
    R_orig = Rotation.from_quat(quat)
    
    # Apply correction
    R_corrected = correction * R_orig
    return R_corrected.as_quat()

def are_there_cylinder_collisions(cylinders):
    """
    Check if any pair of cylinders in the list is colliding using python-fcl.

    Parameters:
    - cylinders: list of dicts, each with keys:
        - 'radius': float
        - 'height': float
        - 'transform': 4x4 numpy array (homogeneous transform)

    Returns:
    - True if any pair of cylinders collide, False otherwise
    """
    gc.collect()

    # Create FCL collision objects for each cylinder
    fcl_objects = []
    for i, cyl in enumerate(cylinders):
        radius = cyl.radius
        height = cyl.height
        tf = cyl.transform

        # Decompose transform into rotation (3x3) and translation (3,)
        rotation = tf[:3, :3]
        translation = tf[:3, 3]

        # Create FCL transform and object
        fcl_tf = fcl.Transform(rotation, translation)
        shape = fcl.Cylinder(radius, height)
        obj = fcl.CollisionObject(shape, fcl_tf)

        fcl_objects.append(obj)

    # Check for collisions between all pairs
    for i in range(len(fcl_objects)):
        for j in range(i + 1, len(fcl_objects)):
            request = fcl.CollisionRequest()
            result = fcl.CollisionResult()
            if fcl.collide(fcl_objects[i], fcl_objects[j], request, result) > 0:
                return True

    return False
