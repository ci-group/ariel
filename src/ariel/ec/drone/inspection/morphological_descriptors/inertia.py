"""
Inertia Calculation using Drone-Hover Physics

Computes realistic inertia matrix components using parallel axis theorem
and beam contributions. Integrates with drone-hover package for physics-based
inertia calculation with automatic mounting point assignment.
"""

import numpy as np
from numpy.linalg import norm
from dronehover.bodies.custom_bodies import Custombody
from ariel.ec.drone.genome_handlers.mounting_points import (
    generate_disc_mounting_points,
    assign_nearest_mounting_point
)
import ariel.ec.drone.inspection.utils as u


def inertia(individual):
    """
    Compute inertia matrix components for an individual drone.

    Uses drone-hover's physics-based inertia calculation with automatic
    mounting point assignment from an 8-point disc (60mm diameter).

    Args:
        individual (np.ndarray): Genome array with shape (n_arms, 6)
            Columns: [magnitude, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction]

    Returns:
        tuple: (Ix, Iy, Iz, Ixy, Ixz, Iyz) inertia components in kg*m^2
    """
    # Remove rows with NaN values
    individual = individual[~np.isnan(individual).any(axis=1)]

    if len(individual) == 0:
        # No valid arms, return default minimal inertia
        return [0.01, 0.01, 0.01, 0, 0, 0]

    # Convert genome to propeller locations
    props = []
    propeller_positions = []

    for mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction in individual:
        # Convert spherical to Cartesian (ENU frame)
        global_x, global_y, global_z = u.convert_to_cartesian(mag, arm_yaw, arm_pitch)

        # Convert to NED frame (required by drone-hover)
        x, y, z = u.ENU_to_NED(global_x, global_y, global_z)

        propeller_positions.append([float(x), float(y), float(z)])

        # Determine rotation direction
        rotation = "cw" if direction > 0.5 else "ccw"

        # Motor thrust direction (simplified - mostly downward)
        # In NED frame, motors typically point downward (positive Z)
        props.append({
            "loc": [float(x), float(y), float(z)],
            "dir": [0, 0, -1, rotation],  # Downward thrust in NED
            "propsize": 2  # Default 2-inch propeller
        })

    # Generate 8 mounting points on 60mm diameter disc
    disc_mounting_points = generate_disc_mounting_points(num_points=8, diameter=0.060)

    # Assign each propeller to nearest mounting point
    mounting_points = assign_nearest_mounting_point(propeller_positions, disc_mounting_points)

    try:
        # Use drone-hover's Custombody with mounting points for physics-based calculation
        drone = Custombody(props, mountpoints=mounting_points)

        return [drone.Ix, drone.Iy, drone.Iz, drone.Ixy, drone.Ixz, drone.Iyz]

    except Exception as e:
        # Fallback to minimal inertia if calculation fails
        print(f"Warning: Inertia calculation failed: {e}. Using default values.")
        return [0.01, 0.01, 0.01, 0, 0, 0]