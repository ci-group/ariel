"""
Mounting Point Generator

This module provides utilities for generating mounting points on a central disc structure.
Used for realistic inertia calculations where drone arms attach to a central body/disc
rather than all emanating from a single origin point.
"""

import numpy as np


def generate_disc_mounting_points(num_points=8, diameter=0.060, z_offset=0.0):
    """
    Generate equally-spaced mounting points around a disc.

    This creates mounting points on a flat disc in the XY plane, representing
    where drone arms attach to a central body structure. Points are evenly
    distributed in a circular pattern.

    Args:
        num_points (int): Number of mounting points to generate (default: 8)
        diameter (float): Disc diameter in meters (default: 0.060 = 60mm)
        z_offset (float): Z-coordinate offset for all points (default: 0.0)

    Returns:
        list: List of np.array([x, y, z]) mounting point coordinates

    Example:
        >>> # Generate 8 mounting points on 60mm diameter disc
        >>> points = generate_disc_mounting_points(8, 0.060)
        >>> len(points)
        8
        >>> # Points are evenly spaced at 45 degree intervals
        >>> angles = [0, π/4, π/2, 3π/4, π, 5π/4, 3π/2, 7π/4]
    """
    radius = diameter / 2.0
    mounting_points = []

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = z_offset
        mounting_points.append(np.array([x, y, z]))

    return mounting_points


def assign_nearest_mounting_point(propeller_positions, mounting_points):
    """
    Assign each propeller to its nearest mounting point.

    For each propeller location, finds the closest mounting point on the disc.
    This simulates realistic attachment where arms connect to the nearest
    available mounting point on the central body.

    Args:
        propeller_positions (list or np.ndarray): List of [x, y, z] propeller locations
        mounting_points (list): List of np.array mounting point coordinates

    Returns:
        list: List of np.array mounting points, one per propeller (same order)

    Example:
        >>> prop_locs = [[0.10, 0.10, 0], [-0.10, 0.10, 0]]
        >>> mount_pts = generate_disc_mounting_points(8, 0.060)
        >>> assigned = assign_nearest_mounting_point(prop_locs, mount_pts)
        >>> len(assigned) == len(prop_locs)
        True
    """
    propeller_positions = np.array(propeller_positions)
    assigned_points = []

    for prop_pos in propeller_positions:
        # Calculate distances to all mounting points (XY plane only)
        distances = []
        for mount_pt in mounting_points:
            # Use XY distance only (ignore Z for assignment)
            dist = np.sqrt((prop_pos[0] - mount_pt[0])**2 +
                          (prop_pos[1] - mount_pt[1])**2)
            distances.append(dist)

        # Find nearest mounting point
        nearest_idx = np.argmin(distances)
        assigned_points.append(mounting_points[nearest_idx].copy())

    return assigned_points


def get_default_mounting_points(num_propellers):
    """
    Get default mounting points configuration for a given number of propellers.

    Uses an 8-point disc (60mm diameter) and assigns each propeller to the
    nearest mounting point. This is the standard configuration for evolved drones.

    Args:
        num_propellers (int): Number of propellers on the drone

    Returns:
        list: List of mounting points, one per propeller (all at origin for backwards compatibility)
              To use disc mounting, call generate_disc_mounting_points() directly

    Note:
        For backward compatibility, this returns origin points by default.
        Use generate_disc_mounting_points() for disc-based mounting.
    """
    # Return origin points for backward compatibility
    return [np.array([0.0, 0.0, 0.0]) for _ in range(num_propellers)]
