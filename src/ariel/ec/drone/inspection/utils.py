"""
Utility functions for coordinate conversions and transformations used in visualization.

This module provides coordinate system conversions between different genome representations
and common transformation utilities for the airevolve inspection tools.
"""

from __future__ import annotations
from typing import Union, Tuple
import numpy as np
import numpy.typing as npt


def convert_to_cartesian(
    magnitude: float, 
    azimuth: float, 
    pitch: float, 
    in_degrees: bool = False
) -> Tuple[float, float, float]:
    """
    Convert spherical/polar coordinates to Cartesian coordinates.
    
    Args:
        magnitude: Distance from origin
        azimuth: Azimuthal angle (rotation around z-axis)
        pitch: Pitch angle (elevation from xy-plane)
        in_degrees: If True, angles are in degrees, else radians
        
    Returns:
        Tuple of (x, y, z) coordinates
    """
    if in_degrees:
        azimuth = np.radians(azimuth)
        pitch = np.radians(pitch)
    
    x = magnitude * np.cos(pitch) * np.cos(azimuth)
    y = magnitude * np.cos(pitch) * np.sin(azimuth)
    z = magnitude * np.sin(pitch)
    
    return (x, y, z)


def convert_to_spherical(
    x: float, 
    y: float, 
    z: float, 
    in_degrees: bool = False
) -> Tuple[float, float, float]:
    """
    Convert Cartesian coordinates to spherical/polar coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate  
        z: Z coordinate
        in_degrees: If True, return angles in degrees, else radians
        
    Returns:
        Tuple of (magnitude, azimuth, pitch)
    """
    magnitude = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    pitch = np.arcsin(z / magnitude) if magnitude > 0 else 0.0
    
    if in_degrees:
        azimuth = np.degrees(azimuth)
        pitch = np.degrees(pitch)
    
    return (magnitude, azimuth, pitch)


def create_rotation_matrix_euler(
    roll: float, 
    pitch: float, 
    yaw: float
) -> npt.NDArray[np.floating]:
    """
    Create a 3D rotation matrix from Euler angles (ZYX convention).
    
    Args:
        roll: Rotation around x-axis (radians)
        pitch: Rotation around y-axis (radians)
        yaw: Rotation around z-axis (radians)
        
    Returns:
        3x3 rotation matrix
    """
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (ZYX order)
    return Rz @ Ry @ Rx

def create_rotation_matrix_quaternion(
    quaternion: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """
    Create a 3D rotation matrix from a quaternion.
    
    Args:
        quaternion: A 4-element array-like representing the quaternion [w, x, y, z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = quaternion
    
    # Compute the rotation matrix elements
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])
    
    return R

def create_rotation_matrix_motor_orientation(
    motor_yaw: float, 
    motor_pitch: float
) -> npt.NDArray[np.floating]:
    """
    Create a rotation matrix for motor orientation in polar coordinate system.
    
    Args:
        motor_yaw: Motor yaw angle (radians)
        motor_pitch: Motor pitch angle (radians)
        
    Returns:
        3x3 rotation matrix
    """
    # For polar coordinates, we use a different convention
    Ry = np.array([
        [np.cos(-motor_pitch), 0, np.sin(-motor_pitch)],
        [0, 1, 0],
        [-np.sin(-motor_pitch), 0, np.cos(-motor_pitch)]
    ])
    
    Rz = np.array([
        [np.cos(motor_yaw), -np.sin(motor_yaw), 0],
        [np.sin(motor_yaw), np.cos(motor_yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry


def normalize_angles(angles: Union[float, npt.NDArray]) -> Union[float, npt.NDArray]:
    """
    Normalize angles to [-π, π] range.
    
    Args:
        angles: Single angle or array of angles in radians
        
    Returns:
        Normalized angles in [-π, π] range
    """
    return ((angles + np.pi) % (2 * np.pi)) - np.pi


def extract_genome_data(genome_handler) -> dict:
    """
    Extract visualization data from a Cartesian genome handler.
    
    Args:
        genome_handler: CartesianEulerDroneGenomeHandler instance
        
    Returns:
        Dictionary with positions, orientations, and directions
    """
    return {
        'positions': genome_handler.get_motor_positions(),
        'orientations': genome_handler.get_motor_orientations(), 
        'directions': genome_handler.get_propeller_directions(),
        'coordinate_system': 'cartesian'
    }


def extract_polar_genome_data(genome_array: npt.NDArray) -> dict:
    """
    Extract visualization data from a polar coordinate genome array.
    
    Args:
        genome_array: Array with shape (n_arms, 6) containing
                     [magnitude, azimuth, pitch, motor_pitch, motor_yaw, direction]
        
    Returns:
        Dictionary with converted data for visualization
    """
    # Remove any invalid (NaN) entries
    valid_mask = ~np.isnan(genome_array).any(axis=1)
    valid_genome = genome_array[valid_mask]
    
    positions = []
    orientations = []
    directions = []
    
    for arm in valid_genome:
        mag, azimuth, pitch, motor_pitch, motor_yaw, direction = arm[:6]
        
        # Convert position to Cartesian
        x, y, z = convert_to_cartesian(mag, azimuth, pitch)
        positions.append([x, y, z])
        
        # Store motor orientation
        orientations.append([0, motor_pitch, motor_yaw])  # roll=0 for polar system
        
        # Store direction
        directions.append(int(direction))
    
    return {
        'positions': np.array(positions),
        'orientations': np.array(orientations),
        'directions': np.array(directions),
        'coordinate_system': 'polar'
    }

def extract_cartesian_genome_data(genome_array: npt.NDArray) -> dict:
    """
    Extract visualization data from a Cartesian genome array.
    
    Args:
        genome_array: Array with shape (n_arms, 7) containing
                     [x, y, z, roll, pitch, yaw, direction]
        
    Returns:
        Dictionary with converted data for visualization
    """
    # Remove any invalid (NaN) entries
    valid_mask = ~np.isnan(genome_array).any(axis=1)
    valid_genome = genome_array[valid_mask]
    
    positions = []
    orientations = []
    directions = []
    
    for arm in valid_genome:
        x, y, z, roll, pitch, yaw, direction = arm[:7]
        
        positions.append([x, y, z])
        orientations.append([roll, pitch, yaw])
        directions.append(int(direction))
    
    return {
        'positions': np.array(positions),
        'orientations': np.array(orientations),
        'directions': np.array(directions),
        'coordinate_system': 'cartesian'
    }


def auto_extract_genome_data(genome_data) -> dict:
    """
    Automatically extract visualization data from different genome formats.
    
    Args:
        genome_data: Either a genome handler object or numpy array
        
    Returns:
        Dictionary with standardized visualization data
        
    Raises:
        ValueError: If genome format is not recognized
    """
    # Check if it's a genome handler with methods
    if hasattr(genome_data, 'get_motor_positions'):
        return extract_genome_data(genome_data)
    
    # Check if it's a numpy array (polar format)
    elif isinstance(genome_data, np.ndarray):
        if genome_data.ndim == 2 and genome_data.shape[1] == 6:
            return extract_polar_genome_data(genome_data)
        elif genome_data.ndim == 2 and genome_data.shape[1] == 7:
            return extract_cartesian_genome_data(genome_data)
    
    # Check if it has a 'genome' attribute (genome handler)
    elif hasattr(genome_data, 'genome'):
        if hasattr(genome_data, 'get_motor_positions'):
            return extract_cartesian_genome_data(genome_data)
        else:
            # Assume polar format in genome attribute
            return extract_polar_genome_data(genome_data.genome)
    
    else:
        raise ValueError(f"Unrecognized genome format: {type(genome_data)}")


def compute_visualization_bounds(positions: npt.NDArray, scale_factor: float = 1.2) -> dict:
    """
    Compute appropriate bounds for visualization based on genome positions.
    
    Args:
        positions: Array of positions with shape (n_arms, 3)
        scale_factor: Factor to scale bounds beyond max position
        
    Returns:
        Dictionary with xlim, ylim, zlim bounds
    """
    if len(positions) == 0:
        # Default bounds if no positions
        default_lim = [-0.5, 0.5]
        return {'xlim': default_lim, 'ylim': default_lim, 'zlim': default_lim}
    
    max_extent = np.max(np.abs(positions))
    limit = max_extent * scale_factor
    
    return {
        'xlim': [-limit, limit],
        'ylim': [-limit, limit], 
        'zlim': [-limit, limit]
    }


def evolution_dataframe_to_fitness_array(evolution_df, population_size: int = None) -> npt.NDArray[np.floating]:
    """
    Convert pandas DataFrame from evolve() to fitness array format expected by plot_fitness().
    
    Args:
        evolution_df: DataFrame returned by evolve() with columns ['generation', 'fitness', 'in_pop']
        population_size: Expected population size. If None, inferred from data
        
    Returns:
        2D numpy array of shape (n_generations, pop_size) with fitness values
        
    Raises:
        ValueError: If DataFrame is empty or missing required columns
    """
    import pandas as pd
    
    if evolution_df.empty:
        raise ValueError("Evolution DataFrame is empty")
    
    required_columns = ['generation', 'fitness']
    missing_columns = [col for col in required_columns if col not in evolution_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Get generation info
    generations = sorted(evolution_df['generation'].unique())
    n_generations = len(generations)
    
    # Determine population size
    if population_size is None:
        # Use the maximum number of individuals in any generation
        gen_counts = evolution_df['generation'].value_counts()
        population_size = gen_counts.max()
    
    # Initialize fitness array with NaN
    fitness_array = np.full((n_generations, population_size), np.nan)
    
    # Fill fitness values for each generation
    for gen_idx, gen in enumerate(generations):
        gen_data = evolution_df[evolution_df['generation'] == gen]
        
        # Get fitness values, sorted by fitness (best first)
        fitness_values = gen_data['fitness'].sort_values(ascending=False).values
        
        # Fill as many values as we have, up to population_size
        n_individuals = min(len(fitness_values), population_size)
        fitness_array[gen_idx, :n_individuals] = fitness_values[:n_individuals]
    
    return fitness_array


def get_points(individual: npt.NDArray) -> npt.NDArray:
    """Convert a phenotype array to 3D Cartesian points.

    Args:
        individual: Array with shape (n_arms, 6) containing
                   [magnitude, azimuth, pitch, motor_pitch, motor_yaw, direction].
                   Rows that are all-NaN are skipped.

    Returns:
        Array of shape (n_valid_arms, 3) with Cartesian [x, y, z] coordinates.
    """
    valid_mask = ~np.isnan(individual).any(axis=1)
    valid = individual[valid_mask]

    points = []
    for arm in valid:
        mag, azimuth, pitch = arm[0], arm[1], arm[2]
        x, y, z = convert_to_cartesian(mag, azimuth, pitch)
        points.append([x, y, z])

    return np.array(points) if points else np.empty((0, 3))


def ENU_to_NED(x, y, z):
    """
    Convert Earth-Centered, Earth-Fixed (ENU) coordinates to North-East-Down (NED) coordinates.
    
    Parameters:
    x (float): The x-coordinate in ENU.
    y (float): The y-coordinate in ENU.
    z (float): The z-coordinate in ENU.
    
    Returns:
    tuple: The x, y, and z coordinates in NED.
    """
    return (y, x, -z)

