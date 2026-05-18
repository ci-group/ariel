"""
Drone Visualization Components

This module provides drone-specific graphics functions including mesh creation,
coordinate transformations, and drone configuration utilities.

Functions:
    Coordinate Conversion:
        - convert_to_cartesian: Convert magnitude/angles to cartesian coordinates
        - ENU_to_NED: Convert between coordinate systems
        - euler_to_rotation_matrix: Create rotation matrices from euler angles
    
    Mesh Creation:
        - create_grid: Create ground grid mesh
        - create_path: Create path/line mesh from points
        - create_circle: Create circular mesh (for propellers)
        - group: Combine multiple meshes
        
    Drone Construction:
        - create_drone: Build complete drone mesh from individual configuration
        - set_thrust: Update force vectors for thrust visualization
        - create_individual_from_config: Generate drone configuration arrays
"""

import numpy as np
from .graphics_3d import Mesh, Force, rotation_matrix

def convert_to_cartesian(magnitude, yaw, pitch, in_degrees=False):
    """
    Convert a vector from magnitude, yaw, and pitch to Cartesian coordinates (x, y, z).
    
    Parameters:
    magnitude (float): The magnitude of the vector.
    yaw (float): The yaw angle in degrees or radians.
    pitch (float): The pitch angle in degrees or radians.
    in_degrees (bool): Whether the yaw and pitch angles are given in degrees. Default is True.
    
    Returns:
    tuple: Cartesian coordinates (x, y, z).
    """
    if in_degrees:
        yaw = np.radians(yaw)
        pitch = np.radians(pitch)
    
    # Calculate Cartesian coordinates
    x = magnitude * np.cos(pitch) * np.cos(yaw)
    y = magnitude * np.cos(pitch) * np.sin(yaw)
    z = magnitude * np.sin(pitch)
    
    return (x, y, z)

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

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    The rotation matrix follows the z-y-x convention (yaw-pitch-roll).
    
    Args:
        roll, pitch, yaw: Euler angles in radians
        
    Returns:
        3x3 rotation matrix
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

def create_grid(rows, cols, length):
    """
    Create a grid mesh for ground plane visualization.
    
    Args:
        rows: Number of rows
        cols: Number of columns  
        length: Grid cell size
        
    Returns:
        Mesh object representing the grid
    """
    rows, cols = rows+1, cols+1     # extra vertex in each direction
    vertices = np.zeros([rows * cols, 3])
    edges = []
    for i in range(rows):
        for j in range(cols):
            vertices[i * cols + j] = [
                i * length - (rows - 1) * length / 2,
                j * length - (cols - 1) * length / 2,
                0.
            ]
            if i != 0:
                edges.append((cols * (i - 1) + j, cols * i + j))
            if j != 0:
                edges.append((cols * i + j - 1, cols * i + j))
    return Mesh(vertices, np.array(edges))

def create_path(vertices, loop=False):
    """
    Create a path mesh connecting vertices in sequence.
    
    Args:
        vertices: Array of 3D points to connect
        loop: Whether to connect last point back to first
        
    Returns:
        Mesh object representing the path
    """
    edges = [(i, i+1) for i in range(len(vertices)-1)]
    if loop:
        edges.append((0, len(vertices)-1))
    return Mesh(np.array(vertices), np.array(edges))

def create_circle(r, px, py, pz, num=20, angle_x=0, angle_y=0, angle_z=0):
    """
    Create a circular mesh (used for propellers).

    Args:
        r: Radius of circle
        px, py, pz: Center position
        num: Number of vertices around circle
        angle_x, angle_y, angle_z: Rotation angles

    Returns:
        Mesh object representing the circle
    """
    vertices = np.array([[
        r * np.cos(i * 2 * np.pi / num),
        r * np.sin(i * 2 * np.pi / num),
        0
    ] for i in range(num)])

    R = euler_to_rotation_matrix(0, angle_y, angle_z)
    transform_from_ENU_to_NED = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    R = transform_from_ENU_to_NED @ R

    vertices = (R @ vertices.T).T
    vertices += np.array([px, py, pz])

    return create_path(vertices, loop=True)

def create_circle_oriented(r, px, py, pz, normal, num=20):
    """
    Create a circular mesh perpendicular to a given normal vector.

    Args:
        r: Radius of circle
        px, py, pz: Center position
        normal: 3-vector; the circle's plane is perpendicular to this (the
            disk's normal will equal this direction, normalized).
        num: Number of vertices around circle

    Returns:
        Mesh object representing the circle
    """
    n = np.asarray(normal, dtype=float)
    nlen = np.linalg.norm(n)
    if nlen < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = n / nlen

    # Pick a reference axis that isn't parallel to the normal
    ref = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)

    angles = np.arange(num) * (2 * np.pi / num)
    vertices = (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)) * r
    vertices += np.array([px, py, pz])

    return create_path(vertices, loop=True)

def group(mesh_list):
    """
    Combine multiple meshes into a single mesh.
    
    Args:
        mesh_list: List of Mesh objects to combine
        
    Returns:
        Combined Mesh object
    """
    vertices = np.concatenate([
        mesh.vertices for mesh in mesh_list
    ])
    index_shifts = np.cumsum(
        [0] + [len(mesh.vertices) for mesh in mesh_list][:-1]
    )
    edges = np.concatenate([
        mesh.edges + shift for (mesh, shift) in zip(mesh_list, index_shifts)
    ])
    return Mesh(vertices, edges)

def create_drone(propellers, box_size=[0.2,0.2,0.2], prop_radius=0.0254, scale=0.5, motor_colors=None):
    """
    Create a complete drone mesh from propeller configuration.
    
    Args:
        propellers: List of propeller dictionaries, each containing:
            - "loc": [x, y, z] position in body frame (meters)
            - "dir": [x, y, z, rotation] thrust direction and spin direction
            - "propsize": propeller size in inches (optional, not used for visualization)
        box_size: [width, depth, height] of central body
        prop_radius: Radius of propeller circles
        scale: Overall scale factor
        
    Returns:
        tuple: (drone_mesh, force_objects)
            drone_mesh: Complete Mesh object for the drone
            force_objects: List of Force objects for thrust visualization
    """

    box_size = np.array(box_size) * scale
    prop_radius *= scale

    # Create central body box
    bot_box = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2, box_size[2]/2],
    ]), loop=True)

    top_box = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, -box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2, -box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2, -box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2, -box_size[2]/2],
    ]), loop=True)

    # Create vertical edges connecting top and bottom
    box_side_line1 = create_path(np.array([
        [ box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2,  box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line2 = create_path(np.array([
        [-box_size[0]/2,  box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2,  box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line3 = create_path(np.array([
        [-box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [-box_size[0]/2, -box_size[1]/2,-box_size[2]/2],
    ]))

    box_side_line4 = create_path(np.array([
        [ box_size[0]/2, -box_size[1]/2, box_size[2]/2],
        [ box_size[0]/2, -box_size[1]/2,-box_size[2]/2],
    ]))

    # Start with central body components
    drawings = [bot_box, top_box, box_side_line1, box_side_line2, box_side_line3, box_side_line4]
    num_body_meshes = len(drawings)
    centres = []

    # Create arms and propellers based on propeller configuration
    for prop in propellers:
        # Get propeller location and direction directly
        loc = np.array(prop["loc"]) * scale
        x, y, z = loc[0], loc[1], loc[2]

        # Motor thrust direction (body frame, NED). The propeller disk's normal
        # equals this direction, so the disk lies perpendicular to the thrust.
        motor_dir = np.array(prop["dir"][:3], dtype=float)

        # Create propeller disk perpendicular to thrust direction
        circle = create_circle_oriented(prop_radius, x, y, z, normal=motor_dir, num=20)
        # Create arm line from center to propeller
        arm_line = create_path(np.array([[0, 0, 0], [x, y, z]]))

        drawings.append(circle)
        drawings.append(arm_line)
        centres.append([x, y, z])
    
    # Combine all mesh components
    drone = group(drawings)

    # Per-edge colors: body stays default, each propeller's circle + arm get its motor color.
    # Matches the edge order produced by group() above.
    if motor_colors is not None:
        edge_colors = []
        for m in drawings[:num_body_meshes]:
            edge_colors.extend([None] * len(m.edges))
        prop_drawings = drawings[num_body_meshes:]
        for prop_idx in range(len(propellers)):
            color = motor_colors[prop_idx % len(motor_colors)]
            circle = prop_drawings[2 * prop_idx]
            arm_line = prop_drawings[2 * prop_idx + 1]
            edge_colors.extend([color] * len(circle.edges))
            edge_colors.extend([color] * len(arm_line.edges))
        drone.edge_colors = edge_colors

    # Add propeller centers as additional vertices for force attachment
    drone.vertices = np.concatenate([
        drone.vertices,
        np.array(centres)  # centers of the circles
    ])

    # Create force objects at each propeller location. Each force carries the
    # motor's thrust direction (body frame) and color, so the arrow points the
    # right way for canted motors and matches the rotor's color.
    forces = []
    for v, prop in zip(drone.vertices[-len(propellers):], propellers):
        f = Force(v)
        f.body_dir = np.array(prop["dir"][:3], dtype=float)
        forces.append(f)
    if motor_colors is not None:
        for i, f in enumerate(forces):
            f.color = motor_colors[i % len(motor_colors)]

    return drone, forces

def set_thrust(drone, forces, T, base_len=0.0):
    """
    Update force arrows to represent per-motor thrust direction and magnitude.

    Each Force uses its own body-frame direction if available, so canted motors
    point correctly. ``base_len`` adds a small always-on length so the direction
    stays visible even at low thrust.

    Args:
        drone: Drone mesh object (for orientation)
        forces: List of Force objects
        T: Array of thrust magnitudes for each motor
        base_len: Minimum arrow length added to every non-skipped motor
    """
    R = rotation_matrix(drone.theta)
    num_forces = len(forces)
    num_thrusts = len(T) if hasattr(T, '__len__') else 1
    num_motors = min(num_forces, num_thrusts)

    for i in range(num_motors):
        length = T[i] + base_len
        if forces[i].body_dir is not None:
            forces[i].F = length * (R @ forces[i].body_dir)
        else:
            forces[i].F = -length * R[:, 2]

    for i in range(num_motors, num_forces):
        forces[i].F = np.zeros(3)

