"""
B-Spline Pipe Generator

This module creates 3D hollow pipes from B-spline curves and exports them as STL files.
It includes visualization capabilities using PyVista and ensures consistency with MuJoCo
simulations by replicating the same alignment transformations.

Author: [Your Name]
Date: [Current Date]
"""

import pyvista as pv
import numpy as np
import trimesh
from typing import List, Tuple, Optional, Union

from lynx_manipulator.modules.bspline_tube import BSplineTube
from tools.math_utils import Vector3, Quaternion


def make_polyline(points: np.ndarray) -> pv.PolyData:
    """
    Create a PyVista PolyData polyline from an array of 3D points.
    
    Args:
        points: Nx3 numpy array of 3D coordinates
        
    Returns:
        PyVista PolyData object representing the polyline
        
    Raises:
        ValueError: If points array doesn't have the correct shape
    """
    points = np.asarray(points)
    
    # Handle transposed input (3xN instead of Nx3)
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T
        
    # Validate input shape
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must be Nx3, got shape {points.shape}")

    n = len(points)
    # VTK polyline format: [n, 0, 1, 2, ..., n-1]
    lines = np.hstack([[n], np.arange(n)])
    
    poly = pv.PolyData()
    poly.points = points
    poly.lines = lines
    return poly


def extend_spline(points: np.ndarray, extension: float = 0.2) -> np.ndarray:
    """
    Extend a spline curve at both ends by extrapolating the direction vectors.
    
    This is useful for creating inner tubes that extend beyond the outer tube
    to ensure proper boolean operations.
    
    Args:
        points: Nx3 array of spline points
        extension: Length to extend in each direction (default: 0.2)
        
    Returns:
        Extended points array with 2 additional points (one at each end)
    """
    # Calculate extension at start
    p0, p1 = points[0], points[1]
    start_dir = p0 - p1
    start_ext = p0 + start_dir / np.linalg.norm(start_dir) * extension

    # Calculate extension at end
    pn, pm = points[-1], points[-2]
    end_dir = pn - pm
    end_ext = pn + end_dir / np.linalg.norm(end_dir) * extension

    return np.vstack([start_ext, points, end_ext])


def spline_to_pipe(
    curve_points: Union[List, np.ndarray], 
    outer_radius: float = 0.3, 
    wall_thickness: float = 0.05,
    filename: str = "pipe.stl", 
    inner_extension: float = 0.2,
    visualize: bool = True
) -> pv.PolyData:
    """
    Convert a B-spline curve into a hollow pipe and export as STL.
    
    Args:
        curve_points: List or array of 3D points defining the curve centerline
        outer_radius: Outer radius of the pipe (default: 0.3)
        wall_thickness: Thickness of the pipe wall (default: 0.05)
        filename: Output STL filename (default: "pipe.stl")
        inner_extension: How much to extend the inner tube beyond outer tube (default: 0.2)
        visualize: Whether to show interactive visualization (default: True)
        
    Returns:
        PyVista PolyData object of the final pipe mesh
        
    Note:
        The inner tube is extended to prevent boolean operation artifacts
        at the pipe ends. The extension should be larger than the outer radius.
    """
    points = np.array(curve_points)

    # Create outer tube by sweeping a circle along the spline
    outer = make_polyline(points).tube(radius=outer_radius, n_sides=128, capping=True).triangulate()

    # Create extended inner tube for proper boolean subtraction
    extended_points = extend_spline(points, extension=inner_extension)
    inner_radius = outer_radius - wall_thickness
    inner = make_polyline(extended_points).tube(radius=inner_radius, n_sides=128, capping=True).triangulate()

    # Convert to trimesh for robust boolean operations
    outer_trimesh = trimesh.Trimesh(
        vertices=outer.points,
        faces=outer.faces.reshape(-1, 4)[:, 1:]  # Remove VTK cell size info
    )
    inner_trimesh = trimesh.Trimesh(
        vertices=inner.points,
        faces=inner.faces.reshape(-1, 4)[:, 1:]
    )
    
    # Perform boolean difference to create hollow pipe
    pipe_trimesh = outer_trimesh.difference(inner_trimesh)

    # Export STL file
    pipe_trimesh.export(filename)
    print(f"Pipe exported to: {filename}")

    # Convert back to PyVista for visualization
    pipe = pv.wrap(pipe_trimesh)
    
    if visualize:
        plotter = pv.Plotter()
        plotter.add_mesh(pipe, color="lightblue", smooth_shading=True, 
                        label="Hollow Pipe")
        plotter.add_mesh(make_polyline(points), color="red", line_width=3,
                        label="Centerline")
        plotter.add_legend()
        plotter.show_grid()
        plotter.show()

    return pipe


def calculate_mujoco_direction_vector(attachment_point_quat: Quaternion) -> Vector3:
    """
    Calculate the direction vector using the same method as MuJoCo's build() function.
    
    This ensures consistency between the exported STL and the MuJoCo simulation
    by applying the same alignment transformations.
    
    Args:
        attachment_point_quat: Quaternion representing the attachment point orientation
        
    Returns:
        Direction vector for B-spline alignment
        
    Note:
        This replicates the exact calculation from BSplineTube.build():
        vector_part = np.array(attachment_point_quat[:-1])
        q = Quaternion([0.0, -1.0/sqrt(2), 0.0, 1.0/sqrt(2)])
        direction_vector = q * Vector3(vector_part / |vector_part|)
    """
    vector_part = np.array([attachment_point_quat.x, attachment_point_quat.y, attachment_point_quat.z])
    q = Quaternion([0.0, -1.0/np.sqrt(2), 0.0, 1.0/np.sqrt(2)])
    direction_vector = q * Vector3(vector_part / np.linalg.norm(vector_part))
    return direction_vector


def create_bspline_pipe(
    tube_design: dict,
    attachment_point_quat: Quaternion,
    pipe_params: Optional[dict] = None,
    scale_factor: float = 1.0
) -> Tuple[BSplineTube, pv.PolyData]:
    """
    Create a B-spline tube and convert it to a pipe, ensuring MuJoCo consistency.
    
    Args:
        tube_design: Dictionary containing B-spline parameters:
            - control_points: List of Vector3 control points
            - degree: B-spline degree
            - num_segments: Number of curve segments
            - cylinder_radius: Radius for visualization
            - name: Tube name
        attachment_point_quat: Quaternion for alignment (from MuJoCo)
        pipe_params: Optional dictionary for pipe generation:
            - outer_radius: Pipe outer radius
            - wall_thickness: Pipe wall thickness  
            - filename: STL output filename
            - inner_extension: Inner tube extension
            - visualize: Whether to show visualization
        scale_factor: Scaling factor to convert units to meters (default: 1.0)
            - Use 0.001 if your coordinates are in mm
            - Use 0.01 if your coordinates are in cm
            - Use 1.0 if your coordinates are already in meters
            
    Returns:
        Tuple of (BSplineTube object, PyVista pipe mesh)
    """
    # Default pipe parameters
    default_pipe_params = {
        'outer_radius': 0.042,
        'wall_thickness': 0.01,
        'filename': 'pipe.stl',
        'inner_extension': 0.01,
        'visualize': True
    }
    
    if pipe_params:
        default_pipe_params.update(pipe_params)

    # Scale tube design parameters
    scaled_tube_design = tube_design.copy()
    
    # Scale control points
    scaled_control_points = []
    for point in tube_design["control_points"]:
        scaled_point = Vector3([
            point.x * scale_factor,
            point.y * scale_factor,
            point.z * scale_factor
        ])
        scaled_control_points.append(scaled_point)
    scaled_tube_design["control_points"] = scaled_control_points
    
    # Scale cylinder radius
    scaled_tube_design["cylinder_radius"] = tube_design["cylinder_radius"] * scale_factor
    
    # Scale pipe parameters
    scaled_pipe_params = {}
    for key, value in default_pipe_params.items():
        if key in ['outer_radius', 'wall_thickness', 'inner_extension']:
            scaled_pipe_params[key] = value * scale_factor
        else:
            scaled_pipe_params[key] = value

    # Create B-spline tube with scaled parameters
    tube = BSplineTube(
        control_points=scaled_tube_design["control_points"],
        degree=scaled_tube_design["degree"],
        num_segments=scaled_tube_design["num_segments"],
        cylinder_radius=scaled_tube_design["cylinder_radius"],
        name=scaled_tube_design["name"]
    )

    # Apply MuJoCo-consistent alignment
    direction_vector = calculate_mujoco_direction_vector(attachment_point_quat)
    print(f"Alignment direction: [{direction_vector.x:.3f}, {direction_vector.y:.3f}, {direction_vector.z:.3f}]")
    print(f"Scale factor applied: {scale_factor} (coordinates scaled to meters)")
    
    tube.align_start_with_direction(direction_vector)

    # Sample curve points and convert to numpy array
    curve_points = tube._sample_bspline_curve()
    curve_points_array = np.array([[point.x, point.y, point.z] for point in curve_points])

    # Generate pipe with scaled parameters
    pipe = spline_to_pipe(curve_points_array, **scaled_pipe_params)

    return tube, pipe


# =============================================================================
# PREDEFINED TUBE DESIGNS
# =============================================================================

TUBE_DESIGNS = {
    
    "slight_bend": {  # Coordinates already in meters
        "control_points": [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.14, 0.0]),
            Vector3([0.0, 0.28, 0.0]),
        ],
        "degree": 2,
        "num_segments": 10,
        "cylinder_radius": 0.035,
        "name": "straight_tube_m"
    },
    
    "right_angle": {  # Coordinates already in meters
        "control_points": [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.0, 0.1]),
            Vector3([0.0, 0.1, 0.1]),
            Vector3([0.0, 0.14, 0.14]),
        ],
        "degree": 3,
        "num_segments": 20,
        "cylinder_radius": 0.035,
        "name": "right_angle_tube_m"
    },

    "straight": {  # Coordinates already in meters
        "control_points": [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.28, 0.0]),
        ],
        "degree": 1,
        "num_segments": 5,
        "cylinder_radius": 0.035,
        "name": "straight_tube_m"
    },
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    attachment_quat = Quaternion([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)])
    
    tube1, pipe1 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["slight_bend"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            'filename': 'slight_bend_pipe.stl',
            'outer_radius': 0.035,    # 42mm = 0.042m
            'wall_thickness': 0.001,   # 10mm = 0.01m
            'visualize': False
        }
    )

    tube2, pipe2 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["right_angle"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            'filename': 'right_angle_pipe.stl',
            'outer_radius': 0.035,    # 42mm = 0.042m
            'wall_thickness': 0.001,   # 10mm = 0.01m
            'visualize': False
        }
    )

    tube3, pipe3 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["straight"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            'filename': 'straight_pipe.stl',
            'outer_radius': 0.035,    # 42mm = 0.042m
            'wall_thickness': 0.001,   # 10mm = 0.01m
            'visualize': False
        }
    )

    print("All pipes generated with proper meter scaling!")
    print("Check the STL files - they should now be properly sized for MuJoCo.")

