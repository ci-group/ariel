"""
B-Spline Pipe Generator.

This module creates 3D hollow pipes from B-spline curves and exports them as STL files.
It includes visualization capabilities using PyVista and ensures consistency with MuJoCo
simulations by replicating the same alignment transformations.

Author: [Your Name]
Date: [Current Date]
"""


import numpy as np
import pyvista as pv
import trimesh
from omegaconf import DictConfig, OmegaConf

from ariel import ROOT, CWD
from ariel.body_phenotypes.Lynx_Arm.lynx.morphlib.tools.math_utils import Quaternion, Vector3
from ariel.body_phenotypes.Lynx_Arm.lynx.robots.lynx_manipulator.modules.bspline_tube_jed import (
    BSplineTube,
)
from ariel.body_phenotypes.Lynx_Arm.lynx.robots.lynx_manipulator.modules.bspline_tube_with_clamps import (
    BSplineTubeWithClamps,
)


def make_polyline(points: np.ndarray) -> pv.PolyData:
    """
    Create a PyVista PolyData polyline from an array of 3D points.

    Args:
        points: Nx3 numpy array of 3D coordinates

    Returns
    -------
        PyVista PolyData object representing the polyline

    Raises
    ------
        ValueError: If points array doesn't have the correct shape
    """
    points = np.asarray(points)

    # Handle transposed input (3xN instead of Nx3)
    if points.shape[0] == 3 and points.shape[1] != 3:
        points = points.T

    # Validate input shape
    if points.ndim != 2 or points.shape[1] != 3:
        msg = f"Points must be Nx3, got shape {points.shape}"
        raise ValueError(msg)

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

    Returns
    -------
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


def create_cross_structure(
    curve_points: np.ndarray,
    cross_radius: float,
    cross_length: float,
    cross_thickness: float = 0.001,
) -> trimesh.Trimesh:
    """
    Create two intersecting rectangular planes through the center of the pipe.

    This creates a cross (+) structure when viewed from the end of the pipe,
    with two thin rectangular planes passing through the centerline.

    Args:
        curve_points: Nx3 array of points along the pipe centerline
        cross_radius: Radius of each cross member (how far from centerline)
        cross_length: Length along the pipe that the cross spans
        cross_thickness: Thickness of each rectangular plane (default: 0.001)

    Returns
    -------
        Trimesh object containing the cross structure
    """
    # Calculate total curve length to determine which points to use
    total_length = 0.0
    for i in range(len(curve_points) - 1):
        segment_length = np.linalg.norm(curve_points[i + 1] - curve_points[i])
        total_length += segment_length

    # Select curve points that span the desired cross_length from the start
    accumulated_length = 0.0
    selected_indices = [0]

    for i in range(len(curve_points) - 1):
        segment_length = np.linalg.norm(curve_points[i + 1] - curve_points[i])
        accumulated_length += segment_length

        if accumulated_length <= cross_length:
            selected_indices.append(i + 1)
        else:
            break

    # Make sure we have at least 2 points
    if len(selected_indices) < 2:
        selected_indices = [0, 1]

    selected_points = curve_points[selected_indices]

    rectangles = []

    # Pre-compute tangents and perpendicular frames for all selected points
    # This ensures smooth frame propagation without flipping
    tangents = []
    perp1_list = []
    perp2_list = []

    for j, center in enumerate(selected_points):
        idx = selected_indices[j]
        if idx == 0:
            tangent = curve_points[1] - curve_points[0]
        elif idx == len(curve_points) - 1:
            tangent = curve_points[-1] - curve_points[-2]
        else:
            tangent = curve_points[idx + 1] - curve_points[idx - 1]
        tangent /= np.linalg.norm(tangent)
        tangents.append(tangent)

        if j == 0:
            # Initialize perpendicular frame at first point
            if abs(tangent[2]) < 0.9:
                arbitrary = np.array([0, 0, 1])
            else:
                arbitrary = np.array([1, 0, 0])
            perp1 = np.cross(tangent, arbitrary)
            perp1 /= np.linalg.norm(perp1)
            perp2 = np.cross(tangent, perp1)
            perp2 /= np.linalg.norm(perp2)
        else:
            # Propagate frame from previous point (rotation minimizing)
            tangents[j - 1]
            prev_perp1 = perp1_list[j - 1]
            prev_perp2 = perp2_list[j - 1]

            # Rotate previous frame to align with new tangent
            # Use Gram-Schmidt to keep perp1 perpendicular to new tangent
            perp1 = prev_perp1 - np.dot(prev_perp1, tangent) * tangent
            perp1_norm = np.linalg.norm(perp1)
            if perp1_norm > 1e-6:
                perp1 /= perp1_norm
            else:
                # Fallback if perp1 becomes parallel to tangent
                perp1 = np.cross(tangent, prev_perp2)
                perp1 /= np.linalg.norm(perp1)

            perp2 = np.cross(tangent, perp1)
            perp2 /= np.linalg.norm(perp2)

        perp1_list.append(perp1)
        perp2_list.append(perp2)

    # Create two perpendicular rectangles
    for angle in [0, np.pi / 2]:  # Two rectangles at 0° and 90°

        vertices = []

        for j, center in enumerate(selected_points):
            tangent = tangents[j]
            perp1 = perp1_list[j]
            perp2 = perp2_list[j]

            # Direction across the width of this rectangle (radial direction from centerline)
            width_direction = np.cos(angle) * perp1 + np.sin(angle) * perp2
            # Direction for the thickness (perpendicular to both tangent and width)
            # This is the OTHER radial direction, making a thin plane
            thickness_direction = np.cos(angle + np.pi / 2) * perp1 + np.sin(angle + np.pi / 2) * perp2

            # Create 4 corners of the rectangle at this cross-section
            # The rectangle extends cross_radius in the radial direction (width)
            # and has thickness cross_thickness perpendicular to the plane
            half_thickness = cross_thickness / 2

            p1 = center - width_direction * cross_radius - thickness_direction * half_thickness
            p2 = center + width_direction * cross_radius - thickness_direction * half_thickness
            p3 = center + width_direction * cross_radius + thickness_direction * half_thickness
            p4 = center - width_direction * cross_radius + thickness_direction * half_thickness

            vertices.extend([p1, p2, p3, p4])

        # Create faces connecting adjacent cross-sections
        faces = []
        for j in range(len(selected_points) - 1):
            base_idx = j * 4
            next_idx = (j + 1) * 4

            # Create two triangles for each of the 4 sides of the rectangle
            # Bottom face
            faces.append([base_idx + 0, base_idx + 1, next_idx + 1])
            faces.append([base_idx + 0, next_idx + 1, next_idx + 0])

            # Right face
            faces.append([base_idx + 1, base_idx + 2, next_idx + 2])
            faces.append([base_idx + 1, next_idx + 2, next_idx + 1])

            # Top face
            faces.append([base_idx + 2, base_idx + 3, next_idx + 3])
            faces.append([base_idx + 2, next_idx + 3, next_idx + 2])

            # Left face
            faces.append([base_idx + 3, base_idx + 0, next_idx + 0])
            faces.append([base_idx + 3, next_idx + 0, next_idx + 3])

        # Add end caps to close the rectangular beam
        # Start cap (first cross-section) - facing backward along the curve
        faces.append([0, 2, 1])  # Counter-clockwise when viewed from outside
        faces.append([0, 3, 2])

        # End cap (last cross-section) - facing forward along the curve
        last_idx = (len(selected_points) - 1) * 4
        faces.extend(([last_idx + 0, last_idx + 1, last_idx + 2], [last_idx + 0, last_idx + 2, last_idx + 3]))

        rect_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        rectangles.append(rect_mesh)

    # Combine both rectangles into a cross
    return trimesh.util.concatenate(rectangles)


def attach_clamp_at_position(
    clamp_mesh: trimesh.Trimesh,
    position: np.ndarray,
    tangent: np.ndarray,
    scale_factor: float = 1.0,
    offset: float = 0.0,
    rotation_offset: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """
    Position and orient a clamp at a specific location along the pipe.

    Args:
        clamp_mesh: The clamp mesh to position
        position: 3D position where the clamp should be placed
        tangent: Direction vector (tangent) at this position
        scale_factor: Scale to apply to the clamp (default: 1.0 for mm to meters conversion)
        offset: Distance to offset the clamp along the tangent direction (default: 0.0)
        rotation_offset: Optional 3x3 rotation matrix to apply to the clamp before alignment

    Returns
    -------
        Transformed clamp mesh
    """
    # Make a copy to avoid modifying the original
    clamp = clamp_mesh.copy()

    # Scale the clamp (assuming it's in mm, scale to meters)
    clamp.apply_scale(scale_factor)

    # Apply optional rotation offset if provided
    if rotation_offset is not None:
        transform = np.eye(4)
        transform[:3, :3] = rotation_offset
        clamp.apply_transform(transform)

    # First, flip the clamp 180 degrees around X-axis (in local frame)
    flip_x = np.array([
        [1, 0, 0, 0],
        [0, np.cos(np.pi), -np.sin(np.pi), 0],
        [0, np.sin(np.pi), np.cos(np.pi), 0],
        [0, 0, 0, 1],
    ])
    clamp.apply_transform(flip_x)

    # Calculate rotation to align clamp's Y-axis with the tangent direction
    # The clamp is oriented along Y-axis by default
    y_axis = np.array([0, 1, 0])
    tangent_normalized = tangent / np.linalg.norm(tangent)

    # Calculate rotation axis and angle
    rotation_axis = np.cross(y_axis, tangent_normalized)
    rotation_axis_norm = np.linalg.norm(rotation_axis)

    if rotation_axis_norm > 1e-6:  # Not parallel
        rotation_axis /= rotation_axis_norm
        angle = np.arccos(np.clip(np.dot(y_axis, tangent_normalized), -1.0, 1.0))

        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ])
        rotation_matrix = (np.eye(3) +
                          np.sin(angle) * K +
                          (1 - np.cos(angle)) * K @ K)
    # Parallel or anti-parallel
    elif np.dot(y_axis, tangent_normalized) > 0:
        rotation_matrix = np.eye(3)  # Same direction
    else:
        rotation_matrix = -np.eye(3)  # Opposite direction
        rotation_matrix[1, 1] = -1  # Flip around Y

    # Apply rotation
    clamp.apply_transform(np.vstack([
        np.hstack([rotation_matrix, [[0], [0], [0]]]),
        [0, 0, 0, 1],
    ]))

    # Apply offset along tangent direction if specified
    offset_position = position + tangent_normalized * offset

    # Translate to position
    clamp.apply_translation(offset_position)

    return clamp


def spline_to_pipe(
    curve_points: list | np.ndarray,
    outer_radius: float = 0.3,
    wall_thickness: float = 0.05,
    filename: str = "pipe.stl",
    inner_extension: float = 0.2,
    visualize: bool = True,
    add_cross: bool = False,
    cross_radius: float | None = None,
    cross_length: float | None = None,
    cross_thickness: float = 0.001,
    add_clamps: bool = False,
    clamp_stl_path: str = "clamp.stl",
    clamp_scale: float = 1.0,
    clamp_rotation_offset: np.ndarray | None = None,
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
        add_cross: Whether to add cross structure through center for rigidity (default: False)
        cross_radius: Radius of cross members; defaults to inner radius if None
        cross_length: Length along pipe that cross spans; defaults to total length if None
        cross_thickness: Thickness of cross planes in meters (default: 0.001)
        add_clamps: Whether to add clamps at pipe ends for smooth attachment transitions (default: False)
        clamp_stl_path: Path to the clamp STL file (default: "clamp.stl")
        clamp_scale: Scale factor for the clamp (default: 1.0, use same units as pipe)
        clamp_rotation_offset: Optional 3x3 rotation matrix to apply to the clamp (default: None)

    Returns
    -------
        PyVista PolyData object of the final pipe mesh

    Note:
        The inner tube is extended to prevent boolean operation artifacts
        at the pipe ends. The extension should be larger than the outer radius.
        When add_cross=True, two rectangular planes are added through the centerline
        forming a cross (+) pattern for structural rigidity.
        When add_clamps=True, clamps from an STL file are attached at both ends.
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
        faces=outer.faces.reshape(-1, 4)[:, 1:],  # Remove VTK cell size info
    )
    inner_trimesh = trimesh.Trimesh(
        vertices=inner.points,
        faces=inner.faces.reshape(-1, 4)[:, 1:],
    )

    # Perform boolean difference to create hollow pipe
    pipe_trimesh = outer_trimesh.difference(inner_trimesh)

    # Add cross structure if requested
    if add_cross:
        inner_radius = outer_radius - wall_thickness

        # Default cross radius to inner radius if not specified
        if cross_radius is None:
            cross_radius = inner_radius

        # Default cross length to full pipe length if not specified
        if cross_length is None:
            cross_length = sum(np.linalg.norm(points[i + 1] - points[i])
                             for i in range(len(points) - 1))

        cross = create_cross_structure(
            points,
            cross_radius=cross_radius,
            cross_length=cross_length,
            cross_thickness=cross_thickness,
        )
        pipe_trimesh = trimesh.util.concatenate([pipe_trimesh, cross])

    # Add clamps at pipe ends if requested
    if add_clamps:
        try:
            # Load the clamp mesh
            clamp_base = trimesh.load(clamp_stl_path)

            # Calculate tangent at start (first two points)
            start_tangent = points[1] - points[0]
            start_tangent /= np.linalg.norm(start_tangent)

            # Calculate tangent at end (last two points)
            end_tangent = points[-1] - points[-2]
            end_tangent /= np.linalg.norm(end_tangent)

            # Get clamp length (Y-dimension after scaling)
            clamp_bounds = clamp_base.bounds
            clamp_length = (clamp_bounds[1][1] - clamp_bounds[0][1]) * clamp_scale  # Y-dimension

            # Attach clamp at start (extends backward, so use negative tangqent)
            # Offset by clamp_length so clamp extends outward from pipe end
            start_clamp = attach_clamp_at_position(
                clamp_base,
                points[0],
                -start_tangent,  # Negative to extend backward from pipe start
                scale_factor=clamp_scale,
                offset=clamp_length,  # Move outward by clamp length
                rotation_offset=clamp_rotation_offset,
            )

            # Attach clamp at end (extends forward along tangent)
            # Offset by clamp_length so clamp extends outward from pipe end
            end_clamp = attach_clamp_at_position(
                clamp_base,
                points[-1],
                end_tangent,
                scale_factor=clamp_scale,
                offset=clamp_length,  # Move outward by clamp length
                rotation_offset=clamp_rotation_offset,
            )

            # Combine with pipe
            pipe_trimesh = trimesh.util.concatenate([pipe_trimesh, start_clamp, end_clamp])

        except FileNotFoundError:
            pass
        except Exception:
            pass

    # Export STL file
    pipe_trimesh.export(filename)

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

    Returns
    -------
        Direction vector for B-spline alignment

    Note:
        This replicates the exact calculation from BSplineTube.build():
        vector_part = np.array(attachment_point_quat[:-1])
        q = Quaternion([0.0, -1.0/sqrt(2), 0.0, 1.0/sqrt(2)])
        direction_vector = q * Vector3(vector_part / |vector_part|)
    """
    vector_part = np.array([attachment_point_quat.x, attachment_point_quat.y, attachment_point_quat.z])
    q = Quaternion([0.0, -1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2)])
    return q * Vector3(vector_part / np.linalg.norm(vector_part))


def create_bspline_pipe(
    tube_design: dict,
    attachment_point_quat: Quaternion,
    pipe_params: dict | None = None,
    scale_factor: float = 1.0,
) -> tuple[BSplineTube, pv.PolyData]:
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

    Returns
    -------
        Tuple of (BSplineTube object, PyVista pipe mesh)
    """
    # Default pipe parameters
    default_pipe_params = {
        "outer_radius": 0.042,
        "wall_thickness": 0.01,
        "filename": "pipe.stl",
        "inner_extension": 0.01,
        "visualize": True,
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
            point.z * scale_factor,
        ])
        scaled_control_points.append(scaled_point)
    scaled_tube_design["control_points"] = scaled_control_points

    # Scale cylinder radius
    scaled_tube_design["cylinder_radius"] = tube_design["cylinder_radius"] * scale_factor

    # Scale pipe parameters
    scaled_pipe_params = {}
    for key, value in default_pipe_params.items():
        if key in {"outer_radius", "wall_thickness", "inner_extension", "cross_radius", "cross_length", "cross_thickness"}:
            scaled_pipe_params[key] = value * scale_factor
        else:
            scaled_pipe_params[key] = value

    # Handle clamp rotation offset
    # Apply 90 degree rotation around Y axis for all clamps
    angle = np.pi / 2
    scaled_pipe_params["clamp_rotation_offset"] = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

    # Create B-spline tube with scaled parameters
    scaled_tube_design["mounting_length_start"]
    tube = BSplineTube(
        control_points=scaled_tube_design["control_points"],
        degree=scaled_tube_design["degree"],
        num_segments=scaled_tube_design["num_segments"],
        cylinder_radius=scaled_tube_design["cylinder_radius"],
        name=scaled_tube_design["name"],
        mounting_length_start=scaled_tube_design["mounting_length_start"],
        mounting_length_end=scaled_tube_design["mounting_length_end"],
    )

    # Apply MuJoCo-consistent alignment
    direction_vector = calculate_mujoco_direction_vector(attachment_point_quat)

    tube.align_start_with_direction(direction_vector)

    # Sample curve points and convert to numpy array
    curve_points = tube._sample_bspline_curve()
    curve_points_array = np.array([[point.x, point.y, point.z] for point in curve_points])

    # Generate pipe with scaled parameters
    pipe = spline_to_pipe(curve_points_array, **scaled_pipe_params)

    return tube, pipe


def create_bspline_pipe_from_config(
    morph_config: dict | DictConfig,
    attachment_point_quat: Quaternion,
    pipe_params: dict | None = None,
    scale_factor: float = 1.0,
    link_index: int = 2,
) -> tuple[BSplineTubeWithClamps, pv.PolyData]:
    """
    Create a B-spline tube and convert it to a pipe using MorphConfig from sim.yaml.

    Args:
        morph_config: MorphConfig dictionary or DictConfig from sim.yaml
        attachment_point_quat: Quaternion for alignment (from MuJoCo)
        pipe_params: Optional dictionary for pipe generation
        scale_factor: Scaling factor to convert units to meters (default: 1.0)
        link_index: Index of the link (2 for tube1, 3 for tube2)

    Returns
    -------
        Tuple of (BSplineTubeWithClamps object, PyVista pipe mesh)
    """
    if isinstance(morph_config, DictConfig):
        morph_config = OmegaConf.to_container(morph_config, resolve=True)

    robot_dict = morph_config.get("robot_description_dict", {})
    prefix = f"l{link_index}_"

    # Extract parameters from config
    tube_params = {
        "num_segments": robot_dict.get("num_segments", 100),
        "cylinder_radius": robot_dict.get(f"{prefix}next_joint_radius", 0.042),  # Using next_joint_radius as tube radius
        "mounting_length_start": robot_dict.get("clamp_length", 0.0359),
        "mounting_length_end": robot_dict.get("clamp_length", 0.0359),
        "pre_joint_radius": robot_dict.get(f"{prefix}pre_joint_radius", 0.062),
        "next_joint_radius": robot_dict.get(f"{prefix}next_joint_radius", 0.042),
        "end_point_pos": Vector3(robot_dict.get(f"{prefix}end_point_pos", [0.0, 0.0, 0.36])),
        "end_point_theta": float(robot_dict.get(f"{prefix}end_point_theta", 0.0)),
        "dual_point_distance": float(robot_dict.get("dual_point_distance", 0.15)),
        "name": f"tube_{link_index}",
        "clamp_stl": robot_dict.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl")),
        "count_joint_volumes": False,
    }

    # Default pipe parameters
    default_pipe_params = {
        "outer_radius": tube_params["next_joint_radius"],
        "wall_thickness": 0.01,
        "filename": f"tube_{link_index}.stl",
        "inner_extension": 0.01,
        "visualize": True,
    }

    if pipe_params:
        default_pipe_params.update(pipe_params)

    # Scale pipe parameters
    scaled_pipe_params = {}
    for key, value in default_pipe_params.items():
        if key in {"outer_radius", "wall_thickness", "inner_extension", "cross_radius", "cross_length", "cross_thickness"}:
            scaled_pipe_params[key] = value * scale_factor
        else:
            scaled_pipe_params[key] = value

    # Handle clamp rotation offset
    # Apply 90 degree rotation around Y axis for all clamps
    angle = np.pi / 2
    scaled_pipe_params["clamp_rotation_offset"] = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])

    # Create B-spline tube
    tube = BSplineTubeWithClamps(
        num_segments=tube_params["num_segments"],
        cylinder_radius=tube_params["cylinder_radius"] * scale_factor,
        mounting_length_start=tube_params["mounting_length_start"] * scale_factor,
        mounting_length_end=tube_params["mounting_length_end"] * scale_factor,
        pre_joint_radius=tube_params["pre_joint_radius"] * scale_factor,
        next_joint_radius=tube_params["next_joint_radius"] * scale_factor,
        end_point_pos=tube_params["end_point_pos"] * scale_factor,
        end_point_theta=np.deg2rad(tube_params["end_point_theta"]),
        dual_point_distance=tube_params["dual_point_distance"] * scale_factor,
        name=tube_params["name"],
        clamp_stl=tube_params["clamp_stl"],
        count_joint_volumes=tube_params["count_joint_volumes"],
    )

    # Re-parameterize control points based on end constraints
    tube.set_end_constraints(tube.end_point_pos, tube.end_point_theta, tube.dual_point_distance)

    # Sample curve points and convert to numpy array
    curve_points = tube._sample_bspline_curve()
    curve_points_array = np.array([[point.x, point.y, point.z] for point in curve_points])

    # Generate pipe
    pipe = spline_to_pipe(curve_points_array, **scaled_pipe_params)

    return tube, pipe


def create_bspline_pipe_with_clamps(
    tube_design: dict,
    attachment_point_quat: Quaternion,
    pipe_params: dict | None = None,
    scale_factor: float = 1.0,
) -> tuple[BSplineTube, pv.PolyData]:
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

    Returns
    -------
        Tuple of (BSplineTube object, PyVista pipe mesh)
    """
    # Default pipe parameters
    default_pipe_params = {
        "outer_radius": 0.042,
        "wall_thickness": 0.01,
        "filename": "pipe.stl",
        "inner_extension": 0.01,
        "visualize": True,
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
            point.z * scale_factor,
        ])
        scaled_control_points.append(scaled_point)
    scaled_tube_design["control_points"] = scaled_control_points

    # Scale cylinder radius
    scaled_tube_design["cylinder_radius"] = tube_design["cylinder_radius"] * scale_factor

    # Scale pipe parameters
    scaled_pipe_params = {}
    for key, value in default_pipe_params.items():
        if key in {"outer_radius", "wall_thickness", "inner_extension", "cross_radius", "cross_length", "cross_thickness"}:
            scaled_pipe_params[key] = value * scale_factor
        else:
            scaled_pipe_params[key] = value

    # Create B-spline tube with scaled parameters
    # This is the new version of the tube creation:
    tube = BSplineTubeWithClamps(
        num_segments=scaled_tube_design["num_segments"],
        cylinder_radius=scaled_tube_design["cylinder_radius"],
        mounting_length_start=scaled_tube_design["mounting_length_start"],
        mounting_length_end=scaled_tube_design["mounting_length_end"],
        pre_joint_radius=scaled_tube_design.get("pre_joint_radius", 0.062),
        next_joint_radius=scaled_tube_design.get("next_joint_radius", 0.042),
        end_point_pos=Vector3(scaled_tube_design.get("end_point_pos", [0.0, 0.0, 0.36])),
        end_point_theta=float(scaled_tube_design.get("end_point_theta", 0.0)),
        dual_point_distance=float(scaled_tube_design.get("dual_point_distance", 0.15)),
        name=scaled_tube_design.get("name", "bspline_tube_with_clamps"),
        angle=float(scaled_tube_design.get("angle", 0.0)),
        control_points=scaled_tube_design.get("control_points"),
        count_joint_volumes=scaled_tube_design.get("count_joint_volumes", False),
        color=scaled_tube_design.get("color"),
        clamp_stl=scaled_tube_design.get("clamp_stl", str(CWD / "src/ariel/body_phenotypes/Lynx_Arm/models/clamp_0205.stl")),
        collision_radius=scaled_tube_design.get("collision_radius"),
    )

    # Apply MuJoCo-consistent alignment
    direction_vector = calculate_mujoco_direction_vector(attachment_point_quat)

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
        "name": "straight_tube_m",
        "mounting_length_start": 0.01,
        "mounting_length_end": 0.01,
    },

    "slight_bend_short": {  # Coordinates already in meters
        "control_points": [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.07, 0.0]),
            Vector3([0.0, 0.14, 0.0]),
        ],
        "degree": 2,
        "num_segments": 10,
        "cylinder_radius": 0.0416,
        "name": "straight_tube_m",
        "mounting_length_start": 0.01,
        "mounting_length_end": 0.01,
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
        "name": "right_angle_tube_m",
        "mounting_length_start": 0.01,
        "mounting_length_end": 0.01,
    },

    "straight": {  # Coordinates already in meters
        "control_points": [
            Vector3([0.0, 0.0, 0.0]),
            Vector3([0.0, 0.14 - 0.0035 * 2 - 0.0325 * 2, 0.0]),
        ],
        "degree": 1,
        "num_segments": 5,
        "cylinder_radius": 0.035,
        "name": "straight_tube_m",
        "mounting_length_start": 0.01,
        "mounting_length_end": 0.01,
    },
}


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    attachment_quat = Quaternion([1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)])

    tube1, pipe1 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["slight_bend"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            "filename": "slight_bend_pipe.stl",
            "outer_radius": 0.035,       # 35mm outer radius
            "wall_thickness": 0.005,     # 5mm wall thickness
            "visualize": False,
            "add_cross": True,           # Add cross structure for rigidity
            "cross_thickness": 0.003,    # 3mm thick cross planes
            "add_clamps": True,          # Add clamps at pipe ends
            "clamp_stl_path": "clamp.stl",
            "clamp_scale": 1.0,          # No scaling needed (already in mm)
            # cross_radius defaults to inner radius
            # cross_length defaults to full pipe length
        },
    )

    tube2, pipe2 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["right_angle"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            "filename": "right_angle_pipe.stl",
            "outer_radius": 0.035,       # 35mm outer radius
            "wall_thickness": 0.005,     # 5mm wall thickness
            "visualize": False,
            "add_cross": True,           # Add cross structure for rigidity
            "cross_thickness": 0.003,    # 3mm thick cross planes
            "add_clamps": True,          # Add clamps at pipe ends
            "clamp_stl_path": "clamp.stl",
            "clamp_scale": 1.0,          # No scaling needed (already in mm)
            # cross_radius defaults to inner radius
            # cross_length defaults to full pipe length
        },
    )

    tube3, pipe3 = create_bspline_pipe(
        tube_design=TUBE_DESIGNS["straight"],
        attachment_point_quat=attachment_quat,
        scale_factor=1000.0,  # Scale to metres (No idea why 1000)
        pipe_params={
            "filename": "straight_pipe.stl",
            "outer_radius": 0.035,       # 35mm outer radius
            "wall_thickness": 0.005,     # 5mm wall thickness
            "visualize": False,
            "add_cross": True,           # Add cross structure for rigidity
            "cross_thickness": 0.002,    # 2mm thick cross planes
            "add_clamps": True,          # Add clamps at pipe ends
            "clamp_stl_path": "clamp.stl",
            "clamp_scale": 1.0,          # No scaling needed (already in mm)
            # cross_radius defaults to inner radius
            # cross_length defaults to full pipe length
        },
    )
