"""
Optimization-based repair operator for drone genomes.

This module implements a deterministic optimization-based repair operator that
formulates collision repair as a constrained optimization problem. It minimizes
changes to the original genome while ensuring:
- Cylinders don't collide with each other
- Cylinders don't intersect the central core sphere
- Arms attach to a disc base (not at the origin)
- All spatial constraints are satisfied

The disc-based attachment model:
- Arms attach at the edge of a disc perpendicular to the z-axis
- The disc has radius > core_radius, allowing arms to tilt inward
- The arm base point is computed as the closest point on the disc edge to the target position
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing import Any, Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import minimize, Bounds
import warnings

from ariel.ec.drone.genome_handlers.conversions.arm_conversions import (
    Cylinder,
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_polar_angular,
    arms_to_cylinders_cartesian_euler,
    cylinders_to_arms_cartesian_euler,
)
from .repair_base import RepairOperator, RepairConfig


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationRepairConfig:
    """Configuration for optimization-based repair."""

    # Disc parameters
    disc_radius: float = 0.15      # Radius of disc (> core_radius)
    disc_height: float = 0.0       # Z-coordinate of disc plane

    # Core sphere parameters
    core_radius: float = 0.05      # Radius of central core sphere

    # Optimization parameters
    optimization_method: str = 'SLSQP'  # 'SLSQP' or 'trust-constr'
    max_iterations: int = 1000
    constraint_tolerance: float = 1e-6
    optimization_tolerance: float = 1e-6

    # Multi-start optimization parameters
    use_multi_start: bool = True       # Enable multi-start optimization
    n_starts: int = 3                  # Number of starting points to try
    multi_start_perturbation: float = 0.1  # Perturbation magnitude for additional starts

    # Angle normalization
    normalize_angles: bool = True      # Wrap angles to [-π, π] before optimization

    # Clearance parameters
    propeller_radius: float = 0.0254  # 2-inch propeller radius in meters
    propeller_tolerance: float = 0.1
    cylinder_height: float = None  # Will be set to 8 * propeller_radius in __post_init__

    # Boundary constraints
    inner_boundary_radius: float = 0.09
    outer_boundary_radius: float = 0.4

    # Singularity handling (avoid targets directly above disc center)
    min_xy_projection: float = 0.01

    # Parameter bounds (for spherical coordinates)
    r_bounds: Tuple[float, float] = (0.09, 0.4)
    theta_bounds: Tuple[float, float] = (-np.pi, np.pi)
    phi_bounds: Tuple[float, float] = (-np.pi, np.pi)
    pitch_bounds: Tuple[float, float] = (-2*np.pi/3, 2*np.pi/3)  # ±120° (was ±180°)
    yaw_bounds: Tuple[float, float] = (-np.pi, np.pi)

    # Fixed parameters (parameters that should not be modified during optimization)
    # List of parameter indices to fix: 0=r, 1=theta, 2=phi, 3=pitch, 4=yaw
    # For example, [1, 2] will fix theta and phi
    fixed_params: List[int] = None  # None means all parameters can be optimized

    def __post_init__(self):
        """Validate configuration parameters and set derived values."""
        # Set cylinder_height to 8 * propeller_radius if not explicitly set
        if self.cylinder_height is None:
            object.__setattr__(self, 'cylinder_height', 8 * self.propeller_radius)

        assert self.disc_radius > self.core_radius, \
            "Disc radius must be greater than core radius"
        assert self.core_radius > 0, "Core radius must be positive"
        assert self.propeller_radius > 0, "Propeller radius must be positive"
        assert self.cylinder_height > 0, "Cylinder height must be positive"
        assert self.inner_boundary_radius < self.outer_boundary_radius, \
            "Inner boundary must be less than outer boundary"
        assert self.n_starts >= 1, "Must have at least one starting point"
        assert 0.0 <= self.multi_start_perturbation <= 1.0, \
            "Perturbation magnitude must be in [0, 1]"


# ============================================================================
# GEOMETRIC UTILITY FUNCTIONS
# ============================================================================

def compute_base_point(
    target_position: npt.NDArray[Any],
    disc_radius: float,
    disc_height: float,
    min_xy_projection: float = 0.01
) -> npt.NDArray[Any]:
    """
    Compute the base attachment point on the disc edge for a given target position.

    The base point is the point on the disc edge (circle at z=disc_height with
    radius=disc_radius) that is closest to the target position in the xy-plane.

    Parameters:
    -----------
    target_position : array-like of shape (3,)
        Target position [x, y, z] from genome
    disc_radius : float
        Radius of the disc
    disc_height : float
        Z-coordinate of the disc plane
    min_xy_projection : float
        Minimum xy-projection to avoid singularity

    Returns:
    --------
    base_point : ndarray of shape (3,)
        Base attachment point [x, y, z] on disc edge
    """
    x, y, z = target_position
    r_xy = np.sqrt(x**2 + y**2)

    # Handle singularity: target directly above/below disc center
    if r_xy < min_xy_projection:
        # Default to arbitrary point on disc edge (along x-axis)
        return np.array([disc_radius, 0.0, disc_height])

    # Compute base point on disc edge
    base_x = disc_radius * (x / r_xy)
    base_y = disc_radius * (y / r_xy)
    base_z = disc_height

    return np.array([base_x, base_y, base_z])


def closest_point_on_line_segment_to_point(
    p1: npt.NDArray[Any],
    p2: npt.NDArray[Any],
    point: npt.NDArray[Any]
) -> Tuple[npt.NDArray[Any], float]:
    """
    Find the closest point on a line segment to a given point.

    Parameters:
    -----------
    p1 : array-like of shape (3,)
        First endpoint of line segment
    p2 : array-like of shape (3,)
        Second endpoint of line segment
    point : array-like of shape (3,)
        Query point

    Returns:
    --------
    closest_point : ndarray of shape (3,)
        Closest point on line segment to query point
    t : float
        Parameter t ∈ [0, 1] where closest_point = p1 + t*(p2-p1)
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    point = np.asarray(point)

    segment_vec = p2 - p1
    segment_length_sq = np.dot(segment_vec, segment_vec)

    # Handle degenerate case: p1 == p2
    if segment_length_sq < 1e-12:
        return p1.copy(), 0.0

    # Project point onto infinite line
    t = np.dot(point - p1, segment_vec) / segment_length_sq

    # Clamp to segment
    t = np.clip(t, 0.0, 1.0)

    closest_point = p1 + t * segment_vec

    return closest_point, t


def line_segment_to_point_distance(
    p1: npt.NDArray[Any],
    p2: npt.NDArray[Any],
    point: npt.NDArray[Any]
) -> float:
    """
    Compute minimum distance from a line segment to a point.

    Parameters:
    -----------
    p1, p2 : array-like of shape (3,)
        Endpoints of line segment
    point : array-like of shape (3,)
        Query point

    Returns:
    --------
    distance : float
        Minimum distance from line segment to point
    """
    closest_point, _ = closest_point_on_line_segment_to_point(p1, p2, point)
    return np.linalg.norm(closest_point - point)


def line_segment_to_sphere_distance(
    p1: npt.NDArray[Any],
    p2: npt.NDArray[Any],
    sphere_center: npt.NDArray[Any],
    sphere_radius: float
) -> float:
    """
    Compute the signed distance from a line segment to a sphere surface.

    Returns:
    --------
    distance : float
        Signed distance to sphere surface:
        - Positive: line segment is outside sphere (no collision)
        - Zero: line segment touches sphere surface
        - Negative: line segment intersects sphere (collision)
    """
    min_dist_to_center = line_segment_to_point_distance(p1, p2, sphere_center)
    return min_dist_to_center - sphere_radius


def line_segment_to_line_segment_distance(
    p1: npt.NDArray[Any],
    p2: npt.NDArray[Any],
    q1: npt.NDArray[Any],
    q2: npt.NDArray[Any]
) -> float:
    """
    Compute minimum distance between two line segments in 3D.

    Uses the algorithm from:
    http://geomalgorithms.com/a07-_distance.html

    Parameters:
    -----------
    p1, p2 : array-like of shape (3,)
        Endpoints of first line segment
    q1, q2 : array-like of shape (3,)
        Endpoints of second line segment

    Returns:
    --------
    distance : float
        Minimum distance between the two line segments
    """
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)

    u = p2 - p1  # Direction of segment 1
    v = q2 - q1  # Direction of segment 2
    w = p1 - q1  # Vector from q1 to p1

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w)
    e = np.dot(v, w)

    denominator = a * c - b * b  # Always >= 0

    # Handle parallel/degenerate segments
    if denominator < 1e-12:
        # Segments are parallel or degenerate
        # Use distance from p1 to segment q1-q2
        s = 0.0
        t = (b > c) and (d / b) or (e / c) if c > 1e-12 else 0.0
        t = np.clip(t, 0.0, 1.0)
    else:
        # General case: compute closest points
        s = (b * e - c * d) / denominator
        t = (a * e - b * d) / denominator

        # Clamp s and t to [0, 1]
        s = np.clip(s, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)

        # Re-clamp after first clamping (handles edge cases)
        if s == 0.0 or s == 1.0:
            t = np.clip((b * s + e) / c, 0.0, 1.0) if c > 1e-12 else 0.0
        if t == 0.0 or t == 1.0:
            s = np.clip((b * t - d) / a, 0.0, 1.0) if a > 1e-12 else 0.0

    # Compute closest points
    closest_p = p1 + s * u
    closest_q = q1 + t * v

    return np.linalg.norm(closest_p - closest_q)


def cylinder_to_cylinder_distance(
    cyl1_p1: npt.NDArray[Any],
    cyl1_p2: npt.NDArray[Any],
    cyl1_radius: float,
    cyl2_p1: npt.NDArray[Any],
    cyl2_p2: npt.NDArray[Any],
    cyl2_radius: float
) -> float:
    """
    Compute the minimum distance between two cylinders (as capsules).

    Parameters:
    -----------
    cyl1_p1, cyl1_p2 : array-like of shape (3,)
        Endpoints of first cylinder axis
    cyl1_radius : float
        Radius of first cylinder
    cyl2_p1, cyl2_p2 : array-like of shape (3,)
        Endpoints of second cylinder axis
    cyl2_radius : float
        Radius of second cylinder

    Returns:
    --------
    distance : float
        Minimum surface-to-surface distance between cylinders
        (positive = no collision, zero/negative = collision)
    """
    axis_distance = line_segment_to_line_segment_distance(
        cyl1_p1, cyl1_p2, cyl2_p1, cyl2_p2
    )
    return axis_distance - (cyl1_radius + cyl2_radius)


# ============================================================================
# CONVERSION FUNCTIONS WITH DISC BASE
# ============================================================================

def genome_to_arm_cylinders_with_disc_base(
    genome: npt.NDArray[Any],
    config: OptimizationRepairConfig,
    arms_to_cylinders_func: Callable = arms_to_cylinders_polar_angular
) -> List[Tuple[npt.NDArray[Any], npt.NDArray[Any], Cylinder]]:
    """
    Convert genome to arm cylinder representations with disc-based attachment.

    Parameters:
    -----------
    genome : array-like of shape (n_arms, 6)
        Genome with parameters [r, theta, phi, pitch, yaw, direction]
    config : OptimizationRepairConfig
        Configuration with disc and cylinder parameters
    arms_to_cylinders_func : callable
        Function to convert genome to cylinder target positions

    Returns:
    --------
    arm_cylinders : list of tuples
        Each tuple contains (endpoint1, endpoint2, cylinder)
        where endpoint1 and endpoint2 are the actual cylinder axis endpoints
    """
    # Convert genome to cylinders (to get positions and orientations)
    cylinders = arms_to_cylinders_func(
        genome,
        propeller_radius=config.propeller_radius,
        cylinder_height=config.cylinder_height
    )

    arm_cylinders = []

    for cyl in cylinders:
        # Get the actual cylinder endpoints
        endpoint1, endpoint2 = cyl.get_endpoints()
        arm_cylinders.append((endpoint1, endpoint2, cyl))

    return arm_cylinders


# ============================================================================
# GEOMETRY CACHE FOR CONSTRAINT EVALUATIONS
# ============================================================================

class GeometryCache:
    """
    Cache for arm cylinder geometry to avoid redundant conversions.

    During optimization, many constraint functions need the same geometry.
    This cache ensures we only compute it once per unique parameter vector.
    """

    def __init__(self):
        self._cache_key = None
        self._arm_cylinders = None
        self._hit_count = 0
        self._miss_count = 0

    def get_arm_cylinders(
        self,
        x_flat: npt.NDArray[Any],
        config: OptimizationRepairConfig,
        arms_to_cylinders_func: Callable,
        n_params: int,
        direction_column: npt.NDArray[Any]
    ) -> List[Tuple[npt.NDArray[Any], npt.NDArray[Any], Cylinder]]:
        """
        Get arm cylinders for given parameters, using cache when possible.

        Parameters:
        -----------
        x_flat : array
            Flattened parameter vector
        config : OptimizationRepairConfig
            Configuration
        arms_to_cylinders_func : callable
            Conversion function
        n_params : int
            Number of parameters per arm (5 or 6)
        direction_column : array
            Direction values for each arm

        Returns:
        --------
        arm_cylinders : list
            List of (endpoint1, endpoint2, cylinder) tuples
        """
        # Create cache key from parameter vector
        # Using tobytes() is faster than hash for small arrays
        key = x_flat.tobytes()

        if key == self._cache_key:
            self._hit_count += 1
            return self._arm_cylinders

        # Cache miss - compute geometry
        self._miss_count += 1
        n_arms = len(x_flat) // n_params
        genome = x_flat.reshape(n_arms, n_params)

        # Add direction column
        genome_with_dir = np.column_stack([genome, direction_column])

        # Get arm cylinders with disc base
        self._arm_cylinders = genome_to_arm_cylinders_with_disc_base(
            genome_with_dir, config, arms_to_cylinders_func
        )

        self._cache_key = key
        return self._arm_cylinders

    def get_stats(self) -> dict:
        """Get cache statistics."""
        total = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total if total > 0 else 0.0
        return {
            'hits': self._hit_count,
            'misses': self._miss_count,
            'total_requests': total,
            'hit_rate': hit_rate
        }

    def reset(self):
        """Reset cache and statistics."""
        self._cache_key = None
        self._arm_cylinders = None
        self._hit_count = 0
        self._miss_count = 0


# ============================================================================
# CONSTRAINT FUNCTIONS FOR OPTIMIZATION
# ============================================================================

def create_cylinder_cylinder_constraint(
    i: int,
    j: int,
    config: OptimizationRepairConfig,
    arms_to_cylinders_func: Callable,
    geometry_cache: GeometryCache,
    n_params: int,
    direction_column: npt.NDArray[Any]
) -> Callable:
    """
    Create a constraint function for non-collision between two cylinders.

    Parameters:
    -----------
    i, j : int
        Indices of the two arms
    config : OptimizationRepairConfig
        Repair configuration
    arms_to_cylinders_func : callable
        Function to convert genome parameters to cylinders
    geometry_cache : GeometryCache
        Shared cache for geometry computations
    n_params : int
        Number of parameters per arm (5 or 6)
    direction_column : array
        Direction values for each arm

    Returns:
    --------
    constraint_func : callable
        Function that takes flattened genome and returns signed distance
        (positive = no collision, zero/negative = collision)
    """
    required_clearance = (2 + config.propeller_tolerance) * config.propeller_radius

    def constraint(x_flat: npt.NDArray[Any]) -> float:
        # Get arm cylinders from cache (avoiding redundant computation)
        arm_cylinders = geometry_cache.get_arm_cylinders(
            x_flat, config, arms_to_cylinders_func, n_params, direction_column
        )

        # Get the two arms
        base_i, target_i, _ = arm_cylinders[i]
        base_j, target_j, _ = arm_cylinders[j]

        # Compute cylinder-to-cylinder distance
        distance = cylinder_to_cylinder_distance(
            base_i, target_i, config.propeller_radius,
            base_j, target_j, config.propeller_radius
        )

        # Return signed distance (should be >= required_clearance)
        return distance - required_clearance

    return constraint


def create_cylinder_sphere_constraint(
    arm_idx: int,
    config: OptimizationRepairConfig,
    arms_to_cylinders_func: Callable,
    geometry_cache: GeometryCache,
    n_params: int,
    direction_column: npt.NDArray[Any]
) -> Callable:
    """
    Create a constraint function for non-collision between arm cylinder and core sphere.

    Parameters:
    -----------
    arm_idx : int
        Index of the arm
    config : OptimizationRepairConfig
        Repair configuration
    arms_to_cylinders_func : callable
        Function to convert genome parameters to cylinders
    geometry_cache : GeometryCache
        Shared cache for geometry computations
    n_params : int
        Number of parameters per arm (5 or 6)
    direction_column : array
        Direction values for each arm

    Returns:
    --------
    constraint_func : callable
        Function that takes flattened genome and returns signed distance
    """
    sphere_center = np.array([0.0, 0.0, 0.0])
    required_clearance = config.propeller_radius

    def constraint(x_flat: npt.NDArray[Any]) -> float:
        # Get arm cylinders from cache (avoiding redundant computation)
        arm_cylinders = geometry_cache.get_arm_cylinders(
            x_flat, config, arms_to_cylinders_func, n_params, direction_column
        )

        # Get the arm
        base_point, target_point, _ = arm_cylinders[arm_idx]

        # Compute line segment to sphere distance
        distance = line_segment_to_sphere_distance(
            base_point, target_point, sphere_center, config.core_radius
        )

        # Return signed distance (should be >= required_clearance)
        return distance - required_clearance

    return constraint


def create_xy_projection_constraint(
    arm_idx: int,
    param_start_idx: int,
    min_xy_projection: float
) -> Callable:
    """
    Create a constraint to avoid singularity (target directly above disc center).

    This constraint ensures r_xy = sqrt(x^2 + y^2) >= min_xy_projection
    for the target position.
    """
    def constraint(x_flat: npt.NDArray[Any]) -> float:
        # Extract r, theta parameters for this arm
        r = x_flat[param_start_idx]
        theta = x_flat[param_start_idx + 1]
        phi = x_flat[param_start_idx + 2]

        # Convert to Cartesian for xy-projection check
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        r_xy = np.sqrt(x**2 + y**2)

        # Return r_xy - min_xy_projection (should be >= 0)
        return r_xy - min_xy_projection

    return constraint


# ============================================================================
# ANGLE NORMALIZATION AND MULTI-START HELPERS
# ============================================================================

def normalize_angles(genome: npt.NDArray[Any], angle_indices: List[int] = [1, 2, 3, 4]) -> npt.NDArray[Any]:
    """
    Normalize angles to [-π, π] range.

    Parameters:
    -----------
    genome : array-like of shape (n_arms, 5)
        Genome parameters [r, theta, phi, pitch, yaw]
    angle_indices : list of int
        Indices of angular parameters (default: [1, 2, 3, 4] for theta, phi, pitch, yaw)

    Returns:
    --------
    normalized_genome : array-like of shape (n_arms, 5)
        Genome with normalized angles
    """
    genome_normalized = genome.copy()

    for idx in angle_indices:
        # Wrap angles to [-π, π]
        genome_normalized[:, idx] = np.arctan2(
            np.sin(genome[:, idx]),
            np.cos(genome[:, idx])
        )

    return genome_normalized


def generate_starting_points(
    x0: npt.NDArray[Any],
    n_starts: int,
    perturbation: float,
    bounds: Bounds,
    rng: Optional[np.random.Generator] = None
) -> List[npt.NDArray[Any]]:
    """
    Generate multiple starting points for multi-start optimization.

    Parameters:
    -----------
    x0 : array-like
        Initial starting point (flattened genome)
    n_starts : int
        Total number of starting points to generate
    perturbation : float
        Magnitude of perturbation (as fraction of parameter range)
    bounds : Bounds
        Parameter bounds
    rng : np.random.Generator, optional
        Random number generator

    Returns:
    --------
    starting_points : list of arrays
        List of starting points (first is always x0)
    """
    if rng is None:
        rng = np.random.default_rng()

    starting_points = [x0.copy()]

    if n_starts <= 1:
        return starting_points

    # Generate additional starting points with perturbations
    for _ in range(n_starts - 1):
        # Compute parameter ranges
        param_ranges = np.array(bounds.ub) - np.array(bounds.lb)

        # Generate random perturbation
        perturbation_vec = rng.uniform(-1, 1, size=x0.shape) * perturbation * param_ranges

        # Apply perturbation and clip to bounds
        x_perturbed = x0 + perturbation_vec
        x_perturbed = np.clip(x_perturbed, bounds.lb, bounds.ub)

        starting_points.append(x_perturbed)

    return starting_points


def run_optimization(
    objective: Callable,
    objective_grad: Callable,
    x0: npt.NDArray[Any],
    bounds: Bounds,
    constraints: List[Dict],
    config: OptimizationRepairConfig,
    verbose: bool = False
) -> Any:
    """
    Run single optimization from a starting point.

    Returns:
    --------
    result : OptimizeResult
        Optimization result from scipy.optimize.minimize
    """
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        result = minimize(
            objective,
            x0,
            method=config.optimization_method,
            jac=objective_grad,
            bounds=bounds,
            constraints=constraints,
            options={
                'maxiter': config.max_iterations,
                'ftol': config.optimization_tolerance,
                'disp': verbose
            }
        )

    return result


# ============================================================================
# MAIN OPTIMIZATION REPAIR FUNCTION
# ============================================================================

def optimization_repair_individual(
    individual: npt.NDArray[Any],
    config: OptimizationRepairConfig,
    arms_to_cylinders: Callable = arms_to_cylinders_polar_angular,
    cylinders_to_arms: Callable = cylinders_to_arms_polar_angular,
    verbose: bool = False
) -> npt.NDArray[Any]:
    """
    Repair an individual genome using constrained optimization.

    This function formulates the repair problem as:

    minimize: ||genome_repaired - genome_original||^2
    subject to:
        - No cylinder-cylinder collisions
        - No cylinder-core sphere intersections
        - All parameters within bounds
        - xy-projection constraints (avoid singularity)

    Parameters:
    -----------
    individual : array-like of shape (n_arms, 6)
        Genome representation with [r, theta, phi, pitch, yaw, direction]
    config : OptimizationRepairConfig
        Configuration parameters
    arms_to_cylinders : callable
        Function to convert genome to cylinders
    cylinders_to_arms : callable
        Function to convert cylinders back to genome
    verbose : bool
        Print optimization progress

    Returns:
    --------
    repaired_individual : array-like of shape (n_arms, 6)
        Repaired genome
    """
    # Filter valid arms
    valid_arms = ~np.isnan(individual).any(axis=-1)
    n_valid_arms = np.sum(valid_arms)

    if n_valid_arms == 0:
        return individual.copy()

    valid_genome = individual[valid_arms].copy()

    # Step 1: Normalize angles if enabled
    if config.normalize_angles:
        valid_genome_params = valid_genome[:, :-1]  # Exclude direction
        valid_genome_params_normalized = normalize_angles(valid_genome_params)
        valid_genome[:, :-1] = valid_genome_params_normalized

    # Check if repair is needed
    arm_cylinders = genome_to_arm_cylinders_with_disc_base(
        valid_genome, config, arms_to_cylinders
    )

    if not _check_collisions(arm_cylinders, config):
        # No collisions, return original
        return individual.copy()

    # Extract parameters (exclude direction column)
    x0 = valid_genome[:, :-1].flatten()  # Shape: (n_arms * n_params,)

    # Define parameter bounds (pass x0 to support fixed parameters)
    bounds = _create_bounds(n_valid_arms, config, arms_to_cylinders, x0)

    # Get bounds for normalization
    bounds_lb = np.array(bounds.lb)
    bounds_ub = np.array(bounds.ub)
    bounds_range = bounds_ub - bounds_lb

    # Avoid division by zero for any fixed parameters
    bounds_range = np.where(bounds_range == 0, 1.0, bounds_range)

    # Define objective function: minimize normalized changes for equal weighting
    def objective(x):
        # Normalize the differences to [0,1] scale for equal weighting
        normalized_diff = (x - x0) / bounds_range
        return np.sum(normalized_diff**2)

    def objective_grad(x):
        # Gradient of normalized objective
        normalized_diff = (x - x0) / bounds_range
        return 2 * normalized_diff / bounds_range

    # Determine coordinate system
    is_cartesian = (arms_to_cylinders == arms_to_cylinders_cartesian_euler)
    n_params = 6 if is_cartesian else 5

    # Create shared geometry cache for all constraints
    geometry_cache = GeometryCache()
    direction_column = valid_genome[:, -1]  # Extract direction column for cache

    # Define constraints
    constraints = []

    # Cylinder-cylinder collision constraints
    for i in range(n_valid_arms):
        for j in range(i + 1, n_valid_arms):
            constraint_func = create_cylinder_cylinder_constraint(
                i, j, config, arms_to_cylinders, geometry_cache, n_params, direction_column
            )
            constraints.append({
                'type': 'ineq',
                'fun': constraint_func
            })

    # Cylinder-sphere collision constraints
    for i in range(n_valid_arms):
        constraint_func = create_cylinder_sphere_constraint(
            i, config, arms_to_cylinders, geometry_cache, n_params, direction_column
        )
        constraints.append({
            'type': 'ineq',
            'fun': constraint_func
        })

    # XY-projection constraints (avoid singularity) - only for spherical coordinates
    if not is_cartesian:
        for i in range(n_valid_arms):
            constraint_func = create_xy_projection_constraint(
                i, i * n_params, config.min_xy_projection
            )
            constraints.append({
                'type': 'ineq',
                'fun': constraint_func
            })

    # Step 2: Multi-start optimization
    if config.use_multi_start and config.n_starts > 1:
        # Generate multiple starting points
        starting_points = generate_starting_points(
            x0, config.n_starts, config.multi_start_perturbation, bounds
        )

        best_result = None
        best_objective = float('inf')
        best_feasible_result = None
        best_feasible_objective = float('inf')

        if verbose:
            print(f"\n=== Multi-start optimization with {config.n_starts} starting points ===")

        for i, start_point in enumerate(starting_points):
            result = run_optimization(
                objective, objective_grad, start_point,
                bounds, constraints, config, verbose=False
            )

            # Check if result is feasible (satisfies constraints)
            x_test = result.x.reshape(n_valid_arms, n_params)
            genome_test = np.column_stack([x_test, valid_genome[:, -1]])
            arm_cylinders_test = genome_to_arm_cylinders_with_disc_base(
                genome_test, config, arms_to_cylinders
            )
            is_feasible = not _check_collisions(arm_cylinders_test, config)

            if verbose:
                feasible_str = "✓ feasible" if is_feasible else "✗ infeasible"
                print(f"  Start {i+1}/{config.n_starts}: obj={result.fun:.6e} {feasible_str}")

            # Track best overall result
            if result.fun < best_objective:
                best_objective = result.fun
                best_result = result

            # Track best feasible result
            if is_feasible and result.fun < best_feasible_objective:
                best_feasible_objective = result.fun
                best_feasible_result = result

        # Prefer feasible solution if available
        if best_feasible_result is not None:
            result = best_feasible_result
            if verbose:
                print(f"  → Selected feasible solution with obj={result.fun:.6e}")
        else:
            result = best_result
            if verbose:
                print(f"  → No feasible solution found, using best infeasible with obj={result.fun:.6e}")

    else:
        # Single-start optimization
        result = run_optimization(
            objective, objective_grad, x0,
            bounds, constraints, config, verbose=verbose
        )

    if verbose:
        print(f"\nFinal optimization {'succeeded' if result.success else 'failed'}: {result.message}")
        print(f"Function value: {result.fun:.6e}")
        print(f"Constraint violations: {np.sum(result.constr_violation) if hasattr(result, 'constr_violation') else 'N/A'}")

    # Extract result
    is_cartesian = (arms_to_cylinders == arms_to_cylinders_cartesian_euler)
    n_params = 6 if is_cartesian else 5
    x_opt = result.x.reshape(n_valid_arms, n_params)

    # Reconstruct genome with direction column
    repaired_genome = np.column_stack([x_opt, valid_genome[:, -1]])

    # Insert back into full genome
    result_individual = individual.copy()
    result_individual[valid_arms] = repaired_genome

    return result_individual


def _create_bounds(n_arms: int, config: OptimizationRepairConfig, arms_to_cylinders_func: Callable, x0: npt.NDArray[Any] = None) -> Bounds:
    """Create bounds for optimization variables.

    Parameters:
    -----------
    n_arms : int
        Number of arms
    config : OptimizationRepairConfig
        Configuration with parameter bounds
    arms_to_cylinders_func : callable
        Function to convert genome to cylinders (determines coordinate system)
    x0 : array, optional
        Initial parameter values (needed if config.fixed_params is set)

    Returns:
    --------
    bounds : Bounds
        Parameter bounds for optimization
    """
    lower = []
    upper = []

    # Determine coordinate system based on conversion function
    is_cartesian = (arms_to_cylinders_func == arms_to_cylinders_cartesian_euler)

    for arm_idx in range(n_arms):
        if is_cartesian:
            # Cartesian: x, y, z, roll, pitch, yaw
            arm_lower = [
                -config.outer_boundary_radius,  # x
                -config.outer_boundary_radius,  # y
                -config.outer_boundary_radius,  # z
                -np.pi,  # roll
                config.pitch_bounds[0],  # pitch
                config.yaw_bounds[0]  # yaw
            ]
            arm_upper = [
                config.outer_boundary_radius,  # x
                config.outer_boundary_radius,  # y
                config.outer_boundary_radius,  # z
                np.pi,  # roll
                config.pitch_bounds[1],  # pitch
                config.yaw_bounds[1]  # yaw
            ]
        else:
            # Spherical: r, theta, phi, pitch, yaw
            arm_lower = [
                config.r_bounds[0],       # 0: r
                config.theta_bounds[0],   # 1: theta
                config.phi_bounds[0],     # 2: phi
                config.pitch_bounds[0],   # 3: pitch
                config.yaw_bounds[0]      # 4: yaw
            ]
            arm_upper = [
                config.r_bounds[1],       # 0: r
                config.theta_bounds[1],   # 1: theta
                config.phi_bounds[1],     # 2: phi
                config.pitch_bounds[1],   # 3: pitch
                config.yaw_bounds[1]      # 4: yaw
            ]

        # Fix parameters if specified
        if config.fixed_params is not None and x0 is not None:
            n_params = 6 if is_cartesian else 5
            param_start_idx = arm_idx * n_params

            for param_idx in config.fixed_params:
                if param_idx < len(arm_lower):
                    # Fix this parameter at its initial value
                    initial_value = x0[param_start_idx + param_idx]
                    arm_lower[param_idx] = initial_value
                    arm_upper[param_idx] = initial_value

        lower.extend(arm_lower)
        upper.extend(arm_upper)

    return Bounds(lower, upper)


def _check_collisions(
    arm_cylinders: List[Tuple[npt.NDArray[Any], npt.NDArray[Any], Cylinder]],
    config: OptimizationRepairConfig,
    tolerance: float = 1e-5
) -> bool:
    """Check if there are any collisions in the current configuration.

    Parameters:
    -----------
    arm_cylinders : list of tuples
        List of (endpoint1, endpoint2, cylinder) tuples
    config : OptimizationRepairConfig
        Configuration parameters
    tolerance : float
        Numerical tolerance for collision detection (in meters).
        Distances within this tolerance are considered valid.

    Returns:
    --------
    has_collision : bool
        True if there are collisions, False otherwise
    """
    n_arms = len(arm_cylinders)
    required_clearance = (2 + config.propeller_tolerance) * config.propeller_radius
    sphere_center = np.array([0.0, 0.0, 0.0])

    # Check cylinder-cylinder collisions
    for i in range(n_arms):
        for j in range(i + 1, n_arms):
            ep1_i, ep2_i, _ = arm_cylinders[i]
            ep1_j, ep2_j, _ = arm_cylinders[j]

            distance = cylinder_to_cylinder_distance(
                ep1_i, ep2_i, config.propeller_radius,
                ep1_j, ep2_j, config.propeller_radius
            )

            # Use tolerance to account for numerical precision
            if distance < required_clearance - tolerance:
                return True

    # Check cylinder-sphere collisions
    for i in range(n_arms):
        ep1, ep2, _ = arm_cylinders[i]

        distance = line_segment_to_sphere_distance(
            ep1, ep2, sphere_center, config.core_radius
        )

        # Use tolerance to account for numerical precision
        if distance < config.propeller_radius - tolerance:
            return True

    return False


# ============================================================================
# REPAIR OPERATOR CLASS (INTEGRATION WITH BASE)
# ============================================================================

class OptimizationBasedRepairOperator(RepairOperator):
    """
    Optimization-based repair operator that integrates with the RepairOperator framework.

    This repair operator uses constrained optimization to repair genomes while:
    - Minimizing changes to the original genome
    - Ensuring no cylinder-cylinder collisions
    - Ensuring no cylinder-core sphere intersections
    - Respecting disc-based arm attachment geometry
    """

    def __init__(
        self,
        config: Optional[RepairConfig] = None,
        optimization_config: Optional[OptimizationRepairConfig] = None,
        coordinate_system: str = 'spherical',  # 'spherical' or 'cartesian'
        verbose: bool = False
    ):
        """
        Initialize the optimization-based repair operator.

        Parameters:
        -----------
        config : RepairConfig, optional
            Standard repair configuration (for compatibility)
        optimization_config : OptimizationRepairConfig, optional
            Optimization-specific configuration
        coordinate_system : str
            'spherical' for polar angular or 'cartesian' for Cartesian Euler
        verbose : bool
            Print optimization progress
        """
        super().__init__(config)

        # Create optimization config, inheriting values from base config if available
        if optimization_config is None:
            self.optimization_config = OptimizationRepairConfig(
                propeller_radius=self.config.propeller_radius,
                propeller_tolerance=self.config.propeller_tolerance,
                inner_boundary_radius=self.config.inner_boundary_radius,
                outer_boundary_radius=self.config.outer_boundary_radius,
            )
        else:
            self.optimization_config = optimization_config

        self.coordinate_system = coordinate_system
        self.verbose = verbose

        # Select appropriate conversion functions
        if coordinate_system == 'spherical':
            self.arms_to_cylinders = arms_to_cylinders_polar_angular
            self.cylinders_to_arms = cylinders_to_arms_polar_angular
            self.param_count = 5  # r, theta, phi, pitch, yaw (excluding direction)
        elif coordinate_system == 'cartesian':
            self.arms_to_cylinders = arms_to_cylinders_cartesian_euler
            self.cylinders_to_arms = cylinders_to_arms_cartesian_euler
            self.param_count = 6  # x, y, z, roll, pitch, yaw (excluding direction)
        else:
            raise ValueError(f"Unknown coordinate system: {coordinate_system}")

    def repair(self, genome: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """
        Repair a genome using optimization-based approach.

        Parameters:
        -----------
        genome : array-like
            Genome to repair

        Returns:
        --------
        repaired_genome : array-like
            Repaired genome
        """
        return optimization_repair_individual(
            genome,
            self.optimization_config,
            self.arms_to_cylinders,
            self.cylinders_to_arms,
            self.verbose
        )

    def validate(self, genome: npt.NDArray[Any]) -> bool:
        """
        Validate a genome (check for collisions and constraint violations).

        Parameters:
        -----------
        genome : array-like
            Genome to validate

        Returns:
        --------
        is_valid : bool
            True if genome has no collisions or violations
        """
        # Filter valid arms
        valid_arms = ~np.isnan(genome).any(axis=-1)
        n_valid_arms = np.sum(valid_arms)

        if n_valid_arms == 0:
            return True

        valid_genome = genome[valid_arms]

        # Get arm cylinders
        arm_cylinders = genome_to_arm_cylinders_with_disc_base(
            valid_genome,
            self.optimization_config,
            self.arms_to_cylinders
        )

        # Check for collisions
        return not _check_collisions(arm_cylinders, self.optimization_config)

    def get_bounds(self) -> npt.NDArray[Any]:
        """
        Get parameter bounds.

        Returns:
        --------
        bounds : array-like of shape (n_params, 2)
            Parameter bounds [min, max]
        """
        cfg = self.optimization_config

        if self.coordinate_system == 'spherical':
            return np.array([
                [cfg.r_bounds[0], cfg.r_bounds[1]],
                [cfg.theta_bounds[0], cfg.theta_bounds[1]],
                [cfg.phi_bounds[0], cfg.phi_bounds[1]],
                [cfg.pitch_bounds[0], cfg.pitch_bounds[1]],
                [cfg.yaw_bounds[0], cfg.yaw_bounds[1]],
                [0, 1]  # direction
            ])
        else:  # cartesian
            return np.array([
                [-cfg.outer_boundary_radius, cfg.outer_boundary_radius],  # x
                [-cfg.outer_boundary_radius, cfg.outer_boundary_radius],  # y
                [-cfg.outer_boundary_radius, cfg.outer_boundary_radius],  # z
                [-np.pi, np.pi],  # roll
                [cfg.pitch_bounds[0], cfg.pitch_bounds[1]],
                [cfg.yaw_bounds[0], cfg.yaw_bounds[1]],
                [0, 1]  # direction
            ])

    def __repr__(self) -> str:
        return (f"OptimizationBasedRepairOperator("
                f"system={self.coordinate_system}, "
                f"method={self.optimization_config.optimization_method})")
