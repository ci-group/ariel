import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional, Union, Dict, Any, Tuple, List
import numpy.typing as npt
from dataclasses import dataclass

from ariel.ec.drone.genome_handlers.conversions.conversions import (
    spherical_to_cartesian,
    cartesian_to_spherical,
    euler_to_quaternion,
    quaternion_to_euler,
)

from ariel.ec.drone.inspection.utils import (
    create_rotation_matrix_quaternion
)

@dataclass(frozen=True)
class Cylinder:
    """Immutable cylinder representation for visualization and collision detection."""
    
    radius: float
    height: float
    position: npt.NDArray[Any]
    orientation: npt.NDArray[Any] # Quaternion [qw, qx, qy, qz]
    
    def __post_init__(self):
        """Initialize computed properties after dataclass initialization."""
        # Convert to numpy arrays and freeze them
        object.__setattr__(self, 'position', np.array(self.position, dtype=float))
        object.__setattr__(self, 'orientation', np.array(self.orientation, dtype=float))
        object.__setattr__(self, 'radius', float(self.radius))
        object.__setattr__(self, 'height', float(self.height))
        
        # Create computed properties
        rotation_matrix = create_rotation_matrix_quaternion(self.orientation)
        transform = np.eye(4)
        transform[:3, 3] = self.position
        transform[:3, :3] = rotation_matrix
        
        object.__setattr__(self, 'rotation_matrix', rotation_matrix)
        object.__setattr__(self, 'transform', transform)

    def get_endpoints(self) -> Tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        """Get the two endpoints of the cylinder axis.
        
        Returns:
            Tuple of (endpoint1, endpoint2) arrays
        """
        half_height = self.height / 2
        axis_vector = np.array([0, 0, half_height])
        endpoint1 = self.rotation_matrix @ axis_vector + self.position
        endpoint2 = self.rotation_matrix @ -axis_vector + self.position
        return endpoint1, endpoint2

    def sample_points(self, num_points: int) -> npt.NDArray[Any]:
        """Sample points along the cylinder axis.
        
        Args:
            num_points: Number of points to sample
            
        Returns:
            Array of sampled points
        """
        endpoint1, endpoint2 = self.get_endpoints()
        points = np.linspace(endpoint1, endpoint2, num_points)
        return points
    
    def with_position(self, new_position: npt.NDArray[Any]) -> 'Cylinder':
        """Create a new Cylinder instance with modified position.
        
        Args:
            new_position: New position [x, y, z]
            
        Returns:
            New Cylinder instance with updated position
        """
        return Cylinder(
            radius=self.radius,
            height=self.height,
            position=new_position,
            orientation=self.orientation
        )
    
    def with_orientation(self, new_orientation: npt.NDArray[Any]) -> 'Cylinder':
        """Create a new Cylinder instance with modified orientation.
        
        Args:
            new_orientation: New orientation quaternion [qx, qy, qz, qw]
            
        Returns:
            New Cylinder instance with updated orientation
        """
        return Cylinder(
            radius=self.radius,
            height=self.height,
            position=self.position,
            orientation=new_orientation
        )
    
    def __repr__(self):
        return (f"Cylinder(radius={self.radius}, height={self.height}, "
                f"position={self.position.tolist()}, orientation={self.orientation.tolist()})")

def spherical_angular_arms_to_cylinders(individual, propeller_radius, cylinder_height):
    """
    Convert arms to cylinder representations using vectorized operations.
    
    Parameters:
    - individual: numpy array of shape (N, 5) with columns [r, theta, phi, pitch, yaw]
    - propeller_radius: float, radius of the propeller/cylinder
    - cylinder_height: float, height of the cylinder
    
    Returns:
    - List of cylinder dictionaries with keys: 'position', 'radius', 'height', 'orientation'
    """
    cylinders = []

    # Extract spherical coordinates (r, theta, phi) - shape (N, 3)
    spherical_coordinates = individual[:, :3] 
    
    # Vectorized conversion from spherical to Cartesian coordinates - shape (N, 3)
    positions = spherical_to_cartesian(spherical_coordinates)
    
    # Extract pitch and yaw angles - shape (N, 2)
    pitchs, yaws = individual[:, 3:5].T
    
    # Create Euler angles array with roll=0 - shape (N, 3)
    euler_angles = np.column_stack([np.zeros(len(individual)), pitchs, yaws])
    
    # Vectorized conversion from Euler angles to quaternions - shape (N, 4)
    orientations = euler_to_quaternion(euler_angles)

    # Create 4x4 transformation matrix
    # Convert quaternion to rotation matrix

    for i in range(len(individual)):
        cylinders.append(Cylinder(
            radius=propeller_radius,
            height=cylinder_height,
            position=positions[i],
            orientation=orientations[i]
        ))
    return cylinders

    return cylinders

def cylinders_to_spherical_angular_arms(cylinders):
    """
    Convert cylinder representations back to arms using vectorized operations.
    
    Parameters:
    - cylinders: list of cylinder dictionaries with keys: 'position', 'radius', 'height', 'orientation'
    
    Returns:
    - numpy array of shape (N, 5) with columns [r, theta, phi, pitch, yaw]
    """
    # Extract positions and orientations into arrays
    positions = np.array([cyl.position for cyl in cylinders])
    orientations = np.array([cyl.orientation for cyl in cylinders])
    
    # Vectorized conversion from Cartesian to spherical coordinates - shape (N, 3)
    spherical_coordinates = cartesian_to_spherical(positions)
    
    # Vectorized conversion from quaternions to Euler angles - shape (N, 3)
    euler_angles = quaternion_to_euler(orientations)
    
    # Extract pitch and yaw angles (ignore roll) - shape (N, 2)
    motor_yaw_pitch = euler_angles[:, 1:3]  # [pitch, yaw]
    
    # Combine spherical coordinates with motor angles - shape (N, 5)
    arms_without_dir = np.hstack((spherical_coordinates, motor_yaw_pitch))
    
    return arms_without_dir

def spherical_angular_arms_to_cartesian_positions_and_quaternions(arms):
    """
    Convert spherical angular arms to Cartesian positions and quaternions.
    
    Parameters:
    - arms: numpy array of shape (N, 5) with columns [r, theta, phi, pitch, yaw]
    
    Returns:
    - tuple: (positions, quaternions) where:
        - positions: numpy array of shape (N, 3) with Cartesian coordinates [x, y, z]
        - quaternions: numpy array of shape (N, 4) with quaternion components [w, x, y, z]
    """
    # Extract spherical coordinates (r, theta, phi) - shape (N, 3)
    spherical_coordinates = arms[:, :3]
    
    # Vectorized conversion from spherical to Cartesian coordinates - shape (N, 3)
    positions = spherical_to_cartesian(spherical_coordinates)
    
    # Extract pitch and yaw angles - shape (N, 2)
    pitchs, yaws = arms[:, 3:5].T
    
    # Create Euler angles array with roll=0 - shape (N, 3)
    euler_angles = np.column_stack([np.zeros(len(arms)), pitchs, yaws])
    
    # Vectorized conversion from Euler angles to quaternions - shape (N, 4)
    quaternions = euler_to_quaternion(euler_angles)
    
    return positions, quaternions

def cartesian_positions_and_quaternions_to_spherical_angular_arms(positions, quaternions):
    """
    Convert Cartesian positions and quaternions to spherical angular arms.
    
    Parameters:
    - positions: numpy array of shape (N, 3) with Cartesian coordinates [x, y, z]
    - quaternions: numpy array of shape (N, 4) with quaternion components [w, x, y, z]
    
    Returns:
    - numpy array of shape (N, 5) with columns [r, theta, phi, pitch, yaw]
    """
    # Vectorized conversion from Cartesian to spherical coordinates - shape (N, 3)
    spherical_coordinates = cartesian_to_spherical(positions)
    
    # Vectorized conversion from quaternions to Euler angles - shape (N, 3)
    euler_angles = quaternion_to_euler(quaternions)
    
    # Extract pitch and yaw angles (ignore roll) - shape (N, 2)
    motor_angles = euler_angles[:, 1:3]  # [pitch, yaw]
    
    # Wrap motor angles to valid ranges for spherical coordinates
    # pitch: Handle both positive and negative values to map to [0, π]
    # For values outside [-π/2, π/2], use absolute value and wrap appropriately
    pitch = motor_angles[:, 0]
    
    # Normalize pitch to [-π, π] range
    # Use proper angle wrapping to maintain continuity
    normalized_pitch = ((pitch + np.pi) % (2*np.pi)) - np.pi
    motor_angles[:, 0] = normalized_pitch
    
    # yaw: normalize to [-π, π] range
    normalized_yaw = ((motor_angles[:, 1] + np.pi) % (2*np.pi)) - np.pi
    motor_angles[:, 1] = normalized_yaw
    
    # Combine spherical coordinates with motor angles - shape (N, 5)
    arms = np.hstack((spherical_coordinates, motor_angles))
    
    return arms

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_arms_format(arms, expected_shape_cols=5):
    """
    Validate that the arms array has the expected format.
    
    Parameters:
    - arms: numpy array to validate
    - expected_shape_cols: expected number of columns (default 5 for [r, theta, phi, pitch, yaw])
    
    Returns:
    - bool: True if valid format
    
    Raises:
    - ValueError: If format is invalid
    """
    arms = np.asarray(arms)
    
    if arms.ndim != 2:
        raise ValueError(f"Arms array must be 2D, got {arms.ndim}D")
    
    if arms.shape[1] != expected_shape_cols:
        raise ValueError(f"Arms array must have {expected_shape_cols} columns, got {arms.shape[1]}")
    
    # Check for valid spherical coordinates
    r_values = arms[:, 0]
    if np.any(r_values < 0):
        raise ValueError("Radial distances (r) must be non-negative")
    
    theta_values = arms[:, 1]
    if np.any(theta_values < -np.pi) or np.any(theta_values > np.pi):
        raise ValueError("Azimuthal angles (theta) must be in range [-π, π]")
    
    phi_values = arms[:, 2]
    if np.any(phi_values < -np.pi) or np.any(phi_values > np.pi):
        raise ValueError("Polar angles (phi) must be in range [-π, π]")
    
    return True

def create_arm_from_components(r, theta, phi, pitch, yaw):
    """
    Create a single arm array from individual components.
    
    Parameters:
    - r: radial distance
    - theta: azimuthal angle
    - phi: polar angle  
    - pitch: pitch angle
    - yaw: yaw angle
    
    Returns:
    - numpy array of shape (1, 5)
    """
    return np.array([[r, theta, phi, pitch, yaw]])

def extract_arm_components(arms):
    """
    Extract individual components from arms array.
    
    Parameters:
    - arms: numpy array of shape (N, 5)
    
    Returns:
    - tuple: (r, theta, phi, pitch, yaw) where each is a 1D array of length N
    """
    arms = np.asarray(arms)
    validate_arms_format(arms)
    
    return arms[:, 0], arms[:, 1], arms[:, 2], arms[:, 3], arms[:, 4]

# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

if __name__ == "__main__":
    print("=== Enhanced Arm Conversions Testing ===")
    
    # Create test data: 3 arms with different configurations
    test_arms = np.array([
        [1.0, 0.0, np.pi/2, 0.1, 0.0],      # Arm 1: r=1, theta=0, phi=π/2, pitch=0.1, yaw=0
        [1.5, np.pi/2, np.pi/3, 0.0, 0.2],  # Arm 2: r=1.5, theta=π/2, phi=π/3, pitch=0, yaw=0.2
        [2.0, np.pi, np.pi/4, -0.1, 0.5]    # Arm 3: r=2, theta=π, phi=π/4, pitch=-0.1, yaw=0.5
    ])
    
    print("Original arms (r, theta, phi, pitch, yaw):")
    print(test_arms)
    
    # Test conversion to cylinders
    print("\n--- Converting to Cylinders ---")
    cylinders = spherical_angular_arms_to_cylinders(test_arms, propeller_radius=0.0254, cylinder_height=0.3048)
    
    print(f"Created {len(cylinders)} cylinders")
    for i, cyl in enumerate(cylinders):
        print(f"Cylinder {i+1}:")
        print(f"  Position: {cyl['position']}")
        print(f"  Orientation: {cyl['orientation']}")
        print(f"  Radius: {cyl['radius']}, Height: {cyl['height']}")
    
    # Test round-trip conversion
    print("\n--- Round-trip Conversion Test ---")
    arms_reconstructed = cylinders_to_spherical_angular_arms(cylinders)
    print("Reconstructed arms:")
    print(arms_reconstructed)
    
    print("Difference from original:")
    print(np.abs(test_arms - arms_reconstructed))
    print(f"Max difference: {np.max(np.abs(test_arms - arms_reconstructed)):.6f}")
    
    # Test Cartesian position and quaternion conversion
    print("\n--- Cartesian Position and Quaternion Conversion ---")
    positions, quaternions = spherical_angular_arms_to_cartesian_positions_and_quaternions(test_arms)
    print("Positions:")
    print(positions)
    print("Quaternions:")
    print(quaternions)
    
    # Test reverse conversion
    print("\n--- Reverse Cartesian Conversion ---")
    arms_from_cart = cartesian_positions_and_quaternions_to_spherical_angular_arms(positions, quaternions)
    print("Arms from Cartesian:")
    print(arms_from_cart)
    
    print("Difference from original:")
    print(np.abs(test_arms - arms_from_cart))
    print(f"Max difference: {np.max(np.abs(test_arms - arms_from_cart)):.6f}")
    
    # Test validation
    print("\n--- Validation Test ---")
    try:
        validate_arms_format(test_arms)
        print("✓ Arms format validation passed")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Test utility functions
    print("\n--- Utility Functions Test ---")
    single_arm = create_arm_from_components(1.0, 0.5, 1.0, 0.1, 0.2)
    print(f"Single arm created: {single_arm}")
    
    r, theta, phi, pitch, yaw = extract_arm_components(test_arms)
    print(f"Extracted components:")
    print(f"  r: {r}")
    print(f"  theta: {theta}")
    print(f"  phi: {phi}")
    print(f"  pitch: {pitch}")
    print(f"  yaw: {yaw}")
    
    print("\n=== All tests completed ===")

# ============================================================================
# CARTESIAN EULER GENOME CONVERSIONS
# ============================================================================

def cartesian_euler_arms_to_cylinders(individual, propeller_radius, cylinder_height):
    """
    Convert Cartesian Euler arms to cylinder representations using vectorized operations.
    
    Parameters:
    - individual: numpy array of shape (N, 7) with columns [x, y, z, roll, pitch, yaw, direction]
    - propeller_radius: float, radius of the propeller/cylinder
    - cylinder_height: float, height of the cylinder
    
    Returns:
    - List of Cylinder instances
    """
    cylinders = []

    # Extract Cartesian positions (x, y, z) - shape (N, 3)
    positions = individual[:, :3]
    
    # Extract Euler angles (roll, pitch, yaw) - shape (N, 3)
    euler_angles = individual[:, 3:6]
    
    # Vectorized conversion from Euler angles to quaternions - shape (N, 4)
    orientations = euler_to_quaternion(euler_angles)

    # Create Cylinder instances
    for i in range(len(individual)):
        # Create 4x4 transformation matrix
        # First create rotation matrix from quaternion
        qw, qx, qy, qz = orientations[i]
        
        # Convert quaternion to rotation matrix
        rot_matrix = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
            [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
            [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
        ])
        
        # Create 4x4 transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = positions[i]
        
        # Create Cylinder instance
        cylinder = Cylinder(
            position=positions[i],
            radius=propeller_radius,
            height=cylinder_height,
            orientation=orientations[i],  # quaternion [w, x, y, z]
        )
        cylinders.append(cylinder)

    return cylinders

def cylinders_to_cartesian_euler_arms(cylinders):
    """
    Convert Cylinder instances back to Cartesian Euler arms using vectorized operations.
    
    Parameters:
    - cylinders: list of Cylinder instances
    
    Returns:
    - numpy array of shape (N, 6) with columns [x, y, z, roll, pitch, yaw] (without direction)
    """
    # Extract positions and orientations into arrays
    positions = np.array([cylinder.position for cylinder in cylinders])
    orientations = np.array([cylinder.orientation for cylinder in cylinders])
    
    # Vectorized conversion from quaternions to Euler angles - shape (N, 3)
    euler_angles = quaternion_to_euler(orientations)
    
    # Combine Cartesian positions with Euler angles - shape (N, 6)
    arms_without_dir = np.hstack((positions, euler_angles))
    
    return arms_without_dir

def cartesian_euler_arms_to_cartesian_positions_and_quaternions(arms):
    """
    Convert Cartesian Euler arms to Cartesian positions and quaternions.
    
    Parameters:
    - arms: numpy array of shape (N, 7) with columns [x, y, z, roll, pitch, yaw, direction]
    
    Returns:
    - tuple: (positions, quaternions) where:
        - positions: numpy array of shape (N, 3) with Cartesian coordinates [x, y, z]
        - quaternions: numpy array of shape (N, 4) with quaternion components [w, x, y, z]
    """
    # Extract Cartesian positions (x, y, z) - shape (N, 3)
    positions = arms[:, :3]
    
    # Extract Euler angles (roll, pitch, yaw) - shape (N, 3)
    euler_angles = arms[:, 3:6]
    
    # Vectorized conversion from Euler angles to quaternions - shape (N, 4)
    quaternions = euler_to_quaternion(euler_angles)
    
    return positions, quaternions

def cartesian_positions_and_quaternions_to_cartesian_euler_arms(positions, quaternions):
    """
    Convert Cartesian positions and quaternions to Cartesian Euler arms.
    
    Parameters:
    - positions: numpy array of shape (N, 3) with Cartesian coordinates [x, y, z]
    - quaternions: numpy array of shape (N, 4) with quaternion components [w, x, y, z]
    
    Returns:
    - numpy array of shape (N, 6) with columns [x, y, z, roll, pitch, yaw] (without direction)
    """
    # Vectorized conversion from quaternions to Euler angles - shape (N, 3)
    euler_angles = quaternion_to_euler(quaternions)
    
    # Combine Cartesian positions with Euler angles - shape (N, 6)
    arms = np.hstack((positions, euler_angles))
    
    return arms

# ============================================================================
# ALIAS FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================

# Create aliases that match the existing pattern used in particle_repair_operator.py
def arms_to_cylinders_cartesian_euler(individual, propeller_radius=0.0254, cylinder_height=None):
    """Alias for cartesian_euler_arms_to_cylinders with default parameters."""
    if cylinder_height is None:
        cylinder_height = 8 * propeller_radius
    valid_arms = ~np.isnan(individual).any(axis=-1)
    if not np.any(valid_arms):
        return []
    valid_arm_params = individual[valid_arms]
    return cartesian_euler_arms_to_cylinders(valid_arm_params, propeller_radius, cylinder_height)

def cylinders_to_arms_cartesian_euler(cylinders):
    """Alias for cylinders_to_cartesian_euler_arms."""
    return cylinders_to_cartesian_euler_arms(cylinders)

# Create aliases that match the existing pattern used in particle_repair_operator.py for polar angular
def arms_to_cylinders_polar_angular(individual, propeller_radius=0.0254, cylinder_height=None):
    """Alias for spherical_angular_arms_to_cylinders with default parameters."""
    if cylinder_height is None:
        cylinder_height = 8 * propeller_radius
    valid_arms = ~np.isnan(individual).any(axis=-1)
    if not np.any(valid_arms):
        return []
    valid_arm_params = individual[valid_arms]
    return spherical_angular_arms_to_cylinders(valid_arm_params, propeller_radius, cylinder_height)

def cylinders_to_arms_polar_angular(cylinders):
    """Alias for cylinders_to_spherical_angular_arms."""
    return cylinders_to_spherical_angular_arms(cylinders)