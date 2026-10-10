"""
Repair workflow module for drone genomes.

This module implements a three-stage repair process:
1. Optimization repair - Fix collisions using constrained optimization
2. Hover check - Verify the individual can hover
3. Hover repair - Align thrust vectors for optimal hovering

The workflow always returns a tuple (repaired_individual, status_message).
If repair fails at any stage, returns (NaN_individual, failure_message).
"""

from __future__ import annotations

import sys
import os
import numpy as np
import numpy.typing as npt
from typing import Tuple, Optional, List, Dict, Any
import warnings

# Add drone-hover library to path if DRONE_HOVER_PATH environment variable is set
DRONE_HOVER_PATH = os.environ.get("DRONE_HOVER_PATH", None)
if DRONE_HOVER_PATH and DRONE_HOVER_PATH not in sys.path:
    sys.path.insert(0, DRONE_HOVER_PATH)

# Airevolve imports
from ariel.ec.drone.genome_handlers.operators.optimization_repair_operator import (
    OptimizationBasedRepairOperator,
    OptimizationRepairConfig,
)
from ariel.ec.drone.genome_handlers.conversions.arm_conversions import (
    Cylinder,
    arms_to_cylinders_polar_angular,
    cylinders_to_arms_polar_angular,
    arms_to_cylinders_cartesian_euler,
    cylinders_to_arms_cartesian_euler,
)
# Note: get_sim is imported lazily in functions that need it due to import issues in hovering_info.py

# Drone-hover imports
try:
    from dronehover.bodies.custom_bodies import Custombody
    from dronehover.optimization import Hover
    from dronehover.utils import align_vectors
    DRONE_HOVER_AVAILABLE = True
except ImportError as e:
    DRONE_HOVER_AVAILABLE = False
    DRONE_HOVER_IMPORT_ERROR = str(e)
    path_msg = f" Set DRONE_HOVER_PATH environment variable if needed." if not DRONE_HOVER_PATH else f" Current path: {DRONE_HOVER_PATH}"
    warnings.warn(
        f"Failed to import drone-hover library: {e}\n"
        f"Hover repair (Stage 3) will not be available.\n"
        f"Make sure drone-hover is installed.{path_msg}"
    )


# ============================================================================
# CONVERSION UTILITIES: Genome ↔ Drone-Hover Props
# ============================================================================

def genome_to_drone_hover_props(
    genome: npt.NDArray[Any],
    coordinate_system: str = 'spherical',
    propeller_radius: float = 0.0127,  # 1-inch radius (2-inch diameter propellers)
    cylinder_height: float = None,  # Will default to 8 * propeller_radius
    default_propsize: int = 2
) -> List[Dict[str, Any]]:
    """
    Convert airevolve genome to drone-hover props format.

    The props format is:
    {
        "loc": [x, y, z],              # Motor position
        "dir": [dx, dy, dz, rotation], # Thrust direction + rotation (ccw/cw)
        "propsize": int                # Propeller size
    }

    Parameters:
    -----------
    genome : array-like of shape (n_arms, 6) or (n_arms, 7)
        Genome in spherical [r, theta, phi, pitch, yaw, direction]
        or Cartesian [x, y, z, roll, pitch, yaw, direction] format
    coordinate_system : str
        'spherical' or 'cartesian'
    propeller_radius : float
        Radius of propeller (default: 0.0127m = 1 inch radius, 2-inch diameter propellers)
    cylinder_height : float, optional
        Height of motor cylinder (default: 8 * propeller_radius = swept area)
    default_propsize : int
        Default propeller size (default: 2 for 2-inch diameter)

    Returns:
    --------
    props : list of dicts
        List of propeller configurations in drone-hover format
    """
    # Set default cylinder height to 8 * propeller_radius if not provided
    if cylinder_height is None:
        cylinder_height = 8 * propeller_radius

    # Filter out NaN arms
    valid_arms = ~np.isnan(genome).any(axis=-1)
    if not np.any(valid_arms):
        return []

    valid_genome = genome[valid_arms]

    # Convert genome to cylinders (which have position and orientation)
    if coordinate_system == 'spherical':
        cylinders = arms_to_cylinders_polar_angular(
            valid_genome, propeller_radius, cylinder_height
        )
    elif coordinate_system == 'cartesian':
        cylinders = arms_to_cylinders_cartesian_euler(
            valid_genome, propeller_radius, cylinder_height
        )
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")

    # Extract direction column (last column)
    directions = valid_genome[:, -1]

    # Convert cylinders to props
    props = []
    for i, cyl in enumerate(cylinders):
        # Get motor position (cylinder center)
        loc = cyl.position.tolist()

        # Get thrust direction from cylinder orientation
        # The cylinder is oriented along its z-axis, so we extract the z-direction
        # from the rotation matrix
        rotation_matrix = cyl.rotation_matrix
        thrust_dir = rotation_matrix[:, 2]  # Z-axis direction

        # Determine rotation direction (ccw or cw)
        # direction = 0 -> ccw, direction = 1 -> cw
        rotation = "ccw" if directions[i] < 0.5 else "cw"

        # Create prop dict
        prop = {
            "loc": loc,
            "dir": thrust_dir.tolist() + [rotation],
            "propsize": default_propsize
        }
        props.append(prop)

    return props


def drone_hover_props_to_genome(
    props: List[Dict[str, Any]],
    coordinate_system: str = 'spherical',
    original_genome_shape: Optional[Tuple[int, int]] = None
) -> npt.NDArray[Any]:
    """
    Convert drone-hover props back to airevolve genome format.

    Parameters:
    -----------
    props : list of dicts
        Propeller configurations in drone-hover format
    coordinate_system : str
        'spherical' or 'cartesian'
    original_genome_shape : tuple, optional
        Shape of original genome (n_arms, n_params). If provided and larger
        than len(props), will pad with NaN.

    Returns:
    --------
    genome : array-like
        Genome in airevolve format
    """
    if len(props) == 0:
        if original_genome_shape is not None:
            return np.full(original_genome_shape, np.nan)
        else:
            return np.array([])

    # Extract positions and directions (in NED frame from drone-hover)
    positions_ned = []
    thrust_dirs_ned = []
    rotations = []

    for prop in props:
        positions_ned.append(prop["loc"])
        thrust_dirs_ned.append(prop["dir"][:3])  # x, y, z components
        rotations.append(0.0 if prop["dir"][3] == "ccw" else 1.0)

    positions_ned = np.array(positions_ned)
    thrust_dirs_ned = np.array(thrust_dirs_ned)
    rotations = np.array(rotations)

    # Convert to genome format based on coordinate system
    if coordinate_system == 'spherical':
        # Props are in NED frame (from get_sim's ENU_to_NED conversion).
        # We must invert the conversions in get_sim/hovering_info.py:
        #   Position: ENU = convert_to_cartesian(mag, arm_yaw, arm_pitch)
        #             NED = ENU_to_NED(ENU) = (ENU_y, ENU_x, -ENU_z)
        #   Direction: NED = orientation_to_unit_vector(0, mot_pitch, mot_yaw)
        #            = ENU_to_NED @ euler_R(0, pitch, yaw) @ [0, 0, -1]

        # --- Arm positions: NED -> (mag, arm_yaw, arm_pitch) ---
        # NED_to_ENU: (ned_x, ned_y, ned_z) -> (ned_y, ned_x, -ned_z)
        enu_x = positions_ned[:, 1]
        enu_y = positions_ned[:, 0]
        enu_z = -positions_ned[:, 2]

        mag = np.sqrt(enu_x**2 + enu_y**2 + enu_z**2)
        arm_yaw = np.arctan2(enu_y, enu_x)
        # convert_to_cartesian uses: z = mag * sin(pitch), so pitch = arcsin(z/mag)
        arm_pitch = np.where(mag > 0, np.arcsin(np.clip(enu_z / mag, -1, 1)), 0.0)

        # --- Motor orientation: NED direction -> (mot_pitch, mot_yaw) ---
        # orientation_to_unit_vector(0, p, y) produces NED direction:
        #   [-sin(p)*sin(y), -sin(p)*cos(y), cos(p)]
        # Inversion: mot_pitch = arccos(ned_z), mot_yaw = arctan2(-ned_x, -ned_y)
        dirs_norm = np.linalg.norm(thrust_dirs_ned, axis=1, keepdims=True)
        dirs_norm = np.where(dirs_norm > 0, dirs_norm, 1.0)
        dirs_normalized = thrust_dirs_ned / dirs_norm
        mot_pitch = np.arccos(np.clip(dirs_normalized[:, 2], -1, 1))
        mot_yaw = np.arctan2(-dirs_normalized[:, 0], -dirs_normalized[:, 1])

        # Genome columns: [mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, direction]
        genome = np.column_stack([mag, arm_yaw, arm_pitch, mot_pitch, mot_yaw, rotations])

    elif coordinate_system == 'cartesian':
        from scipy.spatial.transform import Rotation as ScipyRotation

        quaternions = []
        for thrust_dir in thrust_dirs_ned:
            thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)
            default_z = np.array([0, 0, 1])
            if np.allclose(thrust_dir, default_z):
                quat = np.array([1, 0, 0, 0])
            elif np.allclose(thrust_dir, -default_z):
                quat = np.array([0, 1, 0, 0])
            else:
                rotation_axis = np.cross(default_z, thrust_dir)
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.clip(np.dot(default_z, thrust_dir), -1, 1))
                rot = ScipyRotation.from_rotvec(rotation_angle * rotation_axis)
                quat = rot.as_quat()
                quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            quaternions.append(quat)
        quaternions = np.array(quaternions)

        from ariel.ec.drone.genome_handlers.conversions.arm_conversions import (
            cartesian_positions_and_quaternions_to_cartesian_euler_arms
        )
        genome_without_dir = cartesian_positions_and_quaternions_to_cartesian_euler_arms(
            positions_ned, quaternions
        )
        genome = np.column_stack([genome_without_dir, rotations])
    else:
        raise ValueError(f"Unknown coordinate system: {coordinate_system}")

    # Pad with NaN if necessary
    if original_genome_shape is not None:
        n_original_arms = original_genome_shape[0]
        n_current_arms = len(genome)

        if n_current_arms < n_original_arms:
            # Pad with NaN rows
            n_params = genome.shape[1]
            padding = np.full((n_original_arms - n_current_arms, n_params), np.nan)
            genome = np.vstack([genome, padding])

    return genome


# ============================================================================
# REPAIR STAGE FUNCTIONS
# ============================================================================

def stage1_optimization_repair(
    individual: npt.NDArray[Any],
    coordinate_system: str = 'spherical',
    config: Optional[OptimizationRepairConfig] = None,
    verbose: bool = False
) -> Tuple[Optional[npt.NDArray[Any]], str]:
    """
    Stage 1: Apply optimization-based repair to fix collisions.

    Parameters:
    -----------
    individual : array-like
        Genome to repair
    coordinate_system : str
        'spherical' or 'cartesian'
    config : OptimizationRepairConfig, optional
        Configuration for optimization repair
    verbose : bool
        Print progress messages

    Returns:
    --------
    (repaired_individual, message) : tuple
        - repaired_individual: Repaired genome or None if failed
        - message: Status message
    """
    try:
        if verbose:
            print("  Stage 1: Optimization Repair...")

        # Create repair operator
        repair_operator = OptimizationBasedRepairOperator(
            optimization_config=config,
            coordinate_system=coordinate_system,
            verbose=verbose
        )

        # Apply repair
        repaired = repair_operator.repair(individual)

        # Validate repair
        if repair_operator.validate(repaired):
            if verbose:
                print("  ✓ Stage 1: Passed")
            return repaired, "Stage 1: Success"
        else:
            if verbose:
                print("  ✗ Stage 1: Failed (validation failed)")
            return None, "Failed at Stage 1: Optimization Repair (validation failed)"

    except Exception as e:
        if verbose:
            print(f"  ✗ Stage 1: Failed with exception: {e}")
        return None, f"Failed at Stage 1: Optimization Repair (exception: {e})"


def stage2_hover_check(
    individual: npt.NDArray[Any],
    verbose: bool = False,
    allow_spinning: bool = False
) -> Tuple[bool, str]:
    """
    Stage 2: Check if the individual can hover.

    Uses the same logic as gate_train.py:311-318

    Parameters:
    -----------
    individual : array-like
        Genome to check
    verbose : bool
        Print progress messages
    allow_spinning : bool
        Whether to allow spinning hover (default: False)

    Returns:
    --------
    (can_hover, message) : tuple
        - can_hover: True if can hover, False otherwise
        - message: Status message
    """
    try:
        if verbose:
            print("  Stage 2: Hover Check...")
            # Debug: Print individual info
            print(f"    Individual shape: {individual.shape}")
            print(f"    Valid arms: {(~np.isnan(individual).any(axis=-1)).sum()}")

        # Lazy import to avoid module-level import issues
        from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim

        # Get simulator
        sim = get_sim(individual)

        # Compute hover
        sim.compute_hover(verbose=False)

        # Check success
        if sim.static_success == False:
            spinning_success = sim.spinning_success if allow_spinning else False
        else:
            spinning_success = False

        success = sim.static_success or spinning_success

        # Determine hover type
        if sim.static_success:
            hover_type = "static"
        elif spinning_success:
            hover_type = "spinning"
        else:
            hover_type = "none"

        if success:
            if verbose:
                print(f"  ✓ Stage 2: Passed ({hover_type} hover)")
            return True, f"Stage 2: Success ({hover_type} hover)"
        else:
            if verbose:
                print("  ✗ Stage 2: Failed (cannot hover)")
                # Debug: Print why it failed
                print(f"    static_success: {sim.static_success}")
                print(f"    spinning_success: {sim.spinning_success}")
                print(f"    allow_spinning: {allow_spinning}")
                # Check if hover solution exists
                if hasattr(sim, 'eta') and sim.eta is not None:
                    print(f"    eta (motor thrusts): {sim.eta}")
                    print(f"    eta min/max: {sim.eta.min():.4f} / {sim.eta.max():.4f}")
                if hasattr(sim, 'residual'):
                    print(f"    residual: {sim.residual}")
            return False, "Failed at Stage 2: Hover Check (cannot hover)"

    except Exception as e:
        if verbose:
            print(f"  ✗ Stage 2: Failed with exception: {e}")
            import traceback
            traceback.print_exc()
        return False, f"Failed at Stage 2: Hover Check (exception: {e})"


def stage3_hover_repair(
    individual: npt.NDArray[Any],
    coordinate_system: str = 'spherical',
    verbose: bool = False
) -> Tuple[Optional[npt.NDArray[Any]], str]:
    """
    Stage 3: Apply hover repair to align thrust vectors.

    Uses hover_repair() from drone-hover library.

    Parameters:
    -----------
    individual : array-like
        Genome to repair
    coordinate_system : str
        'spherical' or 'cartesian'
    verbose : bool
        Print progress messages

    Returns:
    --------
    (repaired_individual, message) : tuple
        - repaired_individual: Repaired genome or None if failed
        - message: Status message
    """
    if not DRONE_HOVER_AVAILABLE:
        return None, f"Failed at Stage 3: Hover Repair (drone-hover not available: {DRONE_HOVER_IMPORT_ERROR})"

    try:
        if verbose:
            print("  Stage 3: Hover Repair...")

        # Use the same get_sim() function as Stage 2 to ensure consistency
        from ariel.ec.drone.inspection.morphological_descriptors.hovering_info import get_sim

        sim = get_sim(individual)

        if sim is None:
            if verbose:
                print("  ✗ Stage 3: Failed (could not create simulator)")
            return None, "Failed at Stage 3: Hover Repair (could not create simulator)"

        # Apply hover repair
        # Reimplementation of hover_repair() from drone-hover/examples/hover_repair.py
        sim.compute_hover(verbose=False)

        if not sim.static_success:
            if verbose:
                print("  ✗ Stage 3: Failed (hover repair requires hoverable drone)")
            return None, "Failed at Stage 3: Hover Repair (drone cannot hover)"

        # Get the drone from sim
        drone = sim.drone

        # Get thrust direction
        f = sim.Bf @ sim.eta
        f = f / np.linalg.norm(f)

        # Align vectors to point downward ([0, 0, -1])
        R2 = align_vectors(f, [0, 0, -1])

        # Apply rotation to all props
        repaired_props = []
        for prop in drone.props:
            new_prop = prop.copy()

            new_pos = R2 @ np.array(prop["loc"])
            new_dir = R2 @ np.array(prop["dir"][0:3])

            new_prop["loc"] = new_pos.tolist()
            new_prop["dir"] = new_dir.tolist() + [prop["dir"][3]]

            repaired_props.append(new_prop)

        # Convert back to genome
        repaired_individual = drone_hover_props_to_genome(
            repaired_props,
            coordinate_system,
            original_genome_shape=individual.shape
        )

        if verbose:
            print("  ✓ Stage 3: Passed")

        return repaired_individual, "Stage 3: Success"

    except Exception as e:
        if verbose:
            print(f"  ✗ Stage 3: Failed with exception: {e}")
        return None, f"Failed at Stage 3: Hover Repair (exception: {e})"


# ============================================================================
# MAIN REPAIR WORKFLOW
# ============================================================================

def repair_operation_process(
    individual: npt.NDArray[Any],
    coordinate_system: str = 'spherical',
    optimization_config: Optional[OptimizationRepairConfig] = None,
    allow_spinning_hover: bool = False,
    verbose: bool = False
) -> Tuple[npt.NDArray[Any], str]:
    """
    Execute the complete three-stage repair workflow.

    Workflow:
    1. Optimization Repair - Fix collisions
    2. Hover Check - Verify can hover
    3. Hover Repair - Align thrust vectors

    If any stage fails, returns NaN individual with failure message.

    Parameters:
    -----------
    individual : array-like
        Genome to repair
    coordinate_system : str
        'spherical' or 'cartesian'
    optimization_config : OptimizationRepairConfig, optional
        Configuration for optimization repair
    allow_spinning_hover : bool
        Whether to allow spinning hover in Stage 2
    verbose : bool
        Print detailed progress messages

    Returns:
    --------
    (repaired_individual, status_message) : tuple
        - repaired_individual: Repaired genome or NaN genome if failed
        - status_message: Detailed status message indicating success or failure stage
    """
    if verbose:
        print("\n=== Repair Operation Process ===")

    # Default config: always fix motor orientation params (pitch=3, yaw=4)
    if optimization_config is None:
        optimization_config = OptimizationRepairConfig(fixed_params=[3, 4])

    # Stage 1: Optimization Repair (fix collisions, preserve motor orientation)
    repaired, msg = stage1_optimization_repair(
        individual, coordinate_system, optimization_config, verbose
    )

    if repaired is None:
        # Stage 1 failed - return NaN individual
        nan_individual = np.full_like(individual, np.nan)
        if verbose:
            print(f"✗ Repair failed: {msg}\n")
        return nan_individual, msg

    # If Stage 1 didn't change the genome (no collisions), the individual is
    # already valid — skip Stages 2+3 to avoid breaking already-repaired genomes
    # whose geometry may not survive the Stage 3 round-trip.
    if np.array_equal(repaired, individual):
        if verbose:
            print("  No collisions found — skipping Stages 2+3\n")
        return individual.copy(), "No repair needed: collision-free"

    if verbose:
        print(f"  Post-Stage 1 genome:\n{repaired}")

    # Stage 2: Hover Check (only after Stage 1 modified the genome)
    can_hover, msg = stage2_hover_check(repaired, verbose, allow_spinning_hover)

    if not can_hover:
        # Stage 2 failed - return NaN individual
        nan_individual = np.full_like(individual, np.nan)
        if verbose:
            print(f"✗ Repair failed: {msg}\n")
        return nan_individual, msg

    # Stage 3: Hover Repair
    final_repaired, msg = stage3_hover_repair(repaired, coordinate_system, verbose)

    if final_repaired is None:
        # Stage 3 failed - return NaN individual
        nan_individual = np.full_like(individual, np.nan)
        if verbose:
            print(f"✗ Repair failed: {msg}\n")
        return nan_individual, msg

    # All stages passed!
    if verbose:
        print("✓ Repair completed successfully\n")

    return final_repaired, "All stages passed: Success"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_nan_individual(shape: Tuple[int, ...]) -> npt.NDArray[Any]:
    """
    Create a NaN individual with the given shape.

    Parameters:
    -----------
    shape : tuple
        Shape of the individual array

    Returns:
    --------
    nan_individual : array-like
        Array filled with NaN values
    """
    return np.full(shape, np.nan)


def is_nan_individual(individual: npt.NDArray[Any]) -> bool:
    """
    Check if an individual is all NaN.

    Parameters:
    -----------
    individual : array-like
        Individual to check

    Returns:
    --------
    is_nan : bool
        True if all values are NaN
    """
    return np.all(np.isnan(individual))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=== Repair Workflow Module Test ===\n")

    # Create a test genome (spherical coordinates)
    # Format: [r, theta, phi, pitch, yaw, direction]
    test_genome = np.array([
        [0.15, 0.0, np.pi/4, 0.1, 0.0, 0.0],
        [0.15, np.pi/2, np.pi/4, 0.1, 0.0, 1.0],
        [0.15, np.pi, np.pi/4, 0.1, 0.0, 0.0],
        [0.15, -np.pi/2, np.pi/4, 0.1, 0.0, 1.0],
    ], dtype=float)

    print("Test genome shape:", test_genome.shape)
    print("Test genome:\n", test_genome)

    # Test conversion to props
    print("\n--- Testing genome_to_drone_hover_props ---")
    props = genome_to_drone_hover_props(test_genome, coordinate_system='spherical')
    print(f"Converted {len(props)} props:")
    for i, prop in enumerate(props):
        print(f"  Prop {i+1}:")
        print(f"    loc: {prop['loc']}")
        print(f"    dir: {prop['dir']}")
        print(f"    propsize: {prop['propsize']}")

    # Test conversion back to genome
    print("\n--- Testing drone_hover_props_to_genome ---")
    reconstructed = drone_hover_props_to_genome(
        props, coordinate_system='spherical', original_genome_shape=test_genome.shape
    )
    print("Reconstructed genome shape:", reconstructed.shape)
    print("Reconstructed genome:\n", reconstructed)

    # Check roundtrip error (excluding direction column which may have precision issues)
    print("\n--- Roundtrip Conversion Test ---")
    error = np.abs(test_genome - reconstructed)
    print(f"Max error: {np.max(error):.6e}")
    print(f"Mean error: {np.mean(error):.6e}")

    # Test full repair workflow
    print("\n--- Testing Full Repair Workflow ---")
    repaired, status = repair_operation_process(
        test_genome,
        coordinate_system='spherical',
        verbose=True
    )

    print(f"\nFinal status: {status}")
    print(f"Repaired individual is NaN: {is_nan_individual(repaired)}")

    if not is_nan_individual(repaired):
        print("Repaired genome:\n", repaired)

    print("\n=== Test Complete ===")
