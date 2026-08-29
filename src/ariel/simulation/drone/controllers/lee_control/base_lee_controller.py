import numpy as np
import logging
from .lee_math_utils import *

logger = logging.getLogger("base_lee_controller")


class BaseLeeController:
    """
    This class will operate as the base class for all controllers.
    It will be inherited by the specific controller classes.
    """

    def __init__(self, control_config, mass=1.0, inertia=None, gravity=None):
        self.cfg = control_config
        self.mass = mass
        
        # Default inertia matrix (3x3 identity scaled by mass if not provided)
        if inertia is None:
            self.robot_inertia = np.eye(3) * mass * 0.1  # Default inertia
        else:
            self.robot_inertia = np.array(inertia)
        
        # Default gravity vector (will be overridden by Lee controller with correct coordinate frame)
        if gravity is None:
            self.gravity = np.array([0.0, 0.0, -9.81])  # Default ENU, but Lee controller will override
        else:
            self.gravity = np.array(gravity)

    def update_state(self, robot_state):
        """Update robot state from external source
        robot_state should be a dict containing:
        - position: [x, y, z]
        - orientation: [qx, qy, qz, qw] quaternion
        - linear_velocity: [vx, vy, vz]
        - angular_velocity: [wx, wy, wz]
        - euler_angles: [roll, pitch, yaw]
        """
        self.robot_position = np.array(robot_state.get("position", [0.0, 0.0, 0.0]))
        self.robot_orientation = np.array(robot_state.get("orientation", [0.0, 0.0, 0.0, 1.0]))
        self.robot_linvel = np.array(robot_state.get("linear_velocity", [0.0, 0.0, 0.0]))
        self.robot_angvel = np.array(robot_state.get("angular_velocity", [0.0, 0.0, 0.0]))
        self.robot_euler_angles = np.array(robot_state.get("euler_angles", [0.0, 0.0, 0.0]))
        
        # Angular velocity from simulator is already in body frame
        self.robot_body_angvel = self.robot_angvel.copy()
        # Linear velocity from simulator is in world frame, body frame would need rotation
        self.robot_body_linvel = self.robot_linvel.copy()  # Keep world frame for now
        
        # Vehicle frame orientation (only yaw component)
        from .lee_math_utils import vehicle_frame_quat_from_quat
        self.robot_vehicle_orientation = vehicle_frame_quat_from_quat(self.robot_orientation)
        self.robot_vehicle_linvel = self.robot_linvel.copy()  # Simplified for now

    def init_controller_gains(self):
        """Initialize controller gains from configuration"""
        # Get gains from config (defaults only used if config doesn't provide them)
        self.K_pos_max = np.array(getattr(self.cfg, 'K_pos_tensor_max', [2.0, 2.0, 3.0]))
        self.K_pos_min = np.array(getattr(self.cfg, 'K_pos_tensor_min', [2.0, 2.0, 3.0]))
        self.K_linvel_max = np.array(getattr(self.cfg, 'K_vel_tensor_max', [3.0, 3.0, 4.0]))
        self.K_linvel_min = np.array(getattr(self.cfg, 'K_vel_tensor_min', [3.0, 3.0, 4.0]))
        self.K_rot_max = np.array(getattr(self.cfg, 'K_rot_tensor_max', [0.3, 0.3, 0.1]))  # Scaled for direct SO(3) control
        self.K_rot_min = np.array(getattr(self.cfg, 'K_rot_tensor_min', [0.3, 0.3, 0.1]))  # Same as max
        self.K_angvel_max = np.array(getattr(self.cfg, 'K_angvel_tensor_max', [0.05, 0.05, 0.03]))  # Rate damping gain
        self.K_angvel_min = np.array(getattr(self.cfg, 'K_angvel_tensor_min', [0.05, 0.05, 0.03]))  # Same as max

        # Use the values directly (since min=max, averaging gives the same result)
        # This ensures we use the configured gains without dilution
        self.K_pos_current = (self.K_pos_max + self.K_pos_min) / 2.0
        self.K_linvel_current = (self.K_linvel_max + self.K_linvel_min) / 2.0
        self.K_rot_current = (self.K_rot_max + self.K_rot_min) / 2.0  # Will equal K_rot_max since min=max
        self.K_angvel_current = (self.K_angvel_max + self.K_angvel_min) / 2.0  # Will equal K_angvel_max since min=max 

        # Working variables that are needed later in the controller
        self.accel = np.zeros(3)
        self.wrench_command = np.zeros(6)  # [fx, fy, fz, tx, ty, tz]

        # Control variables
        self.desired_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.desired_body_angvel = np.zeros(3)
        self.euler_angle_rates = np.zeros(3)

        # Buffer for rotation matrix calculations
        self.rotation_matrix_buffer = np.zeros((3, 3))

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def reset_commands(self):
        """Reset wrench command to zero"""
        self.wrench_command.fill(0.0)

    def reset(self):
        """Reset controller state"""
        self.randomize_params()

    def randomize_params(self):
        """Randomize controller parameters within configured bounds"""
        if not getattr(self.cfg, 'randomize_params', False):
            return
        
        self.K_pos_current = rand_float_uniform(self.K_pos_min, self.K_pos_max)
        self.K_linvel_current = rand_float_uniform(self.K_linvel_min, self.K_linvel_max)
        self.K_rot_current = rand_float_uniform(self.K_rot_min, self.K_rot_max)
        self.K_angvel_current = rand_float_uniform(self.K_angvel_min, self.K_angvel_max)

    def compute_acceleration(self, setpoint_position, setpoint_velocity):
        """Compute desired acceleration from position and velocity setpoints"""
        position_error_world_frame = np.array(setpoint_position) - self.robot_position
        
        # setpoint_velocity is already in world frame, no rotation needed
        setpoint_velocity_world_frame = np.array(setpoint_velocity)
        velocity_error = setpoint_velocity_world_frame - self.robot_linvel

        accel_command = (
            self.K_pos_current * position_error_world_frame
            + self.K_linvel_current * velocity_error
        )
        
        # FIXED: Much more aggressive saturation to prevent force explosion
        max_accel = 5.0  # m/s² (conservative limit - was 20.0 causing huge forces)
        accel_magnitude = np.linalg.norm(accel_command)
        if accel_magnitude > max_accel:
            accel_command = accel_command / accel_magnitude * max_accel
            
        return accel_command

    def compute_body_torque(self, setpoint_orientation, setpoint_angvel):
        """Compute body torque from orientation and angular velocity setpoints"""
        setpoint_angvel = np.array(setpoint_angvel)
        max_yaw_rate = getattr(self.cfg, 'max_yaw_rate', 2.0)  # Default max yaw rate
        setpoint_angvel[2] = np.clip(setpoint_angvel[2], -max_yaw_rate, max_yaw_rate)

        RT_Rd_quat = quat_mul(quat_inverse(self.robot_orientation),
                              np.array(setpoint_orientation))
        RT_Rd = quat_to_rotation_matrix(RT_Rd_quat)

        # Ensure RT_Rd is 2D for vee map computation
        if RT_Rd.ndim == 3:
            RT_Rd = RT_Rd.squeeze()

        skew_symmetric = RT_Rd.T - RT_Rd
        rotation_error = 0.5 * compute_vee_map(skew_symmetric)

        # Ensure rotation_error is 1D
        if rotation_error.ndim > 1:
            rotation_error = rotation_error.flatten()[:3]

        angvel_error = self.robot_body_angvel - quat_rotate(RT_Rd_quat, setpoint_angvel)

        # Feed-forward body rates (gyroscopic effects)
        inertia_angvel = self.robot_inertia @ self.robot_body_angvel
        feed_forward_body_rates = np.cross(self.robot_body_angvel, inertia_angvel)

        # Lee geometric control on SO(3): M = -K_R·e_R - K_omega·e_omega + ω×J·ω.
        # `rotation_error` already equals Lee's e_R (sign convention from
        # `0.5·vee(R_d^T R - R^T R_d)`), and `angvel_error` equals e_omega, so all
        # three terms emit body torque in physical N·m with no downstream negation.
        torque = (
            -self.K_rot_current * rotation_error
            - self.K_angvel_current * angvel_error
            + feed_forward_body_rates
        )

        # Debug: Print torque calculation details periodically (disabled for performance)
        # if not hasattr(self, '_torque_debug_counter'):
        #     self._torque_debug_counter = 0
        # self._torque_debug_counter += 1
        # if self._torque_debug_counter % 100 == 1:  # Print every 100 calls
        #     print(f"\nTORQUE CALCULATION DEBUG:")
        #     print(f"  rotation_error: {rotation_error}")
        #     print(f"  K_rot_current: {self.K_rot_current}")
        #     print(f"  K_rot * rot_err: {self.K_rot_current * rotation_error}")
        #     print(f"  angvel_error: {angvel_error}")
        #     print(f"  K_angvel_current: {self.K_angvel_current}")
        #     print(f"  K_angvel * angvel_err: {self.K_angvel_current * angvel_error}")
        #     print(f"  feed_forward: {feed_forward_body_rates}")
        #     print(f"  TOTAL torque: {torque}")

        return torque


def calculate_desired_orientation_from_forces_and_yaw(forces_command, yaw_setpoint, orient="NED"):
    """Calculate desired orientation from force command and yaw setpoint
    
    Args:
        forces_command: Force command in WORLD frame [Fx, Fy, Fz]
        yaw_setpoint: Desired yaw angle
        orient: Coordinate system ("NED" or "ENU")
    """
    forces_command = np.array(forces_command)
    yaw_setpoint = np.array(yaw_setpoint)
    
    if forces_command.ndim == 1:
        # Single vector case - align body Z-axis with world force direction
        # The sign depends on coordinate system and motor configuration
        if orient == "NED":
            # In NED: motors point upward (negative Z), so body Z should oppose force
            b3_c = -forces_command / np.linalg.norm(forces_command)
        else:  # ENU
            # In ENU: motors point upward (positive Z), so body Z should align with force
            b3_c = forces_command / np.linalg.norm(forces_command)
            
        if orient == "NED":
            # NED: X=North, Y=East, Z=Down - follow PX4 approach
            temp_dir = np.array([np.cos(yaw_setpoint), np.sin(yaw_setpoint), 0.0])
            b2_c = np.cross(b3_c, temp_dir)  # Calculate b2 first: b3 × desired_heading
            b2_c = b2_c / np.linalg.norm(b2_c)
            b1_c = np.cross(b2_c, b3_c)  # Complete right-handed frame: b2 × b3
        else:  # ENU
            # ENU: Original cross product order
            temp_dir = np.array([np.cos(yaw_setpoint), np.sin(yaw_setpoint), 0.0])
            b2_c = np.cross(b3_c, temp_dir)
            b2_c = b2_c / np.linalg.norm(b2_c)
            b1_c = np.cross(b2_c, b3_c)

        rotation_matrix_desired = np.column_stack([b1_c, b2_c, b3_c])
        quat_desired = matrix_to_quaternion(rotation_matrix_desired)
    else:
        # Array case - align body Z-axis with world force direction  
        if orient == "NED":
            # In NED: motors point upward (negative Z), so body Z should oppose force
            b3_c = -forces_command / np.linalg.norm(forces_command, axis=-1, keepdims=True)
        else:  # ENU
            # In ENU: motors point upward (positive Z), so body Z should align with force
            b3_c = forces_command / np.linalg.norm(forces_command, axis=-1, keepdims=True)
            
        temp_dir = np.zeros_like(forces_command)
        temp_dir[..., 0] = np.cos(yaw_setpoint)
        temp_dir[..., 1] = np.sin(yaw_setpoint)

        b2_c = np.cross(b3_c, temp_dir, axis=-1)
        b2_c = b2_c / np.linalg.norm(b2_c, axis=-1, keepdims=True)
        b1_c = np.cross(b2_c, b3_c, axis=-1)

        # Build rotation matrices
        batch_size = forces_command.shape[0]
        rotation_matrices = np.zeros((batch_size, 3, 3))
        rotation_matrices[..., :, 0] = b1_c
        rotation_matrices[..., :, 1] = b2_c
        rotation_matrices[..., :, 2] = b3_c
        
        quat_desired = matrix_to_quaternion(rotation_matrices)
    
    return quat_desired


def calculate_desired_orientation_for_position_velocity_control(
    forces_command, yaw_setpoint, rotation_matrix_buffer=None, orient="NED"
):
    """Calculate desired orientation for position/velocity control
    
    Args:
        forces_command: Force command in WORLD frame [Fx, Fy, Fz]
        yaw_setpoint: Desired yaw angle
        rotation_matrix_buffer: Optional buffer for rotation matrix
        orient: Coordinate system ("NED" or "ENU")
    """
    forces_command = np.array(forces_command)
    yaw_setpoint = np.array(yaw_setpoint)
    
    if forces_command.ndim == 1:
        # Single vector case - align body Z-axis with world force direction
        # The sign depends on coordinate system and motor configuration
        if orient == "NED":
            # In NED: motors point upward (negative Z), so body Z should oppose force
            b3_c = -forces_command / np.linalg.norm(forces_command)
        else:  # ENU
            # In ENU: motors point upward (positive Z), so body Z should align with force
            b3_c = forces_command / np.linalg.norm(forces_command)
            
        if orient == "NED":
            # NED: X=North, Y=East, Z=Down - follow PX4 approach
            temp_dir = np.array([np.cos(yaw_setpoint), np.sin(yaw_setpoint), 0.0])
            b2_c = np.cross(b3_c, temp_dir)  # Calculate b2 first: b3 × desired_heading
            b2_c = b2_c / np.linalg.norm(b2_c)
            b1_c = np.cross(b2_c, b3_c)  # Complete right-handed frame: b2 × b3
        else:  # ENU
            # ENU: Original cross product order
            temp_dir = np.array([np.cos(yaw_setpoint), np.sin(yaw_setpoint), 0.0])
            b2_c = np.cross(b3_c, temp_dir)
            b2_c = b2_c / np.linalg.norm(b2_c)
            b1_c = np.cross(b2_c, b3_c)

        rotation_matrix_desired = np.column_stack([b1_c, b2_c, b3_c])
        quat_desired = matrix_to_quaternion(rotation_matrix_desired)
    else:
        # Array case - align body Z-axis with world force direction
        # The sign depends on coordinate system and motor configuration
        if orient == "NED":
            # In NED: motors point upward (negative Z), so body Z should oppose force
            b3_c = -forces_command / np.linalg.norm(forces_command, axis=-1, keepdims=True)
        else:  # ENU
            # In ENU: motors point upward (positive Z), so body Z should align with force
            b3_c = forces_command / np.linalg.norm(forces_command, axis=-1, keepdims=True)
        temp_dir = np.zeros_like(forces_command)
        temp_dir[..., 0] = np.cos(yaw_setpoint)
        temp_dir[..., 1] = np.sin(yaw_setpoint)

        if orient == "NED":
            # NED: Follow PX4 approach for array case
            b2_c = np.cross(b3_c, temp_dir, axis=-1)  # Calculate b2 first: b3 × desired_heading
            b2_c = b2_c / np.linalg.norm(b2_c, axis=-1, keepdims=True)
            b1_c = np.cross(b2_c, b3_c, axis=-1)  # Complete right-handed frame: b2 × b3
        else:  # ENU
            # ENU: Original cross product order
            b2_c = np.cross(b3_c, temp_dir, axis=-1)
            b2_c = b2_c / np.linalg.norm(b2_c, axis=-1, keepdims=True)
            b1_c = np.cross(b2_c, b3_c, axis=-1)

        # Build rotation matrices
        batch_size = forces_command.shape[0]
        rotation_matrices = np.zeros((batch_size, 3, 3))
        rotation_matrices[..., :, 0] = b1_c
        rotation_matrices[..., :, 1] = b2_c
        rotation_matrices[..., :, 2] = b3_c
        
        quat_desired = matrix_to_quaternion(rotation_matrices)

    return quat_desired


def euler_rates_to_body_rates(robot_euler_angles, desired_euler_rates, rotmat_buffer=None):
    """Convert Euler angle rates to body frame angular rates"""
    robot_euler_angles = np.array(robot_euler_angles)
    desired_euler_rates = np.array(desired_euler_rates)
    
    if robot_euler_angles.ndim == 1:
        # Single vector case
        s_pitch = np.sin(robot_euler_angles[1])
        c_pitch = np.cos(robot_euler_angles[1])
        s_roll = np.sin(robot_euler_angles[0])
        c_roll = np.cos(robot_euler_angles[0])

        rotmat_euler_to_body_rates = np.array([
            [1.0, 0.0, -s_pitch],
            [0.0, c_roll, s_roll * c_pitch],
            [0.0, -s_roll, c_roll * c_pitch]
        ])

        return rotmat_euler_to_body_rates @ desired_euler_rates
    else:
        # Array case (batch processing)
        s_pitch = np.sin(robot_euler_angles[..., 1])
        c_pitch = np.cos(robot_euler_angles[..., 1])
        s_roll = np.sin(robot_euler_angles[..., 0])
        c_roll = np.cos(robot_euler_angles[..., 0])

        batch_size = robot_euler_angles.shape[0]
        rotmat_euler_to_body_rates = np.zeros((batch_size, 3, 3))
        
        rotmat_euler_to_body_rates[:, 0, 0] = 1.0
        rotmat_euler_to_body_rates[:, 1, 1] = c_roll
        rotmat_euler_to_body_rates[:, 0, 2] = -s_pitch
        rotmat_euler_to_body_rates[:, 2, 1] = -s_roll
        rotmat_euler_to_body_rates[:, 1, 2] = s_roll * c_pitch
        rotmat_euler_to_body_rates[:, 2, 2] = c_roll * c_pitch

        return np.einsum('bij,bj->bi', rotmat_euler_to_body_rates, desired_euler_rates)