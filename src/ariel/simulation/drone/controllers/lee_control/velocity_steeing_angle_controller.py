import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController, calculate_desired_orientation_for_position_velocity_control, euler_rates_to_body_rates

logger = logging.getLogger("velocity_steering_controller")


class LeeVelocitySteeringAngleController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None, orient="NED"):
        super().__init__(config, mass, inertia, gravity)
        self.orient = orient
        self.init_controller_gains()
        self.euler_angle_rates = np.zeros(3)

    def update(self, command_actions):
        """
        Lee velocity steering angle controller
        :param command_actions: array of shape (4,) with [vx, vy, vz, steering_angle] velocity and steering command
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Compute desired acceleration (maintaining current position, tracking velocity)
        self.accel = self.compute_acceleration(
            setpoint_position=self.robot_position,  # Maintain current position
            setpoint_velocity=command_actions[0:3],  # Track desired velocity
        )
        
        # Convert acceleration to forces
        forces = (self.accel - self.gravity) * self.mass
        
        # Set complete force command (WORLD frame forces)
        self.wrench_command[0:3] = forces

        # Calculate desired orientation from forces and steering angle
        self.desired_quat = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[3], self.rotation_matrix_buffer, self.orient
        )
        
        # Zero angular velocity setpoint
        self.euler_angle_rates.fill(0.0)
        self.desired_body_angvel = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.rotation_matrix_buffer
        )
        
        # Compute body torques
        self.wrench_command[3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command.copy()