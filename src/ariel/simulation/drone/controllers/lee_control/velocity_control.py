import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController, calculate_desired_orientation_for_position_velocity_control, euler_rates_to_body_rates

logger = logging.getLogger("velocity_controller")


class LeeVelocityController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None, orient="NED"):
        super().__init__(config, mass, inertia, gravity)
        self.orient = orient
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Lee velocity controller
        :param command_actions: array of shape (4,) with [vx, vy, vz, yaw_rate] velocity setpoint and yaw rate command
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Compute desired acceleration (maintaining current position, tracking velocity)
        self.accel = self.compute_acceleration(
            setpoint_position=self.robot_position,  # Maintain current position
            setpoint_velocity=command_actions[0:3],  # Track desired velocity
        )
        
        # Convert acceleration to forces (WORLD frame)
        # self.accel is in world frame, self.gravity is in world frame
        forces = (self.accel - self.gravity) * self.mass
        
        # Set complete force command (WORLD frame forces)
        self.wrench_command[0:3] = forces

        # Calculate desired orientation from WORLD frame forces and current yaw
        self.desired_quat = calculate_desired_orientation_for_position_velocity_control(
            forces, self.robot_euler_angles[2], self.rotation_matrix_buffer, self.orient
        )

        # Set desired angular rates (zero roll/pitch rates, commanded yaw rate)
        self.euler_angle_rates[0:2] = 0.0
        self.euler_angle_rates[2] = command_actions[3]
        self.desired_body_angvel = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.rotation_matrix_buffer
        )

        # Compute body torques
        self.wrench_command[3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command.copy()