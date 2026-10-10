import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController, calculate_desired_orientation_from_forces_and_yaw, euler_rates_to_body_rates

logger = logging.getLogger("acceleration_controller")


class LeeAccelerationController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None, orient="NED"):
        super().__init__(config, mass, inertia, gravity)
        self.orient = orient
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Lee acceleration controller
        :param command_actions: array of shape (4,) with [ax, ay, az, yaw_rate] acceleration command and yaw rate
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Set desired acceleration directly (WORLD frame)
        self.accel = command_actions[0:3].copy()
        # Convert to WORLD frame forces
        forces = self.mass * (self.accel - self.gravity)
        
        # Set complete force command (WORLD frame forces)
        self.wrench_command[0:3] = forces

        # Calculate desired orientation from WORLD frame forces and current yaw
        self.desired_quat = calculate_desired_orientation_from_forces_and_yaw(
            forces, self.robot_euler_angles[2], self.orient
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