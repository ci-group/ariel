import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController, euler_rates_to_body_rates

logger = logging.getLogger("attitude_controller")


class LeeAttitudeController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None, orient="NED"):
        super().__init__(config, mass, inertia, gravity)
        self.orient = orient
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Lee attitude controller
        :param command_actions: array of shape (4,) with [thrust, roll, pitch, yaw_rate] attitude command
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Thrust command (normalized thrust input)
        self.wrench_command[2] = (
            (command_actions[0] + 1.0) * self.mass * np.linalg.norm(self.gravity)
        )

        # Set desired angular rates (zero roll/pitch rates, commanded yaw rate)
        self.euler_angle_rates[0:2] = 0.0
        self.euler_angle_rates[2] = command_actions[3]
        self.desired_body_angvel = euler_rates_to_body_rates(
            self.robot_euler_angles, self.euler_angle_rates, self.rotation_matrix_buffer
        )

        # Desired euler angles: commanded roll, commanded pitch, current yaw
        quat_desired = quat_from_euler_xyz(
            command_actions[1], command_actions[2], self.robot_euler_angles[2]
        )
        
        # Compute body torques
        self.wrench_command[3:6] = self.compute_body_torque(
            quat_desired, self.desired_body_angvel
        )

        return self.wrench_command.copy()