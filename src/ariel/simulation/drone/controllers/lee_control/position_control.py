import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController, calculate_desired_orientation_for_position_velocity_control

logger = logging.getLogger("lee_position_controller")


class LeePositionController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None, orient="NED"):
        super().__init__(config, mass, inertia, gravity)
        self.orient = orient
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Lee position controller
        :param command_actions: array of shape (4,) with [px, py, pz, yaw] position setpoint and yaw command
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Compute desired acceleration
        self.accel = self.compute_acceleration(
            setpoint_position=command_actions[0:3],
            setpoint_velocity=np.zeros(3),  # Zero velocity setpoint
        )
        
        # Convert acceleration to forces (WORLD frame)
        # self.accel is in world frame, self.gravity is in world frame
        forces = (self.accel - self.gravity) * self.mass  # Forces needed to overcome gravity and achieve desired acceleration
        
        # DEBUG: Print force calculation details (disabled for performance)
        # if hasattr(self, '_debug_count') == False:
        #     self._debug_count = 0
        # self._debug_count += 1
        # if self._debug_count % 50 == 1:  # Print every 50 calls
        #     print(f"POSITION CONTROL DEBUG:")
        #     print(f"  setpoint_position: {command_actions[0:3]}")
        #     print(f"  robot_position: {self.robot_position}")
        #     print(f"  position_error: {command_actions[0:3] - self.robot_position}")
        #     print(f"  accel command: {self.accel}")
        #     print(f"  gravity: {self.gravity}")
        #     print(f"  accel - gravity: {self.accel - self.gravity}")
        #     print(f"  mass: {self.mass}")
        #     print(f"  forces: {forces}")
        #     print(f"  orient: {self.orient}")
        
        # Set complete force command (WORLD frame forces)
        self.wrench_command[0:3] = forces

        # Calculate desired orientation from forces and yaw setpoint
        self.desired_quat = calculate_desired_orientation_for_position_velocity_control(
            forces, command_actions[3], self.rotation_matrix_buffer, self.orient
        )

        # Zero angular velocity setpoint for position control
        self.euler_angle_rates.fill(0.0)
        self.desired_body_angvel.fill(0.0)

        # Compute body torques
        self.wrench_command[3:6] = self.compute_body_torque(
            self.desired_quat, self.desired_body_angvel
        )

        return self.wrench_command.copy()