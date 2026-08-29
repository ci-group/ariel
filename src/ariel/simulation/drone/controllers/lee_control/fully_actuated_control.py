import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController

logger = logging.getLogger("fully_actuated_controller")


class FullyActuatedController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None):
        super().__init__(config, mass, inertia, gravity)
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Fully actuated controller. Input is position and orientation setpoints.
        command_actions = [p_x, p_y, p_z, qx, qy, qz, qw]
        Position setpoint is in the world frame
        Orientation reference is w.r.t world frame
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Normalize quaternion command
        command_actions[3:7] = normalize(command_actions[3:7])
        
        # Compute desired acceleration for position tracking
        self.accel = self.compute_acceleration(
            command_actions[0:3], np.zeros(3)
        )
        forces = self.mass * (self.accel - self.gravity)
        
        # Full force control (all three axes)
        self.wrench_command[0:3] = quat_rotate_inverse(self.robot_orientation, forces)
        
        # Set desired orientation from command
        self.desired_quat = command_actions[3:7].copy()
        
        # Compute body torques for orientation tracking (zero angular velocity setpoint)
        self.wrench_command[3:6] = self.compute_body_torque(
            self.desired_quat, np.zeros(3)
        )
        return self.wrench_command.copy()