import numpy as np
import logging
from .lee_math_utils import *
from .base_lee_controller import BaseLeeController

logger = logging.getLogger("rates_controller")


class LeeRatesController(BaseLeeController):
    def __init__(self, config, mass=1.0, inertia=None, gravity=None):
        super().__init__(config, mass, inertia, gravity)
        self.init_controller_gains()

    def update(self, command_actions):
        """
        Lee rates controller (direct angular rate control)
        :param command_actions: array of shape (4,) with [thrust, wx, wy, wz] thrust and angular rate commands
        :return: wrench command [fx, fy, fz, tx, ty, tz]
        """
        command_actions = np.array(command_actions)
        self.reset_commands()
        
        # Direct thrust command
        self.wrench_command[2] = (command_actions[0] - np.linalg.norm(self.gravity)) * self.mass
        
        # Direct angular rate control - maintain current orientation, track desired rates
        self.wrench_command[3:6] = self.compute_body_torque(
            self.robot_orientation, command_actions[1:4]
        )

        return self.wrench_command.copy()