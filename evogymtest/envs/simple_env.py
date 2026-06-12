from typing import Optional, Dict, Any

import gymnasium as gym
from gymnasium import spaces
from evogym import EvoWorld, utils
from evogym.envs import EvoGymBase

import numpy as np
import os

from sensors import Sensors


class SimpleWalkerEnvClass(EvoGymBase):

    def __init__(self, body):
        self.world = EvoWorld.from_json(os.path.join('evogymtest', 'world_data', 'simple_environment.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=utils.get_full_connectivity(body))
        EvoGymBase.__init__(self, self.world)

        num_actuators = self.get_actuator_indices('robot').size
        
        # Use get_observation to get the size
        obs = self.get_observation()
        obs_size = obs.size

        self.action_space = spaces.Box(low=0.6, high=1.6, shape=(num_actuators,), dtype=np.float64)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_size,), dtype=np.float64)

        self.default_viewer.track_objects('robot')
        self.sensors = Sensors(body)

    def get_observation(self):
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
        ))
        return obs

    def get_action(self, controller):
        sensor_input = self.sensors.get_input_from_sensors(self.sim)
        return self.action_space.sample()

    def step(self, action: np.ndarray):
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        super().step({'robot': action})
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])

        return None, reward, False, False, {}

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset()
        obs = self.get_observation()
        return obs, {}
