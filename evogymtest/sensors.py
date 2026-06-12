import math
from collections import defaultdict
from typing import Any

import numpy as np

from touch_sensor_util import detect_ground_contact

class Sensors:
    sensor_grid_to_sensor_index: dict
    voxel_index_to_sensor_index: defaultdict[Any, list]

    def __init__(self, robot_structure):
        self.robot_structure = robot_structure
        self._get_sensor_grid()
        self._get_voxel_to_sensor_index()

    def _get_sensor_grid(self):
        sensor_index = 0
        self.sensor_grid_to_sensor_index = {}
        for x_sensor in range(6):
            for y_sensor in range(6):
                if x_sensor < 5 and y_sensor < 5 and self.robot_structure[x_sensor, y_sensor] > 0:
                    # Add sensors to sensor grid if it doesn't exist yet
                    for dx in [0, 1]:
                        for dy in [0, 1]:
                            if (x_sensor + dx, y_sensor + dy) in self.sensor_grid_to_sensor_index:
                                continue
                            self.sensor_grid_to_sensor_index[(x_sensor + dx, y_sensor + dy)] = sensor_index
                            sensor_index += 1

    def _get_voxel_to_sensor_index(self):
        self.voxel_index_to_sensor_index = defaultdict(list)
        for x_voxel in range(5):
            for y_voxel in range(5):
                if self.robot_structure[x_voxel, y_voxel] == 0:
                    continue
                for dx in [0, 1]:
                    for dy in [0, 1]:
                        self.voxel_index_to_sensor_index[x_voxel * 5 + y_voxel].append(
                            self.sensor_grid_to_sensor_index[(x_voxel + dx, y_voxel + dy)])

    def _get_time_features(self, current_time):
        cyc = current_time % 25
        theta = 2 * np.pi * cyc / 25
        return [np.sin(theta), np.cos(theta)]

    def get_input_from_sensors(self, sim):
        current_time = sim.get_time()
        robot_positions = sim.object_pos_at_time(current_time, 'robot')
        robot_velocities = sim.object_vel_at_time(current_time, 'robot')
        ground_positions = sim.object_pos_at_time(current_time, 'ground')
        actuator_indices = sim.get_actuator_indices('robot')

        input_vectors = []

        for actuator_index in actuator_indices:
            features = []

            # Actuator features
            features.extend(
                self._get_input_actuator(actuator_index, robot_positions, robot_velocities, ground_positions)
            )

            # Time features
            features.extend(self._get_time_features(current_time))

            input_vectors.append(np.array(features))

        return input_vectors

    def _get_input_actuator(self, actuator_index, robot_positions, robot_velocities, ground_positions):
        voxel_sizes = []
        voxel_velocities = []
        voxel_contacts = []
        actuator_x, actuator_y = (actuator_index // 5, actuator_index % 5)
        neighbors = [0]
        for i in range(1, 3):
            neighbors.append(i)
            neighbors.append(-i)
        neighbors.sort()
        for x_neighbor in neighbors:
            for y_neighbor in neighbors:
                voxel_size, voxel_velocity_x, voxel_velocity_y = self._get_input_from_voxel(
                    actuator_x + x_neighbor,
                    actuator_y + y_neighbor,
                    robot_positions,
                    robot_velocities)
                contact = detect_ground_contact(
                    robot_positions, ground_positions,
                    self.voxel_index_to_sensor_index,
                    [(actuator_x + x_neighbor) * 5 + (actuator_y + y_neighbor)]
                )
                voxel_contact = 1.0 if contact else 0.0
                voxel_sizes.append(voxel_size)
                voxel_velocities.append(voxel_velocity_x)
                voxel_velocities.append(voxel_velocity_y)
                voxel_contacts.append(voxel_contact)
        return np.array(voxel_sizes + voxel_velocities + voxel_contacts)

    def _get_input_package(self, actuator_index, robot_positions, package_positions):
        if actuator_index not in self.voxel_index_to_sensor_index.keys():
            return np.array([math.inf, math.inf])
        sensor_indices = self.voxel_index_to_sensor_index[actuator_index]
        minimum_x_distance = math.inf
        minimum_y_distance = math.inf
        for sensor_index in sensor_indices:
            robot_sensor_position = (robot_positions[0][sensor_index], robot_positions[1][sensor_index])
            for package_index in range(len(package_positions[0])):
                package_sensor_position = (package_positions[0][package_index], package_positions[1][package_index])

                x_distance = package_sensor_position[0] - robot_sensor_position[0]
                y_distance = package_sensor_position[1] - robot_sensor_position[1]

                if abs(x_distance) < abs(minimum_x_distance):
                    minimum_x_distance = x_distance
                if abs(y_distance) < abs(minimum_y_distance):
                    minimum_y_distance = y_distance
        return np.array([minimum_x_distance, minimum_y_distance])

    def _get_input_from_voxel(self, x, y, positions, velocities):
        index = x * 5 + y
        if (x < 0 or x >= 5 or
                y < 0 or y >= 5 or
                index not in self.voxel_index_to_sensor_index.keys()):
            return 0, 0, 0

        sensor_indices = self.voxel_index_to_sensor_index[index]
        corners = []
        velocities_x = []
        velocities_y = []
        for sensor_index in sensor_indices:
            corners.append((positions[0][sensor_index], positions[1][sensor_index]))
            velocities_x.append(velocities[0][sensor_index])
            velocities_y.append(velocities[1][sensor_index])

        return (Sensors.rectangle_size(corners),
                sum(velocities_x) / len(velocities_x),
                sum(velocities_y) / len(velocities_y))

    @staticmethod
    def distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    @staticmethod
    def rectangle_size(corners):
        a, b, c, d = corners
        width = Sensors.distance(a, b)
        height = Sensors.distance(a, c)
        return width * height