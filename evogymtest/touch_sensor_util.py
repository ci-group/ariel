import numpy as np
from typing import Tuple, Union

# Stolen from https://github.com/giorgia-nadizar/BBB-QD
def detect_ground_contact(
        robot: np.ndarray,
        ground: np.ndarray,
        voxel_index_to_sensor_index,
        actuator_indices,
        side: float = 0.1,
        offset: float = 0.005
):
    # trim terrain around robot
    current_ground = _extract_local_ground(robot, ground, side)
    for actuator_index in actuator_indices:
        voxel_corner_masses = []
        if actuator_index not in voxel_index_to_sensor_index:
            continue
        for sensor_index in voxel_index_to_sensor_index[actuator_index]:
            voxel_corner_masses.append((robot[0][sensor_index], robot[1][sensor_index]))
        if _detect_ground_contact_voxel(np.asarray(voxel_corner_masses).transpose(), current_ground, side, offset):
            return True
    return False


def _extract_local_ground(robot: np.ndarray, ground: np.ndarray, side: float = 0.1) -> np.ndarray:
    min_robot_x, max_robot_x = np.min(robot[0]), np.max(robot[0])
    return ground[:, (ground[0] > min_robot_x - side) & (ground[0] < max_robot_x + side)]


def _detect_ground_contact_voxel(voxel: np.ndarray,
                                 ground: np.ndarray,
                                 side: float = 0.1,
                                 offset: float = 0.005) -> bool:
    # trim terrain around current voxel
    ground = _extract_local_ground(voxel, ground, side)

    # check if there is terrain below the robot
    if len(ground[0]) == 0:
        return False

    # check if terrain is locally flat and compute collision
    locally_flat, contact = _detect_ground_contact_flat_terrain(voxel, ground, offset)
    if locally_flat and contact:
        return True

    # check if some masses are surely/might be in contact
    potential_contact = False
    lowest_terrain_point = np.min(ground[1])
    highest_terrain_point = np.max(ground[1])
    for mass_y in voxel[1]:
        if mass_y <= (lowest_terrain_point + offset):
            return True
        if mass_y <= (highest_terrain_point + offset):
            potential_contact = True
            break
    if not potential_contact:
        return False

    voxel_pairs_idx = [(2, 3), (0, 2), (1, 3), (0, 1)]  # ranked to inspect the most likely ones first
    for mass_idx in range(len(ground.transpose()) - 1):
        ground_mass_1 = ground.transpose()[mass_idx]
        ground_mass_2 = ground.transpose()[mass_idx + 1]
        for voxel_idx_1, voxel_idx_2 in voxel_pairs_idx:
            voxel_mass_1 = voxel.transpose()[voxel_idx_1]
            voxel_mass_2 = voxel.transpose()[voxel_idx_2]
            segment_intersection = _detect_segments_intersection(ground_mass_1, ground_mass_2, voxel_mass_1,
                                                                 voxel_mass_2)
            if segment_intersection:
                return True

    return False


def _detect_ground_contact_flat_terrain(robot: np.ndarray,
                                        ground: np.ndarray,
                                        offset: float = 0.005) -> Tuple[bool, Union[bool, None]]:
    unique_positive_ground_y_values = np.unique(ground[1])
    if len(unique_positive_ground_y_values) == 1:
        return True, np.min(robot[1]) <= (unique_positive_ground_y_values[0] + offset)
    else:
        return False, None


def _on_segment(p, q, r) -> bool:
    # given three collinear points p, q, r, the function checks if point q lies on the line segment 'pr'
    return ((q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and
            (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1])))


def _orientation(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
    # to find the orientation of an ordered triplet (p,q,r) function returns the following values:
    # 0 : Collinear points
    # 1 : Clockwise points
    # -1 : Counterclockwise
    return np.sign(float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))


def _detect_segments_intersection(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> bool:
    # find the 4 orientations required for
    # the general and special cases
    o1 = _orientation(p1, q1, p2)
    o2 = _orientation(p1, q1, q2)
    o3 = _orientation(p2, q2, p1)
    o4 = _orientation(p2, q2, q1)

    return (((o1 != o2) and (o3 != o4)) or  # general case
            (o1 == 0 and _on_segment(p1, p2, q1)) or  # p1 , q1 and p2 are collinear and p2 lies on segment p1q1
            (o2 == 0 and _on_segment(p1, q2, q1)) or  # p1 , q1 and q2 are collinear and q2 lies on segment p1q1
            (o3 == 0 and _on_segment(p2, p1, q2)) or  # p2 , q2 and p1 are collinear and p1 lies on segment p2q2
            (o4 == 0 and _on_segment(p2, q1, q2)))  # p2 , q2 and q1 are collinear and q1 lies on segment p2q2
