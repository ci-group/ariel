import numpy as np


def turning_in_place(xy_history) -> float:
    """
    Determines the total angle turned by a robot based on its path history.

    Parameters
    -----------
    xy_history : list[tuple]
        The history of x, y coordinates from a simulation i.e. robot path.

    Returns
    --------
    float
        The total angle turned by the robot.
    """

    xy = np.array(xy_history)
    if len(xy) < 2:
        return 0.0

    # Headings from XY positions
    deltas = np.diff(xy, axis=0)
    headings = np.arctan2(deltas[:, 1], deltas[:, 0])
    headings_unwrapped = np.unwrap(headings)

    # Total amount turned (absolute rotation)
    total_turning_angle = np.sum(np.abs(np.diff(headings_unwrapped)))

    # Drift from start position
    displacement = np.linalg.norm(xy[-1] - xy[0])

    # Penalize if robot drifts away
    fitness = total_turning_angle / (1.0 + displacement)

    return fitness
