"""Turning in place task."""

# Third-party libraries
import numpy as np


def turning_in_place(xy_history: list[tuple[float, float]]) -> float:
    """
    Determine the total angle turned by a robot based on its path history.

    Parameters
    ----------
    xy_history : list[tuple]
        The history of x, y coordinates from a simulation i.e. robot path.

    Returns
    -------
    float
        The total angle turned by the robot.
    """
    # Convert to numpy array for easier manipulation
    xy = np.array(xy_history)

    # Require at least two positions to compute turning
    min_history_length = 2
    if len(xy) < min_history_length:
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
    return total_turning_angle / (1.0 + displacement)
