"""Low-ceiling traversal task."""

import numpy as np

FALL_PENALTY = 10.0
OUTSIDE_CORRIDOR_PENALTY = 10.0


def fitness_low_ceiling_traversal(positions: np.ndarray, target_x: float, corridor_width: float, fall_height: float = 0.0) -> float:
    """Return a minimization score for traversing the corridor without going around the ceiling."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] < 3:
        raise ValueError("positions must have shape (n, 3) with at least one sample.")

    final_x = float(positions[-1, 0])
    min_z = float(np.min(positions[:, 2]))
    max_abs_y = float(np.max(np.abs(positions[:, 1])))
    remaining_distance = max(0.0, target_x - final_x)
    fall_penalty = FALL_PENALTY if min_z < fall_height else 0.0
    corridor_penalty = OUTSIDE_CORRIDOR_PENALTY if max_abs_y > corridor_width / 2.0 else 0.0
    return float(remaining_distance + fall_penalty + corridor_penalty)
