"""Step-traversal task."""

import numpy as np

FALL_PENALTY = 10.0


def fitness_step_traversal(positions: np.ndarray, step_height: float, target_x: float, fall_height: float = 0.0) -> float:
    """Return a minimization score for climbing the step and reaching the upper platform."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[0] == 0 or positions.shape[1] < 3:
        raise ValueError("positions must have shape (n, 3) with at least one sample.")

    final_position = positions[-1]
    min_z = float(np.min(positions[:, 2]))
    horizontal_remaining = max(0.0, target_x - float(final_position[0]))
    vertical_remaining = max(0.0, step_height - float(final_position[2]))
    fall_penalty = FALL_PENALTY if min_z < fall_height else 0.0
    return float(horizontal_remaining + vertical_remaining + fall_penalty)
