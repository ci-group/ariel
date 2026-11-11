"""Gap-crossing task."""

import numpy as np

FALL_PENALTY = 10.0
RIGHT_PLATFORM_MARGIN = 0.20
FINAL_WINDOW_FRACTION = 0.10


def fitness_gap_crossing(core_positions: np.ndarray, gap_width: float, fall_height: float = 0.0) -> float:
    """Return a minimization score for moving the core safely onto the right platform."""
    core_positions = np.asarray(core_positions, dtype=float)

    if core_positions.ndim != 2 or core_positions.shape[0] == 0 or core_positions.shape[1] < 3:
        raise ValueError("core_positions must have shape (n, 3) with at least one sample.")

    right_edge = gap_width / 2.0
    target_x = right_edge + RIGHT_PLATFORM_MARGIN

    final_window_start = max(0, int(len(core_positions) * (1.0 - FINAL_WINDOW_FRACTION)))
    final_core_x = float(np.mean(core_positions[final_window_start:, 0]))
    min_core_z = float(np.min(core_positions[:, 2]))

    remaining_distance = max(0.0, target_x - final_core_x)
    fall_penalty = FALL_PENALTY if min_core_z < fall_height else 0.0

    return float(remaining_distance + fall_penalty)