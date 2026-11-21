"""Gap-crossing task with passive stability validation."""

import numpy as np

PASSIVE_SECONDS = 3.0
PASSIVE_SETTLING_SECONDS = 0.5
EPISODE_DURATION = 15.0

STABILITY_PENALTY = 10.0
MAX_PASSIVE_PLANAR_DISPLACEMENT = 0.08
MAX_PASSIVE_VERTICAL_DROP = 0.04

APPROACH_MARGIN = 0.20
CROSSING_MARGIN = 0.20
POST_CROSS_DISTANCE = 0.30
FINAL_WINDOW_FRACTION = 0.10


def fitness_gap_crossing(core_positions: np.ndarray, gap_width: float, fall_height: float = 0.0) -> float:
    """Return a minimization score for stable locomotion to, across, and beyond a gap."""
    core_positions = np.asarray(core_positions, dtype=float)

    if core_positions.ndim != 2 or core_positions.shape[0] == 0 or core_positions.shape[1] < 3:
        raise ValueError("core_positions must have shape (n, 3) with at least one sample.")

    passive_fraction = PASSIVE_SECONDS / EPISODE_DURATION
    settling_fraction = PASSIVE_SETTLING_SECONDS / EPISODE_DURATION

    passive_end = min(len(core_positions) - 1, max(1, int(len(core_positions) * passive_fraction)))
    settling_end = min(passive_end - 1, max(1, int(len(core_positions) * settling_fraction)))

    passive_positions = core_positions[:passive_end]
    stability_positions = core_positions[settling_end:passive_end]
    active_positions = core_positions[passive_end:]

    if len(active_positions) == 0 or len(stability_positions) == 0:
        return STABILITY_PENALTY

    passive_start_xy = passive_positions[0, :2]
    passive_final_window_start = max(0, int(len(passive_positions) * (1.0 - FINAL_WINDOW_FRACTION)))
    passive_final_xy = np.mean(passive_positions[passive_final_window_start:, :2], axis=0)
    passive_planar_displacement = float(np.linalg.norm(passive_final_xy - passive_start_xy))

    settled_window_size = max(1, int(len(stability_positions) * 0.10))
    settled_z = float(np.median(stability_positions[:settled_window_size, 2]))
    min_stable_z = float(np.min(stability_positions[:, 2]))
    passive_vertical_drop = max(0.0, settled_z - min_stable_z)

    unstable = passive_planar_displacement > MAX_PASSIVE_PLANAR_DISPLACEMENT or passive_vertical_drop > MAX_PASSIVE_VERTICAL_DROP

    if unstable:
        stability_excess = max(0.0, passive_planar_displacement - MAX_PASSIVE_PLANAR_DISPLACEMENT) + max(0.0, passive_vertical_drop - MAX_PASSIVE_VERTICAL_DROP)
        return float(STABILITY_PENALTY + stability_excess)

    left_edge = -gap_width / 2.0
    right_edge = gap_width / 2.0

    approach_x = left_edge - APPROACH_MARGIN
    crossing_x = right_edge + CROSSING_MARGIN
    final_target_x = crossing_x + POST_CROSS_DISTANCE

    max_core_x = float(np.max(active_positions[:, 0]))

    final_window_start = max(0, int(len(active_positions) * (1.0 - FINAL_WINDOW_FRACTION)))
    final_window_x = active_positions[final_window_start:, 0]
    final_core_x = float(np.mean(final_window_x))

    if max_core_x < approach_x:
        return float(2.0 + approach_x - max_core_x)

    if max_core_x < crossing_x:
        return float(1.0 + crossing_x - max_core_x)

    if not np.all(final_window_x >= crossing_x):
        return float(1.0 + max(0.0, crossing_x - final_core_x))

    return float(max(0.0, final_target_x - final_core_x))
