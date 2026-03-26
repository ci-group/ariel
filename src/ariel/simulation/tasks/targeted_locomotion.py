"""Targeted locomotion."""
import numpy as np

def _xy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance in the horizontal plane (x, y)."""
    return float(np.linalg.norm(a[:2] - b[:2]))


def distance_to_target(final_position: np.ndarray, target_position: np.ndarray) -> float:
    """Remaining planar distance to target (lower is better for minimization)."""
    res = float(np.linalg.norm(final_position[:2] - target_position[:2]))
    return res

def fitness_delta_distance(initial_pos: np.ndarray, final_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """Rewards closing distance. Flipped for minimization (lower score is better)."""
    initial_dist = _xy_distance(initial_pos, target_pos)
    final_dist = _xy_distance(final_pos, target_pos)
    
    return float(final_dist - initial_dist)

def fitness_distance_and_efficiency(initial_pos: np.ndarray, final_pos: np.ndarray, target_pos: np.ndarray, total_control_effort: float) -> float:
    delta_dist = fitness_delta_distance(initial_pos, final_pos, target_pos)
    
    # FIXED: Added penalty (Higher score = worse)
    effort_penalty = total_control_effort * 0.001 
    return float(delta_dist + effort_penalty)

def fitness_survival_and_locomotion(initial_pos: np.ndarray, final_pos: np.ndarray, target_pos: np.ndarray, min_z_height: float) -> float:
    if min_z_height < 0.05: 
        return 10.0 
        
    return fitness_delta_distance(initial_pos, final_pos, target_pos)

def fitness_direct_path(initial_pos: np.ndarray, final_pos: np.ndarray, target_pos: np.ndarray, total_path_length: float) -> float:
    delta_dist = fitness_delta_distance(initial_pos, final_pos, target_pos)
    straight_line_displacement = _xy_distance(final_pos, initial_pos)
    wasted_movement = total_path_length - straight_line_displacement
    
    path_penalty = wasted_movement * 0.5 
    return float(delta_dist + path_penalty)


def fitness_speed_to_target(
    time_to_target: float | None,
    duration: float,
    min_distance_to_target: float,
) -> float:
    """Reward fast target arrival irrespective of path shape.

    Lower score is better (minimization objective):
    - Reached target: score in [0, 1], proportional to arrival time.
    - Not reached: score > 1, penalized by closest achieved distance.
    """
    safe_duration = max(float(duration), 1e-6)
    if time_to_target is not None:
        return float(np.clip(time_to_target / safe_duration, 0.0, 1.0))

    return float(1.0 + min_distance_to_target)