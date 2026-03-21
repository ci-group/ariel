"""Targeted locomotion."""
import numpy as np

def distance_to_target(initial_position: tuple[float, float], target_position: tuple[float, float]) -> float:
    return (
        (initial_position[0] - target_position[0]) ** 2
        + (initial_position[1] - target_position[1]) ** 2
    ) ** 0.5

def fitness_delta_distance(initial_pos: np.ndarray, final_pos: np.ndarray, target_pos: np.ndarray) -> float:
    """Rewards closing distance. Flipped for minimization (lower score is better)."""
    initial_dist = np.linalg.norm(initial_pos - target_pos)
    final_dist = np.linalg.norm(final_pos - target_pos)
    
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
    straight_line_displacement = np.linalg.norm(final_pos - initial_pos)
    wasted_movement = total_path_length - straight_line_displacement
    
    path_penalty = wasted_movement * 0.5 
    return float(delta_dist + path_penalty)