"""Test: simulation task fitness functions."""

import numpy as np
import pytest

from ariel.simulation.tasks.gait_learning import x_speed, xy_displacement, y_speed
from ariel.simulation.tasks.targeted_locomotion import (
    distance_to_target,
    fitness_delta_distance,
    fitness_direct_path,
    fitness_distance_and_efficiency,
    fitness_speed_to_target,
    fitness_survival_and_locomotion,
)
from ariel.simulation.tasks.turning_in_place import turning_in_place


# ---------------------------------------------------------------------------
# gait_learning
# ---------------------------------------------------------------------------


def test_xy_displacement_zero() -> None:
    """Displacement between identical points is zero."""
    assert xy_displacement((0.0, 0.0), (0.0, 0.0)) == pytest.approx(0.0)


def test_xy_displacement_known_value() -> None:
    """3-4-5 triangle gives displacement of 5."""
    assert xy_displacement((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)


def test_xy_displacement_symmetric() -> None:
    """Displacement is symmetric: d(a,b) == d(b,a)."""
    a, b = (1.0, 2.0), (4.0, 6.0)
    assert xy_displacement(a, b) == pytest.approx(xy_displacement(b, a))


def test_x_speed_zero_dt_returns_zero() -> None:
    """x_speed returns 0 when dt is zero to avoid division by zero."""
    assert x_speed((0.0, 0.0), (5.0, 0.0), dt=0.0) == pytest.approx(0.0)


def test_x_speed_known_value() -> None:
    """x_speed = |dx| / dt."""
    assert x_speed((0.0, 0.0), (3.0, 0.0), dt=1.0) == pytest.approx(3.0)


def test_y_speed_zero_dt_returns_zero() -> None:
    """y_speed returns 0 when dt is zero."""
    assert y_speed((0.0, 0.0), (0.0, 5.0), dt=0.0) == pytest.approx(0.0)


def test_y_speed_known_value() -> None:
    """y_speed = |dy| / dt."""
    assert y_speed((0.0, 0.0), (0.0, 4.0), dt=2.0) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# targeted_locomotion
# ---------------------------------------------------------------------------


def test_distance_to_target_at_target() -> None:
    """distance_to_target is zero when final position equals target."""
    pos = np.array([1.0, 1.0, 0.0])
    assert distance_to_target(pos, pos) == pytest.approx(0.0)


def test_distance_to_target_known_value() -> None:
    """distance_to_target ignores the z axis and uses x,y only."""
    final = np.array([3.0, 4.0, 999.0])
    target = np.array([0.0, 0.0, 0.0])
    assert distance_to_target(final, target) == pytest.approx(5.0)


def test_fitness_delta_distance_negative_when_closer() -> None:
    """fitness_delta_distance is negative when the robot moves toward target."""
    initial = np.array([10.0, 0.0, 0.0])
    final = np.array([5.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    assert fitness_delta_distance(initial, final, target) < 0.0


def test_fitness_delta_distance_positive_when_farther() -> None:
    """fitness_delta_distance is positive when the robot moves away from target."""
    initial = np.array([1.0, 0.0, 0.0])
    final = np.array([10.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    assert fitness_delta_distance(initial, final, target) > 0.0


def test_fitness_distance_and_efficiency_adds_effort_penalty() -> None:
    """fitness_distance_and_efficiency > fitness_delta_distance for positive effort."""
    initial = np.array([5.0, 0.0, 0.0])
    final = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    delta = fitness_delta_distance(initial, final, target)
    with_effort = fitness_distance_and_efficiency(initial, final, target, total_control_effort=1000.0)
    assert with_effort > delta


def test_fitness_survival_penalty_when_too_low() -> None:
    """fitness_survival_and_locomotion returns 10.0 when z < 0.05."""
    initial = np.array([0.0, 0.0, 0.0])
    final = np.array([1.0, 0.0, 0.0])
    target = np.array([5.0, 0.0, 0.0])
    result = fitness_survival_and_locomotion(initial, final, target, min_z_height=0.01)
    assert result == pytest.approx(10.0)


def test_fitness_survival_normal_when_above_threshold() -> None:
    """fitness_survival_and_locomotion delegates to delta_distance above z threshold."""
    initial = np.array([5.0, 0.0, 0.0])
    final = np.array([0.0, 0.0, 0.0])
    target = np.array([0.0, 0.0, 0.0])
    result = fitness_survival_and_locomotion(initial, final, target, min_z_height=0.1)
    expected = fitness_delta_distance(initial, final, target)
    assert result == pytest.approx(expected)


def test_fitness_speed_to_target_reached() -> None:
    """fitness_speed_to_target returns value in [0, 1] when target reached."""
    result = fitness_speed_to_target(time_to_target=5.0, duration=10.0, min_distance_to_target=0.0)
    assert 0.0 <= result <= 1.0


def test_fitness_speed_to_target_not_reached() -> None:
    """fitness_speed_to_target returns > 1 when target was not reached."""
    result = fitness_speed_to_target(time_to_target=None, duration=10.0, min_distance_to_target=2.0)
    assert result > 1.0


def test_fitness_direct_path_penalty_for_wasted_movement() -> None:
    """fitness_direct_path penalizes path length beyond the straight line."""
    initial = np.array([0.0, 0.0, 0.0])
    final = np.array([5.0, 0.0, 0.0])
    target = np.array([5.0, 0.0, 0.0])
    # Large wasted movement should push score up (worse)
    score_long = fitness_direct_path(initial, final, target, total_path_length=100.0)
    score_short = fitness_direct_path(initial, final, target, total_path_length=5.0)
    assert score_long > score_short


# ---------------------------------------------------------------------------
# turning_in_place
# ---------------------------------------------------------------------------


def test_turning_in_place_empty_returns_zero() -> None:
    """turning_in_place with fewer than 2 positions returns 0."""
    assert turning_in_place([]) == pytest.approx(0.0)
    assert turning_in_place([(0.0, 0.0)]) == pytest.approx(0.0)


def test_turning_in_place_straight_line_low_score() -> None:
    """A straight-line path produces a lower score than a turning path."""
    straight = [(float(i), 0.0) for i in range(10)]
    result = turning_in_place(straight)
    assert isinstance(result, float)
    assert result >= 0.0


def test_turning_in_place_circle_high_score() -> None:
    """A roughly circular path produces a higher score than a straight line."""
    n = 40
    angles = [2 * 3.14159 * i / n for i in range(n)]
    circle = [(np.cos(a), np.sin(a)) for a in angles]
    straight = [(float(i) * 0.01, 0.0) for i in range(n)]

    circle_score = turning_in_place(circle)
    straight_score = turning_in_place(straight)
    assert circle_score > straight_score
