import pytest
from ariel.simulation.tasks import targeted_locomotion

def test_distance_to_target_basic():
    """Test standard Pythagorean triple (3, 4, 5)."""
    start = (0.0, 0.0)
    target = (3.0, 4.0)
    assert targeted_locomotion.distance_to_target(start, target) == pytest.approx(5.0)

def test_distance_to_target_zero():
    """Test distance when start and target are the same."""
    pos = (12.5, -3.14)
    assert targeted_locomotion.distance_to_target(pos, pos) == 0.0

def test_distance_to_target_negative_coordinates():
    """Test calculation across different quadrants."""
    # From (-2, -2) to (1, 2) is dx=3, dy=4 -> dist=5
    start = (-2.0, -2.0)
    target = (1.0, 2.0)
    assert targeted_locomotion.distance_to_target(start, target) == pytest.approx(5.0)

def test_distance_to_target_commutativity():
    """Test that distance A->B is identical to B->A."""
    p1 = (10.0, 5.0)
    p2 = (-5.0, 20.0)
    dist_a_b = targeted_locomotion.distance_to_target(p1, p2)
    dist_b_a = targeted_locomotion.distance_to_target(p2, p1)
    assert dist_a_b == pytest.approx(dist_b_a)

def test_distance_to_target_large_values():
    """Test stability with larger numbers."""
    start = (0.0, 0.0)
    target = (1e6, 0.0)
    assert targeted_locomotion.distance_to_target(start, target) == pytest.approx(1e6)

@pytest.mark.parametrize("start, target, expected", [
    ((0, 0), (0, 5), 5.0),       # Vertical line
    ((0, 0), (5, 0), 5.0),       # Horizontal line
    ((-1, -1), (-4, -5), 5.0),   # Negative to more negative
    ((1.5, 1.5), (4.5, 5.5), 5.0) # Floats
])
def test_distance_to_target_parametrized(start, target, expected):
    """Parametrized tests for various geometric scenarios."""
    assert targeted_locomotion.distance_to_target(start, target) == pytest.approx(expected)