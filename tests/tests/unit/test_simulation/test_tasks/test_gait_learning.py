import pytest
from ariel.simulation.tasks import gait_learning

# --- Tests for xy_displacement ---

def test_xy_displacement_basic():
    """Test simple integer coordinates."""
    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    # 3-4-5 triangle
    assert gait_learning.xy_displacement(p1, p2) == pytest.approx(5.0)

def test_xy_displacement_zero():
    """Test displacement between identical points."""
    p1 = (10.5, -5.2)
    assert gait_learning.xy_displacement(p1, p1) == 0.0

def test_xy_displacement_negative_coords():
    """Test displacement across quadrants."""
    p1 = (-2.0, -2.0)
    p2 = (1.0, 2.0)
    # dx = 3, dy = 4 -> dist = 5
    assert gait_learning.xy_displacement(p1, p2) == pytest.approx(5.0)

def test_xy_displacement_commutativity():
    """Test that distance A->B is same as B->A."""
    p1 = (10.0, 5.0)
    p2 = (3.0, 1.0)
    dist1 = gait_learning.xy_displacement(p1, p2)
    dist2 = gait_learning.xy_displacement(p2, p1)
    assert dist1 == pytest.approx(dist2)

# --- Tests for x_speed ---

def test_x_speed_positive_movement():
    """Test movement in positive x direction."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 5.0) # y changes but shouldn't affect x_speed
    dt = 2.0
    # dx = 10, dt = 2 -> speed = 5
    assert gait_learning.x_speed(p1, p2, dt) == pytest.approx(5.0)

def test_x_speed_negative_movement():
    """Test movement in negative x direction (should return absolute speed)."""
    p1 = (10.0, 0.0)
    p2 = (0.0, 0.0)
    dt = 5.0
    # dx = -10, abs(dx) = 10, dt = 5 -> speed = 2
    assert gait_learning.x_speed(p1, p2, dt) == pytest.approx(2.0)

def test_x_speed_no_movement():
    """Test zero movement in x."""
    p1 = (5.0, 0.0)
    p2 = (5.0, 10.0) # Moving in y only
    dt = 1.0
    assert gait_learning.x_speed(p1, p2, dt) == 0.0

def test_x_speed_zero_dt():
    """Test handling of zero time difference."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    dt = 0.0
    assert gait_learning.x_speed(p1, p2, dt) == 0.0

def test_x_speed_negative_dt():
    """Test handling of negative time difference."""
    p1 = (0.0, 0.0)
    p2 = (10.0, 0.0)
    dt = -1.0
    assert gait_learning.x_speed(p1, p2, dt) == 0.0

# --- Tests for y_speed ---

def test_y_speed_positive_movement():
    """Test movement in positive y direction."""
    p1 = (0.0, 0.0)
    p2 = (5.0, 10.0) # x changes but shouldn't affect y_speed
    dt = 2.0
    # dy = 10, dt = 2 -> speed = 5
    assert gait_learning.y_speed(p1, p2, dt) == pytest.approx(5.0)

def test_y_speed_negative_movement():
    """Test movement in negative y direction (should return absolute speed)."""
    p1 = (0.0, 10.0)
    p2 = (0.0, 0.0)
    dt = 5.0
    # dy = -10, abs(dy) = 10, dt = 5 -> speed = 2
    assert gait_learning.y_speed(p1, p2, dt) == pytest.approx(2.0)

def test_y_speed_no_movement():
    """Test zero movement in y."""
    p1 = (0.0, 5.0)
    p2 = (10.0, 5.0) # Moving in x only
    dt = 1.0
    assert gait_learning.y_speed(p1, p2, dt) == 0.0

def test_y_speed_zero_dt():
    """Test handling of zero time difference."""
    p1 = (0.0, 0.0)
    p2 = (0.0, 10.0)
    dt = 0.0
    assert gait_learning.y_speed(p1, p2, dt) == 0.0

# --- Parametrized Tests (Optional but Good Practice) ---

@pytest.mark.parametrize("p1, p2, expected", [
    ((0,0), (3,4), 5.0),
    ((1,1), (4,5), 5.0),
    ((0,0), (0,0), 0.0),
    ((-1,-1), (-4,-5), 5.0),
])
def test_xy_displacement_parametrized(p1, p2, expected):
    assert gait_learning.xy_displacement(p1, p2) == pytest.approx(expected)