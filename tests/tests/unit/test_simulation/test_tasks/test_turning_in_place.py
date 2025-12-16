import pytest
import numpy as np
from ariel.simulation.tasks.turning_in_place import turning_in_place

def test_turning_in_place_empty_and_single():
    """Test that insufficient history returns 0.0."""
    assert turning_in_place([]) == 0.0
    assert turning_in_place([(0.0, 0.0)]) == 0.0

def test_turning_in_place_straight_line():
    """Test that moving in a straight line results in 0 turning."""
    # Moving along X axis: Headings are all 0.0
    history = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    score = turning_in_place(history)
    assert score == 0.0

def test_turning_in_place_90_degree_turn():
    """
    Test a simple 90 degree turn.
    Path: (0,0) -> (1,0) -> (1,1)
    Leg 1 Heading: 0 radians
    Leg 2 Heading: pi/2 radians
    Turning Angle: pi/2
    Displacement: sqrt(1^2 + 1^2) = sqrt(2) approx 1.414
    Expected: (pi/2) / (1 + sqrt(2))
    """
    history = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    score = turning_in_place(history)
    
    expected_angle = np.pi / 2
    expected_displacement = np.sqrt(2)
    expected_score = expected_angle / (1.0 + expected_displacement)
    
    assert score == pytest.approx(expected_score)

def test_turning_in_place_u_turn():
    """
    Test a 180 degree turn (back and forth).
    Path: (0,0) -> (1,0) -> (0,0)
    Leg 1 Heading: 0
    Leg 2 Heading: pi (or -pi)
    Turning Angle: pi
    Displacement: Distance from start(0,0) to end(0,0) = 0
    Expected: pi / (1 + 0) = pi
    """
    history = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    score = turning_in_place(history)
    assert score == pytest.approx(np.pi)

def test_turning_in_place_drift_penalty():
    """
    Test that drifting away penalizes the score.
    Compare a U-turn that returns to origin vs a U-turn that ends far away.
    """
    # Case A: Perfect return to origin
    # (0,0) -> (1,0) -> (0,0). Turn = pi. Disp = 0. Score = pi.
    path_a = [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    score_a = turning_in_place(path_a)

    # Case B: U-Turn but drifting far away
    # (0,0) -> (1,0) -> (10, 0). 
    # Leg 1 Heading: 0. 
    # Leg 2 Heading: 0. Wait, this isn't a turn.
    
    # Let's construct a turn that drifts.
    # (0,0) -> (1,0) -> (1, 10).
    # Leg 1: 0 rad. Leg 2: pi/2 rad. Turn = pi/2.
    # Displacement: dist((0,0), (1,10)) = sqrt(101) ~ 10.05
    path_drift = [(0.0, 0.0), (1.0, 0.0), (1.0, 10.0)]
    score_drift = turning_in_place(path_drift)
    
    # Compare with a non-drifting 90 degree turn
    # (0,0) -> (1,0) -> (1,0) (Stopped? No heading undefined).
    # (0,0) -> (1,0) -> (1, 0.001) (Tiny drift).
    # Ideally, we compare against the theoretical max for that angle.
    
    # Calculate score manually for drift case
    angle = np.pi / 2
    disp = np.sqrt(1**2 + 10**2)
    expected_drift_score = angle / (1.0 + disp)
    
    assert score_drift == pytest.approx(expected_drift_score)
    # Ensure drift score is significantly lower than a pure rotation would be
    # (Pure rotation has disp=0, so score = angle)
    assert score_drift < angle

def test_full_circle_unwrap():
    """
    Test a full circle path to ensure np.unwrap handles the 360 degree boundary.
    Path: Diamond shape (0,0) -> (1,1) -> (0,2) -> (-1,1) -> (0,0)
    This is roughly a circle.
    Headings:
    1. (1,1)-(0,0) = (1,1) -> 45 deg (pi/4)
    2. (0,2)-(1,1) = (-1,1) -> 135 deg (3pi/4) -> Delta = +pi/2
    3. (-1,1)-(0,2) = (-1,-1) -> -135 deg (-3pi/4).
       Jump from 135 to -135 is -270 deg raw. Unwrap should make this +90 deg (+pi/2).
    4. (0,0)-(-1,1) = (1,-1) -> -45 deg (-pi/4).
       Jump from -135 to -45 is +90 deg (+pi/2).
    
    Total Turn should be sum of abs diffs: pi/2 + pi/2 + pi/2 = 3pi/2 (270 degrees total turn in this path).
    Displacement: 0 (Returns to start).
    Expected Score: 3pi/2.
    """
    history = [
        (0.0, 0.0),
        (1.0, 1.0),
        (0.0, 2.0),
        (-1.0, 1.0),
        (0.0, 0.0)
    ]
    score = turning_in_place(history)
    
    expected_turn = 3 * (np.pi / 2)
    assert score == pytest.approx(expected_turn)