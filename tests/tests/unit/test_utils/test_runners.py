import pytest
import mujoco
import numpy as np
from unittest.mock import patch
from ariel.utils import runners
@pytest.fixture
def mj_model_and_data():
    """
    Creates a simple MuJoCo model with one actuator to test controls.
    """
    xml = """
    <mujoco>
        <worldbody>
            <body name="robot" pos="0 0 1">
                <joint name="robot_joint" type="free"/>
                <geom type="sphere" size="0.1"/>
            </body>
        </worldbody>
        <actuator>
            <!-- Define one motor so data.ctrl has size > 0 -->
            <motor name="act1" joint="robot_joint"/>
        </actuator>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data

def test_simple_runner_runs_for_duration(mj_model_and_data):
    """Test that the simulation runs for the requested duration."""
    model, data = mj_model_and_data
    target_duration = 0.1  # Keep it short for unit tests
    
    runners.simple_runner(model, data, duration=target_duration, steps_per_loop=10)
    
    # Simulation should stop once time >= duration
    assert data.time >= target_duration
    # It shouldn't overshoot massively (timestep is usually 0.002)
    assert data.time < target_duration + 1.0

def test_simple_runner_populates_controls(mj_model_and_data):
    """Test that random controls are applied to data.ctrl."""
    model, data = mj_model_and_data
    
    # Ensure controls are initially zero
    data.ctrl[:] = 0.0
    assert np.all(data.ctrl == 0.0)
    
    # Run runner (even for a tiny step)
    runners.simple_runner(model, data, duration=0.01)
    
    # Controls should now be randomized (non-zero)
    # The probability of RNG.normal producing exact zeros for all elements is negligible
    assert not np.all(data.ctrl == 0.0)
    assert len(data.ctrl) == 1  # Matches our XML fixture

def test_simple_runner_resets_data(mj_model_and_data):
    """
    Test that mj_resetData is called.
    We verify this by setting data.time > duration before calling the runner.
    If reset happens, time goes to 0 and loop runs.
    If reset fails, loop condition (time < duration) fails immediately.
    """
    model, data = mj_model_and_data
    duration = 0.1
    
    # Artificially set time past the duration
    data.time = 100.0
    
    runners.simple_runner(model, data, duration=duration)
    
    # If reset worked, time should be approx duration (0.1), not 100
    assert data.time < 50.0 
    assert data.time >= duration

def test_simple_runner_step_arguments(mj_model_and_data):
    """
    Verify that steps_per_loop is correctly passed to mujoco.mj_step.
    We mock mujoco.mj_step to inspect calls.
    """
    model, data = mj_model_and_data
    steps_arg = 42
    duration = 0.05
    
    with patch('mujoco.mj_step') as mock_step:
        # Mock side effect to increment time, otherwise loop runs forever
        def side_effect(m, d, nstep):
            d.time += 1.0 
            
        mock_step.side_effect = side_effect
        
        runners.simple_runner(
            model, data, 
            duration=duration, 
            steps_per_loop=steps_arg
        )
        
        # Verify the call
        mock_step.assert_called()
        _, kwargs = mock_step.call_args
        
        # Check that 'nstep' kwarg matched our input
        assert kwargs['nstep'] == steps_arg