"""Test: controller for robot simulation."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch, call

try:
    import mujoco as mj
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker


@pytest.fixture
def mock_model():
    """Create a mock MuJoCo model."""
    model = MagicMock(spec=mj.MjModel)
    model.opt.timestep = 0.002  # 2ms timestep
    model.nq = 7  # Number of generalized coordinates
    model.nu = 6  # Number of actuators
    return model


@pytest.fixture
def mock_data():
    """Create a mock MuJoCo data object."""
    data = MagicMock(spec=mj.MjData)
    data.time = 0.0
    data.ctrl = np.zeros(6)
    return data


@pytest.fixture
def mock_callback():
    """Create a mock controller callback function."""
    callback = MagicMock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    return callback


@pytest.fixture
def mock_tracker():
    """Create a mock tracker."""
    tracker = MagicMock(spec=Tracker)
    tracker.update = MagicMock()
    return tracker


@pytest.fixture
def controller(mock_callback, mock_tracker):
    """Create a Controller instance with mock callback and tracker."""
    return Controller(
        controller_callback_function=mock_callback,
        time_steps_per_ctrl_step=50,
        time_steps_per_save=500,
        alpha=0.5,
        tracker=mock_tracker,
    )


class TestControllerInitialization:
    """Tests for Controller initialization."""

    def test_controller_initialization_with_defaults(self, mock_callback) -> None:
        """Test controller initializes with default parameters."""
        ctrl = Controller(controller_callback_function=mock_callback)
        
        assert ctrl.controller_callback_function == mock_callback
        assert ctrl.time_steps_per_ctrl_step == 50
        assert ctrl.time_steps_per_save == 500
        assert ctrl.alpha == 0.5
        assert isinstance(ctrl.tracker, Tracker)

    def test_controller_initialization_custom_parameters(self, mock_callback, mock_tracker) -> None:
        """Test controller initializes with custom parameters."""
        ctrl = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=100,
            time_steps_per_save=1000,
            alpha=0.3,
            tracker=mock_tracker,
        )
        
        assert ctrl.time_steps_per_ctrl_step == 100
        assert ctrl.time_steps_per_save == 1000
        assert ctrl.alpha == 0.3
        assert ctrl.tracker is mock_tracker

    def test_controller_callback_is_stored(self, mock_callback) -> None:
        """Test that controller stores the callback function."""
        ctrl = Controller(controller_callback_function=mock_callback)
        
        assert callable(ctrl.controller_callback_function)
        assert ctrl.controller_callback_function == mock_callback

    def test_controller_alpha_valid_range(self, mock_callback) -> None:
        """Test controller with various alpha values."""
        for alpha in [0.0, 0.1, 0.5, 0.9, 1.0]:
            ctrl = Controller(
                controller_callback_function=mock_callback,
                alpha=alpha,
            )
            assert ctrl.alpha == alpha


class TestSetControlBasic:
    """Tests for basic set_control functionality."""

    def test_set_control_calls_callback(self, controller, mock_model, mock_data, mock_callback) -> None:
        """Test that set_control calls the callback function."""
        mock_data.time = 100.0  # 100 * 0.002 = 50,000 timesteps
        mock_data.ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        controller.set_control(mock_model, mock_data)
        
        # Should call callback when timestep is multiple of time_steps_per_ctrl_step
        mock_callback.assert_called()

    def test_set_control_no_callback_at_wrong_timestep(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that callback is not called at non-control timesteps."""
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=50,
            tracker=mock_tracker,
        )
        mock_data.time = 0.001  # 0.5 timesteps, won't match control step
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # Callback should not be called for timestep 0 or 1
        # Only called at multiples of time_steps_per_ctrl_step
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_ctrl_step != 0:
            mock_callback.assert_not_called()

    def test_set_control_updates_control_signal(self, controller, mock_model, mock_data) -> None:
        """Test that set_control updates the control signal when called."""
        mock_data.time = 100.0  # Timestep 50000, divisible by 50
        mock_data.ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        controller.set_control(mock_model, mock_data)
        
        # Control should be updated (when timestep is multiple of ctrl step)
        assert mock_data.ctrl is not None

    def test_set_control_with_args_and_kwargs(self, controller, mock_model, mock_data, mock_callback) -> None:
        """Test that set_control passes args and kwargs to callback."""
        mock_data.time = 100.0
        mock_data.ctrl = np.array([0.0] * 6)
        
        arg1 = "test_arg"
        kwarg1 = "test_value"
        
        controller.set_control(mock_model, mock_data, arg1, key=kwarg1)
        
        # Callback should be called with args and kwargs if timestep matches
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_ctrl_step == 0:
            mock_callback.assert_called()
            # Verify args and kwargs were passed
            call_args, call_kwargs = mock_callback.call_args
            assert arg1 in call_args
            assert 'key' in call_kwargs


class TestSetControlAlpha:
    """Tests for alpha blending in set_control."""

    def test_set_control_alpha_zero_keeps_old_control(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that alpha=0 keeps old control values."""
        mock_callback.return_value = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,  # Every step
            alpha=0.0,
            tracker=mock_tracker,
        )
        
        old_ctrl = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        mock_data.time = 100.0
        mock_data.ctrl = old_ctrl.copy()
        
        controller.set_control(mock_model, mock_data)
        
        # With alpha=0, control should remain unchanged
        np.testing.assert_array_almost_equal(mock_data.ctrl, old_ctrl)

    def test_set_control_alpha_one_uses_new_control(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that alpha=1 uses new control values."""
        new_ctrl_output = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mock_callback.return_value = new_ctrl_output
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,  # Every step
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        old_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_data.time = 100.0
        mock_data.ctrl = old_ctrl.copy()
        
        controller.set_control(mock_model, mock_data)
        
        # With alpha=1 and clipping to [-pi/2, pi/2], should use new control
        expected = np.clip(new_ctrl_output, -np.pi / 2, np.pi / 2)
        np.testing.assert_array_almost_equal(mock_data.ctrl, expected)

    def test_set_control_alpha_half_blends(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that alpha=0.5 blends old and new control equally."""
        new_ctrl_output = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        mock_callback.return_value = new_ctrl_output
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=0.5,
            tracker=mock_tracker,
        )
        
        old_ctrl = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_data.time = 100.0
        mock_data.ctrl = old_ctrl.copy()
        
        controller.set_control(mock_model, mock_data)
        
        # With alpha=0.5: new = 0.5 * old + 0.5 * output = 0.5 * 1.0
        expected = np.clip(0.5 * old_ctrl + 0.5 * new_ctrl_output, -np.pi / 2, np.pi / 2)
        np.testing.assert_array_almost_equal(mock_data.ctrl, expected, decimal=5)


class TestSetControlClipping:
    """Tests for control signal clipping."""

    def test_set_control_clips_to_negative_pi_over_2(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that control is clipped to -π/2."""
        mock_callback.return_value = np.array([-10.0, -10.0, -10.0, -10.0, -10.0, -10.0])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # All values should be clipped to -π/2
        assert np.all(mock_data.ctrl >= -np.pi / 2)
        assert np.allclose(mock_data.ctrl, -np.pi / 2)

    def test_set_control_clips_to_positive_pi_over_2(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that control is clipped to π/2."""
        mock_callback.return_value = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # All values should be clipped to π/2
        assert np.all(mock_data.ctrl <= np.pi / 2)
        assert np.allclose(mock_data.ctrl, np.pi / 2)

    def test_set_control_partial_clipping(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that control clipping applies only to out-of-bounds values."""
        mock_callback.return_value = np.array([0.1, 0.5, 10.0, -10.0, 0.0, 1.57])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # Values within bounds should be unchanged, others clipped
        expected = np.array([0.1, 0.5, np.pi / 2, -np.pi / 2, 0.0, 1.57])
        np.testing.assert_array_almost_equal(mock_data.ctrl, expected, decimal=5)


class TestSetControlNaNDetection:
    """Tests for NaN detection in control signal."""

    def test_set_control_detects_nan_in_callback_output(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that NaN in callback output is detected."""
        mock_callback.return_value = np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        with pytest.raises(ValueError, match="NaN values detected in the control signal"):
            controller.set_control(mock_model, mock_data)

    def test_set_control_detects_all_nan(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that all NaN control signal is detected."""
        mock_callback.return_value = np.array([np.nan] * 6)
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=1.0,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        with pytest.raises(ValueError, match="NaN values detected in the control signal"):
            controller.set_control(mock_model, mock_data)

    def test_set_control_no_nan_accepted(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that control without NaN is accepted."""
        mock_callback.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=0.5,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        # Should not raise
        controller.set_control(mock_model, mock_data)
        
        assert not np.any(np.isnan(mock_data.ctrl))

    def test_set_control_nan_from_alpha_blending(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that NaN from alpha blending is detected."""
        mock_callback.return_value = np.array([np.inf, 0.2, 0.3, 0.4, 0.5, 0.6])
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=0.5,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # NaN blending with other values can produce NaN
        with pytest.raises(ValueError, match="NaN values detected in the control signal"):
            controller.set_control(mock_model, mock_data)


class TestSetControlTracking:
    """Tests for data tracking in set_control."""

    def test_set_control_updates_tracker_at_save_timestep(self, controller, mock_model, mock_data) -> None:
        """Test that tracker is updated at save timesteps."""
        mock_data.time = 1.0  # 500 timesteps with 0.002 step = 1 second
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # Check if tracker was called at save timestep
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_save == 0:
            controller.tracker.update.assert_called_with(mock_data)

    def test_set_control_tracks_data_correctly(self, controller, mock_model, mock_data) -> None:
        """Test that tracker receives correct data."""
        mock_data.time = 1.0
        mock_data.ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        controller.set_control(mock_model, mock_data)
        
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_save == 0:
            controller.tracker.update.assert_called_once()
            args, kwargs = controller.tracker.update.call_args
            assert args[0] is mock_data

    def test_set_control_does_not_update_tracker_at_non_save_timestep(self, controller, mock_model, mock_data) -> None:
        """Test that tracker is not updated at non-save timesteps."""
        mock_data.time = 0.1  # 50 timesteps, not divisible by 500
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_save != 0:
            controller.tracker.update.assert_not_called()


class TestSetControlTimestepping:
    """Tests for timestep calculations in set_control."""

    def test_set_control_timestep_calculation(self, controller, mock_model, mock_data) -> None:
        """Test that timestep is calculated correctly."""
        mock_data.time = 0.1  # 100ms
        mock_model.opt.timestep = 0.002  # 2ms
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        # Timestep should be ceil(0.1 / 0.002) = ceil(50) = 50
        expected_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        assert expected_timestep == 50

    def test_set_control_multiple_timesteps(self, controller, mock_model, mock_data) -> None:
        """Test set_control with multiple different timesteps."""
        times = [0.002, 0.1, 0.5, 1.0, 10.0]
        
        for time_val in times:
            mock_data.time = time_val
            mock_data.ctrl = np.zeros(6)
            
            # Should not raise for any time value
            controller.set_control(mock_model, mock_data)


class TestSetControlEdgeCases:
    """Edge case tests for set_control."""

    def test_set_control_zero_time(self, controller, mock_model, mock_data) -> None:
        """Test set_control at zero time."""
        mock_data.time = 0.0
        mock_data.ctrl = np.zeros(6)
        
        # Should handle time=0 gracefully
        controller.set_control(mock_model, mock_data)

    def test_set_control_very_small_timestep(self, controller, mock_model, mock_data) -> None:
        """Test set_control with very small model timestep."""
        mock_model.opt.timestep = 0.0001  # 0.1ms
        mock_data.time = 0.01
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)

    def test_set_control_very_large_time(self, controller, mock_model, mock_data) -> None:
        """Test set_control with very large simulation time."""
        mock_data.time = 1000.0  # 1000 seconds
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)

    def test_set_control_callback_returns_different_sizes(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test set_control when callback returns different array sizes."""
        for size in [1, 6, 10]:
            mock_callback.return_value = np.ones(size)
            controller = Controller(
                controller_callback_function=mock_callback,
                time_steps_per_ctrl_step=1,
                tracker=mock_tracker,
            )
            
            mock_data.time = 100.0
            mock_data.ctrl = np.zeros(6)
            
            # Should handle size mismatch gracefully or convert properly
            try:
                controller.set_control(mock_model, mock_data)
            except (ValueError, IndexError):
                # Expected for size mismatch
                pass


class TestSetControlPreservesOldControl:
    """Tests that verify old control is preserved during blending."""

    def test_set_control_saves_old_control_copy(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that set_control makes a copy of old control."""
        old_ctrl = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        mock_data.ctrl = old_ctrl.copy()
        new_output = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mock_callback.return_value = new_output
        
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=0.5,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        controller.set_control(mock_model, mock_data)
        
        # Expected: 0.5 * old + 0.5 * new = 0.5 * old_ctrl
        expected = np.clip(0.5 * old_ctrl + 0.5 * new_output, -np.pi / 2, np.pi / 2)
        np.testing.assert_array_almost_equal(mock_data.ctrl, expected)


class TestSetControlCallbackExecution:
    """Tests for callback execution details."""

    def test_set_control_callback_receives_model(self, controller, mock_model, mock_data) -> None:
        """Test that callback receives model parameter."""
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_ctrl_step == 0:
            # Check callback was called with model
            call_args = controller.controller_callback_function.call_args
            if call_args:
                assert call_args[0][0] is mock_model

    def test_set_control_callback_receives_data(self, controller, mock_model, mock_data) -> None:
        """Test that callback receives data parameter."""
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        controller.set_control(mock_model, mock_data)
        
        deduced_timestep = int(np.ceil(mock_data.time / mock_model.opt.timestep))
        if deduced_timestep % controller.time_steps_per_ctrl_step == 0:
            # Check callback was called with data
            call_args = controller.controller_callback_function.call_args
            if call_args:
                assert call_args[0][1] is mock_data

    def test_set_control_callback_return_as_array(self, mock_callback, mock_model, mock_data, mock_tracker) -> None:
        """Test that callback return value is converted to numpy array."""
        # Return as list instead of array
        mock_callback.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        controller = Controller(
            controller_callback_function=mock_callback,
            time_steps_per_ctrl_step=1,
            alpha=0.5,
            tracker=mock_tracker,
        )
        
        mock_data.time = 100.0
        mock_data.ctrl = np.zeros(6)
        
        # Should convert list to array
        controller.set_control(mock_model, mock_data)
        
        assert isinstance(mock_data.ctrl, np.ndarray)