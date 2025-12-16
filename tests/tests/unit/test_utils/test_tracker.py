import pytest
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import mujoco as mj
from ariel.utils import tracker
# --- Fixtures ---

@pytest.fixture
def mock_ariel_log():
    """Mock the logging module to prevent side effects."""
    # Use patch.object to avoid resolving the module name via importlib
    # which can fail if the module isn't strictly in sys.path but imported via pytest
    with patch.object(tracker, 'log') as mock_log:
        yield mock_log

@pytest.fixture
def mock_mujoco_objects():
    """
    Creates mocks for 'world' (MjSpec) and 'data' (MjData).
    Simulates finding geoms and binding them.
    """
    mock_world = MagicMock()
    mock_data = MagicMock()

    # Create dummy geoms that 'find_all' would return
    geom1 = MagicMock()
    geom1.name = "robot-core-1"
    geom2 = MagicMock()
    geom2.name = "robot-leg-1" # Should be filtered out by default "core" filter
    geom3 = MagicMock()
    geom3.name = "robot-core-2"
    
    # Mock find_all to return these
    mock_world.worldbody.find_all.return_value = [geom1, geom2, geom3]

    # Mock data.bind to return a "bound object" with attributes
    def side_effect_bind(geom):
        bound = MagicMock()
        bound.name = geom.name
        # Give it some fake data attributes
        bound.xpos = np.array([1.0, 2.0, 3.0])
        bound.xquat = np.array([0.0, 0.0, 0.0, 1.0])
        return bound
    
    mock_data.bind.side_effect = side_effect_bind

    return mock_world, mock_data

# --- Tests ---

def test_tracker_init_defaults(mock_ariel_log):
    """Test initialization with default parameters."""
    # Updated to use module.Class syntax
    test_tracker = tracker.Tracker()
    
    assert test_tracker.mujoco_obj_to_find == mj.mjtObj.mjOBJ_GEOM
    assert test_tracker.name_to_bind == "core"
    assert test_tracker.observable_attributes == ["xpos"]
    assert test_tracker.history == {}
    
    # Verify it logged the default behavior info
    mock_ariel_log.info.assert_called()

def test_tracker_init_custom():
    """Test initialization with custom parameters."""
    test_tracker = tracker.Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_BODY,
        name_to_bind="foot",
        observable_attributes=["xpos", "xquat"]
    )
    
    assert test_tracker.mujoco_obj_to_find == mj.mjtObj.mjOBJ_BODY
    assert test_tracker.name_to_bind == "foot"
    assert test_tracker.observable_attributes == ["xpos", "xquat"]

def test_tracker_setup(mock_mujoco_objects):
    """Test the setup method finds and binds correct objects."""
    world, data = mock_mujoco_objects
    test_tracker = tracker.Tracker(name_to_bind="core")
    
    test_tracker.setup(world, data)
    
    # Check that find_all was called with GEOM (default)
    world.worldbody.find_all.assert_called_with(mj.mjtObj.mjOBJ_GEOM)
    
    # We expected 3 geoms in find_all, but only 2 have "core" in name
    # So we expect data.bind to be called for those 2
    assert len(test_tracker.to_track) == 2
    assert test_tracker.to_track[0].name == "robot-core-1"
    assert test_tracker.to_track[1].name == "robot-core-2"
    
    # Verify history structure is initialized
    # Structure: history[attr][obj_idx] = []
    assert "xpos" in test_tracker.history
    assert isinstance(test_tracker.history["xpos"][0], list)
    assert isinstance(test_tracker.history["xpos"][1], list)
    assert len(test_tracker.history["xpos"][0]) == 0

def test_tracker_update(mock_mujoco_objects):
    """Test that update appends data to history."""
    world, data = mock_mujoco_objects
    test_tracker = tracker.Tracker(name_to_bind="core", observable_attributes=["xpos"])
    test_tracker.setup(world, data)
    
    # Perform one update
    test_tracker.update(data)
    
    # Check if data was added
    assert len(test_tracker.history["xpos"][0]) == 1
    # Check value (mocked as [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(test_tracker.history["xpos"][0][0], np.array([1.0, 2.0, 3.0]))
    
    # Perform another update
    test_tracker.update(data)
    assert len(test_tracker.history["xpos"][0]) == 2

def test_tracker_reset(mock_mujoco_objects):
    """Test that reset clears history lists but keeps keys."""
    world, data = mock_mujoco_objects
    test_tracker = tracker.Tracker(name_to_bind="core")
    test_tracker.setup(world, data)
    
    # Fill with some data
    test_tracker.update(data)
    test_tracker.update(data)
    assert len(test_tracker.history["xpos"][0]) == 2
    
    # Reset
    test_tracker.reset()
    
    # Verify it is empty now
    assert len(test_tracker.history["xpos"][0]) == 0
    # Verify structure still exists
    assert 0 in test_tracker.history["xpos"]

def test_tracker_multiple_attributes(mock_mujoco_objects):
    """Test tracking multiple attributes (xpos and xquat)."""
    world, data = mock_mujoco_objects
    test_tracker = tracker.Tracker(
        name_to_bind="core", 
        observable_attributes=["xpos", "xquat"]
    )
    test_tracker.setup(world, data)
    
    test_tracker.update(data)
    
    assert "xpos" in test_tracker.history
    assert "xquat" in test_tracker.history
    
    # Check values
    assert len(test_tracker.history["xpos"][0]) == 1
    assert len(test_tracker.history["xquat"][0]) == 1
    # Check mock value for quat
    np.testing.assert_array_equal(test_tracker.history["xquat"][0][0], np.array([0., 0., 0., 1.]))