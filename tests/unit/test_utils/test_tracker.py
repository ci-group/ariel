"""Test: Tracker class initialization, setup, update, and reset."""

import mujoco

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.spider import spider
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
from ariel.utils.tracker import Tracker


def _build_spider_in_world():
    """Spawn a spider in a SimpleFlatWorld and return (spec, model, data)."""
    world = SimpleFlatWorld()
    core = spider()
    world.spawn(core.spec, position=(0, 0, 0.3), rotation=(0, 0, 0))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return world.spec, model, data


def test_tracker_default_initialization() -> None:
    """Tracker can be instantiated without any arguments."""
    tracker = Tracker(quiet=True)
    assert tracker.name_to_bind == "core"
    assert tracker.observable_attributes == ["xpos"]
    assert tracker.history == {}


def test_tracker_custom_initialization() -> None:
    """Tracker stores custom parameters."""
    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="floor",
        observable_attributes=["xpos", "xquat"],
        quiet=True,
    )
    assert tracker.name_to_bind == "floor"
    assert tracker.observable_attributes == ["xpos", "xquat"]


def test_tracker_setup_initializes_history() -> None:
    """setup() creates the history dict for tracked attributes."""
    spec, model, data = _build_spider_in_world()
    tracker = Tracker(quiet=True)
    tracker.setup(spec, data)
    assert "xpos" in tracker.history


def test_tracker_setup_finds_core_geom() -> None:
    """setup() binds at least one object when a 'core' geom exists."""
    spec, model, data = _build_spider_in_world()
    tracker = Tracker(quiet=True)
    tracker.setup(spec, data)
    assert len(tracker.to_track) >= 1


def test_tracker_update_appends_to_history() -> None:
    """update() appends data to history on each call."""
    spec, model, data = _build_spider_in_world()
    tracker = Tracker(quiet=True)
    tracker.setup(spec, data)

    n_updates = 3
    for _ in range(n_updates):
        mujoco.mj_step(model, data)
        tracker.update(data)

    for attr in tracker.observable_attributes:
        total = sum(len(v) for v in tracker.history[attr].values())
        assert total == n_updates * len(tracker.to_track)


def test_tracker_reset_clears_history() -> None:
    """reset() empties all history lists."""
    spec, model, data = _build_spider_in_world()
    tracker = Tracker(quiet=True)
    tracker.setup(spec, data)
    mujoco.mj_step(model, data)
    tracker.update(data)

    tracker.reset()
    for attr in tracker.observable_attributes:
        for vals in tracker.history[attr].values():
            assert vals == []


def test_tracker_floor_geom() -> None:
    """Tracker can track a floor geom by partial name match."""
    world = SimpleFlatWorld()
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="floor",
        observable_attributes=["xpos"],
        quiet=True,
    )
    tracker.setup(world.spec, data)
    # floor geom should exist in a flat world
    assert "xpos" in tracker.history
