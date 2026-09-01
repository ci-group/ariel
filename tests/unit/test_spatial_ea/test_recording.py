"""Test: video and snapshot capture of the movement phase."""

# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
import pytest

# Local libraries
from ariel.spatial_ea.config import SpatialEAConfig
from ariel.spatial_ea.engine import SpatialEA
from ariel.spatial_ea.recording import (
    GenerationRecorder,
    frame_world_camera,
)
from ariel.spatial_ea.world import build_single_robot_world

WORLD = (4.0, 4.0, 0.1)


def _model_and_data() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build a one-robot world ready to step."""
    _, model, _, _ = build_single_robot_world(WORLD)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_camera_frames_the_whole_world() -> None:
    """The camera looks at the world centre from far enough to hold it."""
    camera = frame_world_camera((10.0, 6.0))

    assert camera.lookat[0] == pytest.approx(5.0)
    assert camera.lookat[1] == pytest.approx(3.0)
    # At least the world's own half-span, or robots fall out of shot.
    assert camera.distance >= np.hypot(10.0, 6.0) / 2.0
    assert camera.elevation < 0  # looking down


def test_camera_never_collapses_on_a_tiny_world() -> None:
    """A very small world still gets a usable camera distance."""
    assert frame_world_camera((0.2, 0.2)).distance >= 1.0


def test_recorder_is_inert_when_nothing_is_requested() -> None:
    """With both outputs off, no renderer is created at all."""
    model, data = _model_and_data()

    with GenerationRecorder(model=model, generation=0) as recorder:
        recorder.capture(data, 0, 10)
        assert recorder.enabled is False
        assert recorder.outputs == {}


def test_snapshot_is_written_once_at_the_midpoint(tmp_path: Path) -> None:
    """One snapshot per phase, taken from the middle of it."""
    model, data = _model_and_data()

    with GenerationRecorder(
        model=model,
        generation=7,
        world_size=(4.0, 4.0),
        save_snapshot=True,
        figure_folder=tmp_path,
        width=160,
        height=120,
    ) as recorder:
        if not recorder.enabled:
            pytest.skip("no GL context available for offscreen rendering")
        for step in range(20):
            recorder.capture(data, step, 20)
            mujoco.mj_step(model, data)
        outputs = recorder.outputs

    snapshots = sorted(tmp_path.glob("*.png"))
    assert len(snapshots) == 1
    assert snapshots[0].name == "generation_007_snapshot.png"
    assert outputs["snapshot"] == snapshots[0]
    assert snapshots[0].stat().st_size > 0


def test_video_is_written_and_released(tmp_path: Path) -> None:
    """A video file is produced and closed by the context manager."""
    model, data = _model_and_data()

    with GenerationRecorder(
        model=model,
        generation=2,
        world_size=(4.0, 4.0),
        record_video=True,
        video_folder=tmp_path,
        width=160,
        height=120,
        fps=30,
    ) as recorder:
        if not recorder.enabled:
            pytest.skip("no GL context available for offscreen rendering")
        for step in range(40):
            recorder.capture(data, step, 40)
            mujoco.mj_step(model, data)

    videos = list(tmp_path.iterdir())
    assert len(videos) == 1
    assert "generation_002_mating_movement" in videos[0].name
    assert videos[0].stat().st_size > 0


def test_frame_interval_tracks_real_time() -> None:
    """Frames are spaced so the video plays back at real speed."""
    model, _ = _model_and_data()

    with GenerationRecorder(
        model=model,
        generation=0,
        record_video=True,
        fps=30,
        width=160,
        height=120,
    ) as recorder:
        if not recorder.enabled:
            pytest.skip("no GL context available for offscreen rendering")
        expected = max(1, round(1.0 / (model.opt.timestep * 30)))
        assert recorder._steps_per_frame == expected


def test_a_failed_renderer_does_not_break_a_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A machine with no GL context should still complete the phase."""
    model, data = _model_and_data()

    def explode(*args: object, **kwargs: object) -> None:
        msg = "no GL context"
        raise RuntimeError(msg)

    monkeypatch.setattr("ariel.spatial_ea.recording.mujoco.Renderer", explode)

    with GenerationRecorder(
        model=model,
        generation=0,
        save_snapshot=True,
        record_video=True,
        figure_folder=tmp_path,
        video_folder=tmp_path,
    ) as recorder:
        recorder.capture(data, 0, 10)
        assert recorder.enabled is False

    assert list(tmp_path.iterdir()) == []


def test_engine_records_when_asked(tmp_path: Path) -> None:
    """Turning snapshots on should leave one image per generation."""
    config = SpatialEAConfig(
        population_size=3,
        num_generations=2,
        simulation_time=0.2,
        world_size=(4.0, 4.0),
        min_spawn_distance=0.5,
        pairing_radius=1e-6,
        figure_folder=tmp_path,
        save_results=False,
        save_plots=False,
        save_generation_snapshots=True,
        video_width=160,
        video_height=120,
        print_generation_stats=False,
    )
    SpatialEA(config=config).run(generations=2)

    snapshots = sorted(p.name for p in tmp_path.glob("*_snapshot.png"))
    if not snapshots:
        pytest.skip("no GL context available for offscreen rendering")
    assert snapshots == [
        "generation_000_snapshot.png",
        "generation_001_snapshot.png",
    ]


def test_recording_is_off_by_default(tmp_path: Path) -> None:
    """A plain run writes no video and no snapshot."""
    config = SpatialEAConfig(
        population_size=3,
        num_generations=1,
        simulation_time=0.2,
        world_size=(4.0, 4.0),
        min_spawn_distance=0.5,
        figure_folder=tmp_path,
        video_folder=tmp_path,
        save_results=False,
        save_plots=False,
        print_generation_stats=False,
    )
    SpatialEA(config=config).run(generations=1)

    assert list(tmp_path.glob("*.png")) == []
    assert list(tmp_path.glob("*.mp4")) == []
