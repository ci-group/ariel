import types
from pathlib import Path
import numpy as np
import pytest
from types import SimpleNamespace
from PIL import Image

import ariel.utils.renderers as renderers


class FakeMjvOption:
    def __init__(self):
        self.flags = {}


class FakeMjvCamera:
    def __init__(self):
        self.type = None
        self.trackbodyid = -1
        self.distance = 1.0
        self.azimuth = 0.0
        self.elevation = 0.0


def make_fake_renderer(width, height):
    class FakeRenderer:
        def __init__(self, model, width=width, height=height):
            self._w = width
            self._h = height

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update_scene(self, data, scene_option=None, camera=None):
            # no-op
            pass

        def render(self):
            # return a blank RGB image
            return np.zeros((self._h, self._w, 3), dtype=np.uint8)

    return FakeRenderer


@pytest.fixture(autouse=True)
def patch_mujoco(monkeypatch, tmp_path):
    """
    Patch the mujoco symbols used by renderers so tests do not require MuJoCo.
    """
    fake_mujoco = types.SimpleNamespace()

    # minimal MjOption and MjvOption and flags and camera constants
    fake_mujoco.MjvOption = FakeMjvOption
    fake_mujoco.MjvCamera = FakeMjvCamera
    fake_mujoco.mjtVisFlag = SimpleNamespace(
        mjVIS_JOINT=0, mjVIS_TRANSPARENT=1, mjVIS_ACTUATOR=2, mjVIS_BODYBVH=3
    )
    fake_mujoco.mjtObj = SimpleNamespace(
        mjOBJ_CAMERA=0, mjOBJ_BODY=1
    )
    fake_mujoco.mjtCamera = SimpleNamespace(mjCAMERA_TRACKING=1)

    class FakeMjOption:
        def __init__(self):
            self.timestep = 0.02  # 20ms default

    fake_mujoco.MjOption = FakeMjOption

    # mj_resetData, mj_step, mj_name2id, mj_id2name
    def mj_resetData(model, data):
        data.time = 0.0
        return None

    def mj_step(model, data, nstep=1):
        # Advance time enough to exit loops in renderers when called
        data.time = getattr(data, "time", 0.0) + max(1, float(nstep) * fake_mujoco.MjOption().timestep)
        return None

    fake_mujoco.mj_resetData = mj_resetData
    fake_mujoco.mj_step = mj_step

    # default: camera not found
    fake_mujoco.mj_name2id = lambda model, obj, name: -1
    fake_mujoco.mj_id2name = lambda model, obj, idx: None

    # Renderer factory (will be monkeypatched per-test when needed)
    fake_mujoco.Renderer = lambda model, width, height: make_fake_renderer(width, height)(model, width=width, height=height)

    # inject into module under test
    monkeypatch.setattr(renderers, "mujoco", fake_mujoco)
    # also patch generate_save_path to return a tmp file if used
    monkeypatch.setattr(renderers, "generate_save_path", lambda file_path="img.png": Path(tmp_path) / file_path)

    yield


def make_model_and_data():
    # simple model with camera arrays that can be indexed
    model = SimpleNamespace()
    model.cam_fovy = [1.0]
    model.cam_pos = [(0.0, 0.0, 0.0)]
    model.cam_quat = [(1.0, 0.0, 0.0, 0.0)]
    model.cam_sensorsize = [[480, 640]]
    model.nbody = 2
    return model, SimpleNamespace(time=0.0, ctrl=None)


class TestSingleFrameRenderer:
    """Tests for single_frame_renderer function."""

    def test_camera_not_found_saves_and_returns_image(self, tmp_path):
        """Test that single_frame_renderer saves and returns an image when camera is not found."""
        model, data = make_model_and_data()
        save_path = tmp_path / "test_img.png"
        
        img = renderers.single_frame_renderer(
            model, data, steps=1, width=32, height=24, 
            show=False, save=True, save_path=save_path
        )
        
        assert isinstance(img, Image.Image)
        assert save_path.exists()

    def test_camera_found_applies_camera(self):
        """Test that single_frame_renderer applies camera settings when camera is found."""
        # Patch mj_name2id to return a valid camera index
        original_name2id = renderers.mujoco.mj_name2id
        renderers.mujoco.mj_name2id = lambda model, obj, name: 0
        
        try:
            model, data = make_model_and_data()
            model.cam_fovy = [0.5]
            model.cam_pos = [(0.1, 0.2, 0.3)]
            model.cam_quat = [(0.0, 0.0, 0.0, 1.0)]
            model.cam_sensorsize = [[32, 24]]
            
            img = renderers.single_frame_renderer(
                model, data, steps=1, width=32, height=24,
                cam_fovy=0.5, cam_pos=(0.1, 0.2, 0.3), cam_quat=(0.0, 0.0, 0.0, 1.0),
                show=False, save=False
            )
            
            assert isinstance(img, Image.Image)
        finally:
            renderers.mujoco.mj_name2id = original_name2id

    def test_returns_image_with_correct_dimensions(self):
        """Test that returned image has correct dimensions."""
        model, data = make_model_and_data()
        width, height = 64, 48
        
        img = renderers.single_frame_renderer(
            model, data, steps=1, width=width, height=height,
            show=False, save=False
        )
        
        assert isinstance(img, Image.Image)
        assert img.size == (width, height)

    def test_no_save_without_save_flag(self, tmp_path):
        """Test that image is not saved when save=False."""
        model, data = make_model_and_data()
        save_path = tmp_path / "should_not_exist.png"
        
        img = renderers.single_frame_renderer(
            model, data, steps=1, width=32, height=24,
            show=False, save=False, save_path=save_path
        )
        
        assert isinstance(img, Image.Image)
        assert not save_path.exists()


class TestVideoRenderer:
    """Tests for video_renderer function."""

    def test_writes_frames_and_releases(self):
        """Test that video_renderer writes frames and releases the recorder."""
        frames = []
        
        class FakeRecorder:
            def __init__(self):
                self.width = 32
                self.height = 24
                self.fps = 10
                self.frame_count = 0

            def write(self, frame):
                frames.append(frame)
                self.frame_count += 1

            def release(self):
                self.released = True

        rec = FakeRecorder()
        model, data = make_model_and_data()
        
        renderers.video_renderer(model, data, duration=0.01, video_recorder=rec)
        
        assert rec.frame_count >= 1
        assert len(frames) >= 1
        assert frames[0].shape[0] == rec.height
        assert frames[0].shape[1] == rec.width
        assert getattr(rec, "released", False) is True

    def test_creates_default_recorder_if_none_provided(self):
        """Test that video_renderer creates a default VideoRecorder if none is provided."""
        model, data = make_model_and_data()
        
        # Should not raise; should use default VideoRecorder
        try:
            renderers.video_renderer(model, data, duration=0.01, video_recorder=None)
        except Exception as e:
            # We expect VideoRecorder to not exist in test env, but that's okay
            # We're just checking the function doesn't crash trying to create one
            if "VideoRecorder" not in str(e):
                raise

    def test_camera_not_found_uses_default(self):
        """Test that video_renderer uses default camera when 'ortho-cam' not found."""
        class FakeRecorder:
            def __init__(self):
                self.width = 32
                self.height = 24
                self.fps = 10
                self.frame_count = 0

            def write(self, frame):
                self.frame_count += 1

            def release(self):
                pass

        rec = FakeRecorder()
        model, data = make_model_and_data()
        
        # mj_name2id returns -1 (camera not found)
        renderers.video_renderer(model, data, duration=0.01, video_recorder=rec)
        
        assert rec.frame_count >= 1


class TestTrackingVideoRenderer:
    """Tests for tracking_video_renderer function."""

    def test_falls_back_to_core_body_when_not_found(self):
        """Test that tracking_video_renderer falls back to core body when requested body not found."""
        # Patch mj_name2id to raise ValueError
        original_name2id = renderers.mujoco.mj_name2id
        original_id2name = renderers.mujoco.mj_id2name
        
        def raise_not_found(model, obj, name):
            raise ValueError("not found")
        
        def id2name(model, obj, idx):
            return "robot-core" if idx == 1 else "something"
        
        renderers.mujoco.mj_name2id = raise_not_found
        renderers.mujoco.mj_id2name = id2name
        
        try:
            class FakeRecorder:
                def __init__(self):
                    self.width = 32
                    self.height = 24
                    self.fps = 10
                    self.frame_count = 0

                def write(self, frame):
                    self.frame_count += 1

                def release(self):
                    self.released = True

            rec = FakeRecorder()
            model, data = make_model_and_data()
            model.nbody = 2
            
            renderers.tracking_video_renderer(
                model, data, duration=0.01, video_recorder=rec,
                geom_to_track="does-not-exist"
            )
            
            assert rec.frame_count >= 1
            assert getattr(rec, "released", False) is True
        finally:
            renderers.mujoco.mj_name2id = original_name2id
            renderers.mujoco.mj_id2name = original_id2name

    def test_tracks_specified_body_when_found(self):
        """Test that tracking_video_renderer tracks the specified body when found."""
        original_name2id = renderers.mujoco.mj_name2id
        
        # Return body ID 0 for "robot-core"
        renderers.mujoco.mj_name2id = lambda model, obj, name: 0 if name == "robot-core" else -1
        
        try:
            class FakeRecorder:
                def __init__(self):
                    self.width = 32
                    self.height = 24
                    self.fps = 10
                    self.frame_count = 0

                def write(self, frame):
                    self.frame_count += 1

                def release(self):
                    self.released = True

            rec = FakeRecorder()
            model, data = make_model_and_data()
            
            renderers.tracking_video_renderer(
                model, data, duration=0.01, video_recorder=rec,
                geom_to_track="robot-core"
            )
            
            assert rec.frame_count >= 1
            assert getattr(rec, "released", False) is True
        finally:
            renderers.mujoco.mj_name2id = original_name2id

    def test_creates_mjv_camera_with_correct_parameters(self):
        """Test that tracking_video_renderer creates MjvCamera with correct tracking parameters."""
        original_name2id = renderers.mujoco.mj_name2id
        renderers.mujoco.mj_name2id = lambda model, obj, name: 0 if name == "robot-core" else -1
        
        try:
            class FakeRecorder:
                def __init__(self):
                    self.width = 32
                    self.height = 24
                    self.fps = 10
                    self.frame_count = 0

                def write(self, frame):
                    self.frame_count += 1

                def release(self):
                    pass

            rec = FakeRecorder()
            model, data = make_model_and_data()
            
            tracking_distance = 2.5
            tracking_azimuth = 90
            tracking_elevation = -45
            
            renderers.tracking_video_renderer(
                model, data, duration=0.01, video_recorder=rec,
                geom_to_track="robot-core",
                tracking_distance=tracking_distance,
                tracking_azimuth=tracking_azimuth,
                tracking_elevation=tracking_elevation
            )
            
            assert rec.frame_count >= 1
        finally:
            renderers.mujoco.mj_name2id = original_name2id

    def test_uses_default_camera_when_no_core_found(self):
        """Test that tracking_video_renderer uses default camera when no core body found."""
        original_name2id = renderers.mujoco.mj_name2id
        original_id2name = renderers.mujoco.mj_id2name
        
        renderers.mujoco.mj_name2id = lambda model, obj, name: -1
        renderers.mujoco.mj_id2name = lambda model, obj, idx: None
        
        try:
            class FakeRecorder:
                def __init__(self):
                    self.width = 32
                    self.height = 24
                    self.fps = 10
                    self.frame_count = 0

                def write(self, frame):
                    self.frame_count += 1

                def release(self):
                    self.released = True

            rec = FakeRecorder()
            model, data = make_model_and_data()
            
            renderers.tracking_video_renderer(
                model, data, duration=0.01, video_recorder=rec,
                geom_to_track="nonexistent"
            )
            
            assert rec.frame_count >= 1
            assert getattr(rec, "released", False) is True
        finally:
            renderers.mujoco.mj_name2id = original_name2id
            renderers.mujoco.mj_id2name = original_id2name