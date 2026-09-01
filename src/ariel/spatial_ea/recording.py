"""Video and snapshot capture for the mating movement phase.

Watching a generation move is the fastest way to tell whether controllers are
walking or thrashing, which the trajectory plots only imply.

ARIEL's ``video_renderer`` and ``tracking_video_renderer`` drive their own
simulation loops, so they cannot wrap the movement phase — that loop already
exists and owns control and boundary wrapping. This module therefore reuses
``VideoRecorder`` as the frame sink and drives ``mujoco.Renderer`` from inside
the existing loop.

Notes
-----
    * Offscreen rendering needs a GL context. When one cannot be created the
      recorder disables itself and logs a warning rather than aborting a run
      that is otherwise fine.
    * Frame resolution is capped by the model's ``offwidth``/``offheight``,
      which ``MujocoConfig`` sets to 1280x960.

"""

# Standard library
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

# Third-party libraries
import mujoco
import numpy as np
from PIL import Image

# Local libraries
from ariel import log
from ariel.utils.video_recorder import VideoRecorder

# Evaluate type annotations in a deferred manner (ruff: UP037)
if TYPE_CHECKING:
    from types import TracebackType

# Global constants
MAX_RENDER_WIDTH = 1280
MAX_RENDER_HEIGHT = 960
SNAPSHOT_POSITION = 0.5
DEFAULT_AZIMUTH = 90.0
DEFAULT_ELEVATION = -72.0
# The free camera must sit far enough back to hold the whole world; a little
# over the diagonal frames it with a small margin at this elevation.
DISTANCE_PER_WORLD_SPAN = 0.95


def frame_world_camera(
    world_size: tuple[float, float],
    azimuth: float = DEFAULT_AZIMUTH,
    elevation: float = DEFAULT_ELEVATION,
) -> mujoco.MjvCamera:
    """Build a free camera that holds the whole world in frame.

    The default free camera frames the origin at a fixed distance, which leaves
    a population spread across the world half out of shot. Recording exists to
    show how the population moves, so the camera is pulled back to the world's
    own scale.

    Parameters
    ----------
    world_size
        World dimensions ``(width, height)``.
    azimuth
        Horizontal camera angle in degrees.
    elevation
        Vertical camera angle in degrees; negative looks down.

    Returns
    -------
        A free camera centred on the world.
    """
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [world_size[0] / 2.0, world_size[1] / 2.0, 0.0]
    span = float(np.hypot(world_size[0], world_size[1]))
    camera.distance = max(1.0, span * DISTANCE_PER_WORLD_SPAN)
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def _scene_option() -> mujoco.MjvOption:
    """Build the scene options used for every captured frame.

    Joints are drawn so that a still frame shows how the body is articulated,
    not just its silhouette.

    Returns
    -------
        Scene options with joint visualisation enabled.
    """
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    return option


@dataclass
class GenerationRecorder:
    """Captures a video and/or a still frame of one movement phase.

    Use as a context manager so the renderer and the video file are released
    even if the phase raises.

    Parameters
    ----------
    model
        The compiled model being simulated.
    generation
        Generation number, used in the output filenames.
    world_size
        World dimensions ``(width, height)``, used to frame the camera.
    record_video
        Whether to write a video of the phase.
    save_snapshot
        Whether to write a single still frame from the middle of the phase.
    video_folder
        Directory for the video.
    figure_folder
        Directory for the snapshot.
    width
        Frame width, clamped to the model's offscreen buffer.
    height
        Frame height, clamped to the model's offscreen buffer.
    fps
        Frames per second of the written video.
    """

    model: mujoco.MjModel
    generation: int
    world_size: tuple[float, float] = (10.0, 10.0)
    record_video: bool = False
    save_snapshot: bool = False
    video_folder: Path = field(
        default_factory=lambda: Path.cwd() / "__videos__",
    )
    figure_folder: Path = field(
        default_factory=lambda: Path.cwd() / "__figures__",
    )
    width: int = 640
    height: int = 480
    fps: int = 30

    _camera: mujoco.MjvCamera | None = field(default=None, init=False)
    _renderer: mujoco.Renderer | None = field(default=None, init=False)
    _recorder: VideoRecorder | None = field(default=None, init=False)
    _steps_per_frame: int = field(default=1, init=False)
    _snapshot_written: bool = field(default=False, init=False)
    _video_path: Path | None = field(default=None, init=False)
    _snapshot_path: Path | None = field(default=None, init=False)

    @property
    def enabled(self) -> bool:
        """Whether anything will actually be captured.

        Returns
        -------
            ``True`` when a renderer was created and at least one output was
            requested.
        """
        return self._renderer is not None

    @property
    def outputs(self) -> dict[str, Path]:
        """Files this recorder wrote.

        Returns
        -------
            Paths keyed ``video`` and ``snapshot``, omitting whichever was not
            written.
        """
        written: dict[str, Path] = {}
        if self._video_path is not None:
            written["video"] = self._video_path
        if self._snapshot_path is not None:
            written["snapshot"] = self._snapshot_path
        return written

    def __enter__(self) -> Self:
        """Create the renderer and, if requested, open the video file.

        Returns
        -------
            This recorder, enabled or silently disabled.
        """
        if not (self.record_video or self.save_snapshot):
            return self

        width = min(self.width, MAX_RENDER_WIDTH)
        height = min(self.height, MAX_RENDER_HEIGHT)

        try:
            self._renderer = mujoco.Renderer(
                self.model,
                width=width,
                height=height,
            )
        except Exception:  # noqa: BLE001 - any GL failure disables capture
            log.warning(
                "Could not create a MuJoCo renderer; "
                "continuing without video or snapshots",
            )
            self._renderer = None
            return self

        self._camera = frame_world_camera(self.world_size)

        if self.record_video:
            self._recorder = VideoRecorder(
                file_name=f"generation_{self.generation:03d}_mating_movement",
                output_folder=self.video_folder,
                width=width,
                height=height,
                fps=self.fps,
            )
            # One frame per 1 / (timestep * fps) steps of simulated time, so
            # the video plays back at real speed.
            timestep = float(self.model.opt.timestep)
            self._steps_per_frame = max(1, round(1.0 / (timestep * self.fps)))

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the video file and the renderer.

        Parameters
        ----------
        exc_type
            Exception type, if the phase raised.
        exc
            The exception, if the phase raised.
        traceback
            The traceback, if the phase raised.
        """
        del exc_type, exc, traceback

        if self._recorder is not None:
            self._recorder.release()
            self._video_path = Path(self.video_folder)
            self._recorder = None

        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def capture(
        self,
        data: mujoco.MjData,
        step: int,
        total_steps: int,
    ) -> None:
        """Capture whatever this step calls for.

        Parameters
        ----------
        data
            Current simulation state.
        step
            Index of the step just completed.
        total_steps
            Total steps in the phase, used to place the snapshot.
        """
        if self._renderer is None:
            return

        snapshot_step = int(total_steps * SNAPSHOT_POSITION)
        wants_snapshot = (
            self.save_snapshot
            and not self._snapshot_written
            and step >= snapshot_step
        )
        wants_frame = (
            self._recorder is not None and step % self._steps_per_frame == 0
        )
        if not (wants_snapshot or wants_frame):
            return

        self._renderer.update_scene(
            data,
            camera=self._camera,
            scene_option=_scene_option(),
        )
        frame = self._renderer.render()

        if wants_frame and self._recorder is not None:
            self._recorder.write(frame=frame)

        if wants_snapshot:
            folder = Path(self.figure_folder)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"generation_{self.generation:03d}_snapshot.png"
            Image.fromarray(frame).save(path, format="png")
            self._snapshot_path = path
            self._snapshot_written = True

            msg = f"Saved generation snapshot: {path}"
            log.info(msg)
