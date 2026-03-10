"""TODO(jmdm): description of script."""

# Standard library
import datetime
from pathlib import Path

# Third-party libraries
import cv2
import numpy as np
from numpy import typing as npt


class VideoRecorder:
    """Simple video recorder for ariel."""

    # Default to broadly supported MPEG-4 Part 2 to avoid missing H.264 encoder
    # errors on OpenCV/FFmpeg builds without libx264.
    _video_encoding: str = "mp4v"
    _add_timestamp_to_file_name: bool = True
    _file_extension: str = ".mp4"
    _fallback_encoders: list[tuple[str, str]] = [
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    ]

    def __init__(
        self,
        file_name: str = "video",
        output_folder: str | Path | None = None,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        """Create a video recording.

        Parameters
        ----------
        file_name
            Name of the video file.
        output_folder
            Folder where the video will be saved.
            If None, saves to current directory.
        width
            Width of the video frames.
        height
            Height of the video frames.
        fps
            Frames per second for the video.
        """
        # Save local variables
        self.width = width
        self.height = height
        self.fps = fps

        # Set output folder
        if output_folder is None:
            output_folder = Path.cwd()
        elif isinstance(output_folder, str):
            # Convert string to Path
            output_folder = Path(output_folder)

        # Ensure output folder exists
        Path(output_folder).mkdir(exist_ok=True, parents=True)

        # Generate video name
        if self._add_timestamp_to_file_name:
            timestamp = datetime.datetime.now(tz=datetime.UTC).strftime(
                "%Y%m%d_%H%M%S",
            )
            file_name += f"_{timestamp}"

        # Create recorder object with fallback codecs.
        # Some environments (e.g., OpenCV builds without H.264) cannot open `avc1`.
        output_file = output_folder / f"{file_name}{self._file_extension}"

        fourcc = cv2.VideoWriter_fourcc(*self._video_encoding)
        video_writer = cv2.VideoWriter(
            output_file,
            fourcc,
            self.fps,
            (self.width, self.height),
        )

        if not video_writer.isOpened():
            for codec, ext in self._fallback_encoders:
                output_file = output_folder / f"{file_name}{ext}"
                fourcc = cv2.VideoWriter_fourcc(*codec)
                video_writer = cv2.VideoWriter(
                    output_file,
                    fourcc,
                    self.fps,
                    (self.width, self.height),
                )
                if video_writer.isOpened():
                    break

        if not video_writer.isOpened():
            msg = (
                "Could not initialize video writer with codecs "
                f"'{self._video_encoding}' + fallbacks {self._fallback_encoders}. "
                "Install FFmpeg/GStreamer codecs or use an OpenCV build with video encoders."
            )
            raise RuntimeError(msg)

        # Class attributes
        self.frame_count = 0
        self.video_writer = video_writer

    def write(self, frame: npt.ArrayLike) -> None:
        """Write MuJoCo frame to video.

        Parameters
        ----------
        frame
            Frame to write to the video.
        """
        # Convert PIL Image to numpy array (OpenCV uses BGR format)
        opencv_image = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Save frame
        self.video_writer.write(opencv_image)

        # Increment frame counter
        self.frame_count += 1

    def release(self) -> None:
        """Close video writer and save video locally."""
        self.video_writer.release()
