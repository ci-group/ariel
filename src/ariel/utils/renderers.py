"""TODO(jmdm): description of script.

Todo:
----
    [ ] ".rotate" as superclass method?
    [ ] Better documentation
"""

# Standard library
import math
from pathlib import Path

# Third-party libraries
import mujoco
from PIL import Image

# Local libraries
from ariel import log
from ariel.utils.file_ops import generate_save_path
from ariel.utils.video_recorder import VideoRecorder


def single_frame_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    steps: int = 1,
    width: int = 480,
    height: int = 640,
    cam_fovy: float | None = None,
    cam_pos: tuple[float] | None = None,
    cam_quat: tuple[float] | None = None,
    *,
    show: bool = False,
    save: bool = False,
    save_path: str | Path | None = None,
) -> Image.Image:
    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # MuJoCo visualisation configuration
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # Update rendering engine
    camera = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "ortho-cam",
    )

    # If camera not found, use default camera
    if camera == -1:
        msg = "Camera 'ortho-cam' not found. Using default camera."
        log.debug(msg)
        camera = None
    else:
        model.cam_fovy[camera] = cam_fovy or model.cam_fovy[camera]
        model.cam_pos[camera] = cam_pos or model.cam_pos[camera]
        model.cam_quat[camera] = cam_quat or model.cam_quat[camera]
        model.cam_sensorsize[camera] = [width, height]

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=width,
        height=height,
    ) as renderer:
        # Move simulation forward one iteration/step
        mujoco.mj_step(model, data, nstep=steps)

        # Update the renderer's camera
        renderer.update_scene(
            data,
            scene_option=viz_options,
            camera=camera,
        )

        # Generate frame using rendering engine
        frame = renderer.render()

        # Convert frame into an image which can be shown
        img: Image.Image = Image.fromarray(frame)

    # Save image locally
    if save is True:
        if save_path is None:
            save_path = generate_save_path(file_path="img.png")
        img.save(save_path, format="png")

    # Show image
    if show is True:
        img.show()

    # Return image
    return img


def video_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    video_recorder: VideoRecorder | None = None,
    cam_fovy: float | None = None,
    cam_pos: tuple[float] | None = None,
    cam_quat: tuple[float] | None = None,
) -> None:
    """
    Render a video of the simulation using MuJoCo's rendering engine.

    Parameters
    ----------
    model
        The MuJoCo model to render.
    data
        The MuJoCo data to render.
    duration
        The duration of the video in seconds, by default 10.0
    video_recorder
        The video recorder to use, by default None
    """
    # Get video recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()

    # MuJoCo visualisation configuration
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Calculate steps per frame to avoid single iterations (see 'Notes'.)
    options = mujoco.MjOption()
    steps_per_frame = duration / (
        options.timestep * duration * video_recorder.fps
    )

    # Update rendering engine
    camera = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "ortho-cam",
    )

    # If camera not found, use default camera
    if camera == -1:
        msg = "Camera 'ortho-cam' not found. Using default camera."
        log.debug(msg)
        camera = None
    else:
        model.cam_fovy[camera] = (
            cam_fovy if cam_fovy is not None else model.cam_fovy[camera]
        )
        model.cam_pos[camera] = (
            cam_pos if cam_pos is not None else model.cam_pos[camera]
        )
        model.cam_quat[camera] = (
            cam_quat if cam_quat is not None else model.cam_quat[camera]
        )
        model.cam_sensorsize[camera] = [
            video_recorder.width,
            video_recorder.height,
        ]

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=video_recorder.width,
        height=video_recorder.height,
    ) as renderer:
        while data.time < duration:
            # Move simulation forward one iteration/step
            mujoco.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Update rendering engine
            renderer.update_scene(
                data,
                scene_option=viz_options,
                camera=camera,
            )

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Exit (and save locally) the generated video
    msg = "Finish video rendering"
    num_frames = video_recorder.frame_count
    log.info(f"--> {msg}: {num_frames=}")
    video_recorder.release()


def tracking_video_renderer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float = 10.0,
    video_recorder: VideoRecorder | None = None,
    geom_to_track: str = "robot-core",
    tracking_distance: float = 1.5,
    tracking_azimuth: float = 135,
    tracking_elevation: float = -30,
) -> None:
    """
    Render a video of the simulation with camera tracking the "core" module.

    Parameters
    ----------
    model
        The MuJoCo model to render.
    data
        The MuJoCo data to render.
    duration
        The duration of the video in seconds, by default 10.0
    video_recorder
        The video recorder to use, by default None
    geom_to_track
        The name of the body to track, by default "robot-core"
    tracking_distance
        The distance of the camera from the body, by default 1.5
    tracking_azimuth
        The azimuth angle of the camera, by default 135
    tracking_elevation
        The elevation angle of the camera, by default -30
    """
    # Get video recorder
    if video_recorder is None:
        video_recorder = VideoRecorder()

    # MuJoCo visualisation configuration
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Find the core body ID for tracking
    try:
        core_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            geom_to_track,
        )
        msg = f"Tracking core body ID: {core_body_id}"
        log.info(msg)
    except ValueError:
        msg = f"Body name '{geom_to_track}' not found in the model."
        msg += " Using default camera."
        log.warning(msg)
        core_body_id = None
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and "core" in body_name:
                core_body_id = i
                break

    # Calculate steps per frame to avoid single iterations
    options = mujoco.MjOption()
    steps_per_frame = duration / (
        options.timestep * duration * video_recorder.fps
    )

    # Call rendering engine
    with mujoco.Renderer(
        model,
        width=video_recorder.width,
        height=video_recorder.height,
    ) as renderer:
        while data.time < duration:
            # Move simulation forward one iteration/step
            mujoco.mj_step(model, data, nstep=math.floor(steps_per_frame))

            # Set up tracking camera if core body found
            if core_body_id is not None:
                # Create a tracking camera
                camera = mujoco.MjvCamera()
                camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                camera.trackbodyid = core_body_id
                camera.distance = tracking_distance
                camera.azimuth = tracking_azimuth
                camera.elevation = tracking_elevation

                # Update the renderer's camera
                renderer.update_scene(
                    data,
                    scene_option=scene_option,
                    camera=camera,
                )
            else:
                # Use default camera
                renderer.update_scene(data, scene_option=scene_option)

            # Save frame
            video_recorder.write(frame=renderer.render())

    # Exit (and save locally) the generated video
    msg = "Finish video rendering"
    num_frames = video_recorder.frame_count
    log.info(f"[bold green] --> {msg} [/bold green]: {num_frames=}")
    video_recorder.release()
