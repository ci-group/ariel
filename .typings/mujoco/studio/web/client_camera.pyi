import mujoco

class ClientCameraSync:
    """Tracks the viewer's camera generation; see the module docstring."""
    generation: int
    def __init__(self) -> None: ...
    def apply(self, camera: mujoco.MjvCamera, message: bytes) -> bool:
        """Adopts a camera the browser sent, unless it predates `generation`.

    The browser only moves the camera within the mode the viewer streamed to it
    (lookat, distance, azimuth and elevation); switching the mode (type,
    fixedcamid, trackbodyid, orthographic) is the viewer's, which starts a new
    generation. So a message must carry the viewer's current mode: the browser
    never sets camera or body indices, which the viewer would use unchecked.

    Args:
      camera: The viewer's camera, updated in place.
      message: The camera as sent by the browser, a ClientCameraMessage (see
        state_payload.h).

    Returns:
      Whether the camera was adopted; False if the message is malformed (incl.
      non-finite values), changes the camera mode, or predates the viewer's
      latest change (including one not yet streamed).
    """
    def update_generation(self, camera: mujoco.MjvCamera) -> int:
        """Returns the generation to stream with `camera`.

    Starts a new generation if the viewer changed the camera since the last
    call (or since it last adopted the browser's).

    Args:
      camera: The viewer's camera, as about to be streamed to the browser.
    """
