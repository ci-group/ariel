import mujoco
from mujoco import mjrf as mjrf
from mujoco.rendering.filament import renderer as renderer
from mujoco.studio import endpoints as endpoints, messages as messages, ux as ux, viewer_protocol as viewer_protocol, window as window
from mujoco.studio.implot import implot as implot
from typing import Any

class NativeViewer(viewer_protocol.Viewer):
    """Simulation-agnostic native viewer for MuJoCo models."""
    def __init__(self, config: viewer_protocol.ViewerConfig, endpoint: endpoints.ViewerEndpoint, *, model: mujoco.MjModel | None = None, plugins: list[Any] | None = None, camera: mujoco.MjvCamera | None = None, vis_options: mujoco.MjvOption | None = None, perturb: mujoco.MjvPerturb | None = None, render_flags: ux.RenderFlags | None = None, extra_geoms: list[mujoco.MjvGeom] | None = None) -> None:
        """Initializes the NativeViewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. Forwarded to the base Viewer.
      plugins: Optional list of plugin instances.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
    """
    def prepare_next_frame(self) -> bool:
        """Advances to the next frame; returns False when the window is closed."""
    def sync(self) -> None:
        """Render the scene and present it to the window."""
    def close(self) -> None:
        """Close the viewer and explicitly destroy the renderer.

    The C++ Viewer (and its FilamentRenderer) must be destroyed on the same
    thread that created it, because Filament's FEngine::destroy() asserts
    thread affinity.  Without this override the pybind11 prevent object would
    be garbage-collected on the main thread, triggering a SIGABRT.
    """
    def get_drop_file(self) -> str:
        """Returns the path of the file dropped into the window, or empty string."""
    def upload_image(self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int) -> int:
        """Uploads an image to the backend for GUI rendering.

    The ID can be used in subsequent calls to update the texture data. An empty
    `img` argument will free the texture if it exists. A `tex_id` of 0 will
    create a new texture.

    Args:
      tex_id: The texture ID.
      img: The image data as string or bytes.
      width: Width of the image.
      height: Height of the image.
      bpp: Bytes per pixel.

    Returns:
      The texture ID.
    """
