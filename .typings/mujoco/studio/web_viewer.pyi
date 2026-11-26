import mujoco
from _typeshed import Incomplete
from mujoco.studio import endpoints as endpoints, messages as messages, ux as ux, viewer_protocol as viewer_protocol
from mujoco.studio.implot import implot as implot
from mujoco.studio.web import client_camera as client_camera, headless_ui as headless_ui, state_payload as state_payload, web_server as web_server
from typing import Any

class WebViewer(viewer_protocol.Viewer):
    """Simulation-agnostic web viewer for MuJoCo models."""
    client_camera: Incomplete
    def __init__(self, config: viewer_protocol.ViewerConfig, endpoint: endpoints.ViewerEndpoint, *, model: mujoco.MjModel | None = None, plugins: list[Any] | None = None, camera: mujoco.MjvCamera | None = None, vis_options: mujoco.MjvOption | None = None, perturb: mujoco.MjvPerturb | None = None, render_flags: ux.RenderFlags | None = None, extra_geoms: list[mujoco.MjvGeom] | None = None) -> None:
        """Initializes the WebViewer.

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
    @property
    def port(self) -> int:
        """Public HTTP/WebSocket port bound by this viewer."""
    def prepare_next_frame(self) -> bool:
        """Advances to the next headless frame; returns False when disconnected."""
    def sync(self) -> None:
        """Streams state to the browser and ends the headless ImGui frame."""
    def sync_idle(self) -> None:
        """Streams state to spectators when no /ui controller drives frames."""
    def close(self) -> None: ...
    def get_drop_file(self) -> str:
        """Returns the path of a model file dropped onto the browser page, or ''.

    The browser uploads the dropped files' bytes over the /drop WebSocket
    (so drops work even when the browser runs on another machine). They are
    written to a temporary directory here, preserving names and relative
    paths — a dropped folder's asset references resolve, and the mjz/zip
    decoder can locate an archive's root XML by the archive's own name —
    and the regular drop-loading flow applies (studio_app.App sends a
    LoadModelEvent; the sim thread parses the file).
    """
    def upload_image(self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int) -> int:
        """Uploads an image to the browser over the UI stream."""
