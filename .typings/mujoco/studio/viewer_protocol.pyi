import abc
import contextlib
import dataclasses
import mujoco
from _typeshed import Incomplete
from mujoco.studio import endpoints as endpoints, hot_reload as hot_reload, messages as messages, plugin_registry as plugin_registry, ux as ux
from mujoco.studio.web import client_camera as client_camera
from typing import Any

logger: Incomplete
GFX_MODES: Incomplete

@dataclasses.dataclass
class ViewerConfig:
    """Common configuration for creating a viewer window."""
    title: str = ...
    width: int = ...
    height: int = ...
    gfx: str = ...
    http_port: int = ...
    open_browser: bool = ...
    hot_reload: str = ...

@dataclasses.dataclass(frozen=True)
class ViewerInitEvent(messages.Event):
    """Lifecycle event dispatched locally once when a ``Viewer`` initializes.

  ``plugin.ViewerPlugin`` caches ``event.viewer`` as ``self.viewer`` above
  ``CRITICAL`` priority, so ``self.viewer`` is already set when subclass
  handlers run.
  """
    viewer: Viewer

@dataclasses.dataclass(frozen=True)
class ThemeEvent(messages.Event):
    """Dispatched on the viewer when the host changes the UI theme."""
    theme: ux.GuiTheme

class Viewer(abc.ABC, metaclass=abc.ABCMeta):
    """Base class for any viewer.

  Owns the communication endpoint, plugin registry and core visualization
  objects. The application is rendered by calling ``sync()``.
  """
    config: Incomplete
    exit_reason: str
    model: mujoco.MjModel
    data: mujoco.MjData
    model_path: str
    camera: Incomplete
    cam_speed: Incomplete
    client_camera: client_camera.ClientCameraSync | None
    camera_index: Incomplete
    perturb: Incomplete
    vis_options: Incomplete
    extra_geoms: Incomplete
    render_flags: Incomplete
    plugins: Incomplete
    def __init__(self, config: ViewerConfig, endpoint: endpoints.ViewerEndpoint, *, model: mujoco.MjModel | None = None, plugins: list[Any] | None = None, camera: mujoco.MjvCamera | None = None, vis_options: mujoco.MjvOption | None = None, perturb: mujoco.MjvPerturb | None = None, render_flags: ux.RenderFlags | None = None, extra_geoms: list[mujoco.MjvGeom] | None = None) -> None:
        """Initializes the Viewer.

    Args:
      config: Viewer window configuration.
      endpoint: The viewer endpoint for communication with the sim side.
      model: Optional initial MjModel. If None, an empty model is created from
        an empty MjSpec. The Viewer deep-copies this model and creates its own
        MjData.
      plugins: Optional list of plugin instances for viewer-side processing.
      camera: Camera parameters. Internal object is created if None.
      vis_options: Visualization options. Internal object is created if None.
      perturb: Perturbation parameters. Internal object is created if None.
      render_flags: Render flags. Internal object is created if None.
      extra_geoms: List of extra geoms. Internal list is created if None.
    """
    def close(self) -> None:
        """Closes the viewer, sends an exit event and shuts down the endpoint."""
    def is_running(self) -> bool:
        """Returns True while the viewer has not been closed."""
    def send_to_sim(self, message: messages.Message) -> None:
        """Sends a message to the simulation process."""
    def send_if_changes(self, *make_messages: messages.MakeMessage) -> contextlib.AbstractContextManager[None]:
        """Sends the messages that change during the ``with`` block to the sim.

    Calls each ``make_message()`` on entering and on leaving the block and
    sends those that differ. Nothing is sent if the viewer stops running or
    its model is replaced during the block.
    """
    def get_sim_events(self) -> list[messages.Event]:
        """Returns all pending events from the simulation."""
    def get_sim_snapshots(self) -> list[messages.Snapshot]:
        """Returns all pending latest snapshots from the simulation, one per type."""
    def dispatch(self, message: messages.Message) -> None:
        """Dispatches a message to registered handlers in priority order."""
    def load_model(self, model: mujoco.MjModel, model_path: str = '') -> None:
        """Deep-copies a model and creates fresh data for the viewer."""
    @abc.abstractmethod
    def prepare_next_frame(self) -> bool:
        """Advances to the next frame; returns whether one is ready to render."""
    @abc.abstractmethod
    def sync(self) -> None:
        """Renders the scene using the viewer's current model and data."""
    def sync_idle(self) -> None:
        """Updates viewer state when prepare_next_frame() returned False."""
    @abc.abstractmethod
    def get_drop_file(self) -> str: ...
    @abc.abstractmethod
    def upload_image(self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int) -> int: ...

def run_viewer_loop(viewer: Viewer) -> None:
    """Minimal viewer loop: process sim messages, dispatch lifecycle events, sync.

  Runs until the viewer window is closed or an exit event is received.
  On exit, closes the viewer (which sends an ExitEvent to the sim side). The
  viewer is closed even if a plugin handler raises, so its resources are
  released on this thread and the sim side is told to exit before the
  exception propagates.

  An unhandled exception is also logged here, on the viewer thread, as soon as
  it happens: the sim side only re-raises it the next time it touches the
  handle, which may be much later or never (e.g. a notebook cell that has
  already returned).

  Args:
    viewer: A Viewer that owns the endpoint and plugin registry.
  """
