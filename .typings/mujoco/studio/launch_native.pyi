import mujoco
from collections.abc import Iterable
from mujoco.studio import endpoints as endpoints, launch_thread as launch_thread, messages as messages, plugin as plugin, viewer_handle as viewer_handle, viewer_protocol as viewer_protocol
from typing import Any

def run_native_viewer(config: viewer_protocol.ViewerConfig, endpoint: endpoints.ViewerEndpoint, plugins: list[Any] | None = None) -> None:
    """Creates a NativeViewer and runs the viewer loop.

  Args:
    config: Configuration specifying the viewer window settings.
    endpoint: Endpoint for communicating with the simulation side.
    plugins: Optional list of viewer-side plugin instances.
  """
def launch(config: viewer_protocol.ViewerConfig | None = None, *, plugins: Iterable[plugin.Plugin] | None = None) -> viewer_handle.ViewerHandle:
    """Launches the NativeViewer in a daemon thread without blocking.

  Args:
    config: Optional ViewerConfig (defaults to standard ViewerConfig()).
    plugins: Optional viewer and sim plugins.

  Returns:
    A ViewerHandle for interacting with the viewer.
  """
def run(config: viewer_protocol.ViewerConfig | None = None, *, model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None, model_path: str | None = None, plugins: Iterable[plugin.Plugin] | None = None) -> None:
    """Launches the NativeViewer and runs the blocking simulation sync loop.

  Args:
    config: Optional ViewerConfig (defaults to standard ViewerConfig()).
    model: An already compiled model to show; exclusive with ``model_path``.
    data: The data for ``model``; a fresh one is created when None.
    model_path: A model file (XML, URDF, MJB or MJZ) to load on startup.
    plugins: Optional viewer and sim plugins.

  Raises:
    ValueError: If both ``model`` and ``model_path`` are given.
  """
