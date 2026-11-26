import mujoco
from collections.abc import Iterable
from mujoco.studio import messages as messages, plugin as plugin, viewer_handle as viewer_handle, viewer_protocol as viewer_protocol

def launch(config: viewer_protocol.ViewerConfig, *, plugins: Iterable[plugin.Plugin] | None = None) -> viewer_handle.ViewerHandle:
    """Launches the viewer in a daemon thread without blocking.

  The viewer runs on the rendering thread and renders the scene along with any
  GUI built by registered plugins (such as ``studio_app.App``). The
  caller keeps running and pushes state via ``handle.sync()``.

  Args:
    config: Viewer window configuration.
    plugins: Optional viewer and sim plugins. Include a ``SimPlugin`` that
      handles ``StepEvent`` (e.g. ``step_control.StepControl()``) to advance the
      physics and, at minimum, pace each ``sync`` call.

  Returns:
    A ViewerHandle for interacting with the viewer.
  """
def run(config: viewer_protocol.ViewerConfig, *, model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None, model_path: str | None = None, plugins: Iterable[plugin.Plugin] | None = None) -> None:
    """Launches the viewer and runs the blocking simulation sync loop.

  Args:
    config: Viewer window configuration.
    model: An already compiled model to show; exclusive with ``model_path``.
    data: The data for ``model``; a fresh one is created when None.
    model_path: A model file (XML, URDF, MJB or MJZ) to load on startup.
    plugins: Optional viewer and sim plugins.

  Raises:
    ValueError: If both ``model`` and ``model_path`` are given.
  """
