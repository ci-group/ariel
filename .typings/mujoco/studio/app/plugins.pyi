import dataclasses
import logging
import mujoco
import threading
from _typeshed import Incomplete
from mujoco.studio import messages as messages, plugin as plugin, sim as sim, sim_history as sim_history, step_control as step_control, ux as ux, viewer_utils as viewer_utils
from mujoco.studio.app import inputs as inputs
from typing import Any

class _StudioLogHandler(logging.Handler):
    """Adds Python log records from any thread to the Studio Console.

  The records are only added to the console. Python's other logging handlers
  still print them, or logging.lastResort if there are none.
  """
    def emit(self, record: logging.LogRecord) -> None: ...

class _ConsoleHooks:
    """Adds Python warnings and uncaught exceptions to the Studio Console.

  Each hook adds the warning or exception to the console and then calls the
  hook it replaced, which prints it as before.
  """
    orig_showwarning: Incomplete
    orig_threading_excepthook: Incomplete
    orig_sys_excepthook: Incomplete
    def __init__(self) -> None: ...
    def showwarning(self, message: Warning | str, category: type[Warning], filename: str, lineno: int, file: Any = None, line: str | None = None) -> None:
        """Forwards warnings to the Studio Console and original hook."""
    def threading_excepthook(self, args: threading.ExceptHookArgs) -> None:
        """Forwards thread exceptions to the Studio Console and original hook."""
    def sys_excepthook(self, exc_type: type[BaseException], exc_value: BaseException, exc_traceback: Any) -> None: ...

def install_console_hooks() -> None:
    """Adds logs, warnings, and uncaught exceptions to the Studio Console.

  Installs the MuJoCo log handler (see ux.init_console), adds a handler for
  Python log records, and hooks warnings.showwarning, threading.excepthook, and
  sys.excepthook. Each hook adds to the console and then calls what it replaced
  (falling back to logging.lastResort for log records that reach no other
  handler), leaving terminal and file output unchanged. Only the first call
  installs these; later calls only check again how logging is configured, e.g.
  whether web_server has configured its logger since.
  """

@dataclasses.dataclass(frozen=True)
class InitEvent(messages.Event):
    """Lifecycle event dispatched once when the studio_app App is initialised.

  Plugins that need access to the studio_app App should handle this event
  and cache the reference.
  """
    app: App

@dataclasses.dataclass
class Config:
    """Configures the studio_app App.

  Pass an instance to App to choose what is visible on startup. The App keeps
  the instance as its live UI state, so the View menu and the keyboard
  shortcuts toggle these same fields.
  """
    show_info: bool = ...
    show_profiler: bool = ...
    show_console: bool = ...
    show_options: bool = ...
    show_toolbar: bool = ...
    show_view_gizmo: bool = ...
    show_status_bar: bool = ...
    show_inspector: bool = ...

SIM_PLUGINS: tuple[plugin.SimPlugin, ...]

class App(plugin.ViewerPlugin):
    """Wraps a Viewer and adds Studio UI/UX.

  Pair with ``*SIM_PLUGINS`` in ``plugins=[...]`` for the default sim-side
  real-time CPU stepping and history scrubbing.
  """
    @property
    def model(self) -> mujoco.MjModel: ...
    @model.setter
    def model(self, value: mujoco.MjModel) -> None: ...
    @property
    def model_path(self) -> str: ...
    @model_path.setter
    def model_path(self, value: str) -> None: ...
    @property
    def data(self) -> mujoco.MjData: ...
    @data.setter
    def data(self, value: mujoco.MjData) -> None: ...
    theme: Incomplete
    config: Incomplete
    status: str
    def __init__(self, config: Config | None = None) -> None: ...
    def align_camera(self) -> None:
        """Recenter the camera on the model's home camera, else the free camera."""
    def reload_model(self) -> None:
        """Asks the sim to reload the current model."""
    def close(self) -> None: ...
    def handle_keyboard_events(self) -> None:
        """Handles keyboard events."""
    def handle_camera_tracking_mouse_events(self) -> None:
        """Handles mouse events for camera tracking."""
    def handle_mouse_events(self) -> None:
        """Handles mouse events."""
    def reset_physics(self) -> None:
        """Reset the physics."""
    def save_keyframe(self) -> None:
        """Save the active state to the selected keyframe and notify the sim."""
    def apply_perturb(self) -> None:
        """Apply perturbation the model."""
    def reset_physics_gui(self) -> None:
        """GUI to Reset the physics i.e., the reset button."""
    def is_running(self) -> bool:
        """Returns True if the application should continue running."""
    def update(self) -> None:
        """Update the simulation and handle user input.

    Handles mouse input to compute perturbations or camera motion.
    Handles keyboard input e.g., for keybindings or camera motion.
    Applies the perturbations and advances the physics.
    """
    def build_gui(self) -> None:
        """Emit full Studio UI."""
