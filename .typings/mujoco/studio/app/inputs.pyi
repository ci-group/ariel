import mujoco
from mujoco.studio import sim as sim, ux as ux
from typing import Any

def handle_vis_options_keyboard_events(vis_options: mujoco.MjvOption, is_freecam_wasd: bool) -> bool:
    """Toggles visualization flags based on keyboard shortcuts.

  Args:
    vis_options: The visualization options to modify.
    is_freecam_wasd: If True, keys Q/E/A/D are reserved for camera movement and
      will not toggle visualization flags.

  Returns:
    True if a key was handled, False otherwise.
  """
def handle_step_control_keyboard_events(step_control: sim.StepControl, ux_state: ux.UxState) -> bool:
    """Handles keyboard shortcuts for simulation stepping control.

  Args:
    step_control: The simulation step control object.
    ux_state: The UX state object.

  Returns:
    True if a key was handled, False otherwise.
  """
def handle_sim_history_keyboard_events(step_control: sim.StepControl, ux_state: ux.UxState) -> bool:
    """Handles keyboard shortcuts for scrubbing the simulation history.

  Left/Right load the previous/next recorded frame while paused; Right at the
  head of the history requests a single step instead. Matches the C++ Studio
  app.

  Args:
    step_control: The simulation step control object.
    ux_state: The UX state holding the timeline history size and index.

  Returns:
    True if a key was handled, False otherwise.
  """
def handle_reset_keyboard_events(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Handles keyboard shortcuts for simulation reset."""
def handle_camera_select_keyboard_events(model: mujoco.MjModel, camera: mujoco.MjvCamera, ux_state: ux.UxState) -> bool:
    """Handles keyboard shortcuts for camera selection.

  Args:
    model: The MuJoCo model.
    camera: The MuJoCo camera object.
    ux_state: The UX state object.

  Returns:
    True if a key was handled, False otherwise.
  """
def handle_miscellaneous_keyboard_events(app: Any) -> bool:
    """Handles miscellaneous keyboard shortcuts for toggling panels and windows.

  Tab/Shift+Tab toggle Options and Inspector panels.
  F2/F3 the Info and Profiler windows.

  Args:
    app: The viewer application instance, whose config holds the UI state flags.

  Returns:
    True if a key was handled, False otherwise.
  """
def handle_freecam_wasd_keyboard_events(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, cam_speed: float) -> tuple[bool, float]:
    """Handles keyboard shortcuts for free camera movement.

  Flies the camera with W/S, A/D and Q/E using the camera controls shared with
  the C++ Studio app and the web viewer (ux.move_camera_with_keys).

  Args:
    model: The MuJoCo model.
    data: The MuJoCo data.
    camera: The MuJoCo camera object.
    cam_speed: The current camera speed.

  Returns:
    A tuple of (handled, updated_cam_speed).
  """
def handle_keyboard_events(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, vis_options: mujoco.MjvOption, step_control: sim.StepControl, ux_state: ux.UxState, cam_speed: float) -> tuple[bool, float]:
    """Handle keyboard events according to Studio's bindings.

  Args:
    model: The MuJoCo model.
    data: The MuJoCo data.
    camera: The MuJoCo camera object.
    vis_options: The MuJoCo visualization options.
    step_control: The simulation step control object.
    ux_state: The UX state object.
    cam_speed: The current camera speed.

  Returns:
    A tuple of (handled, updated_cam_speed).
  """
def handle_camera_tracking_mouse_events(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, vis_options: mujoco.MjvOption, ux_state: ux.UxState) -> None:
    """Handles mouse events for camera tracking."""
def handle_mouse_events(model: mujoco.MjModel, data: mujoco.MjData, camera: mujoco.MjvCamera, vis_options: mujoco.MjvOption, perturb: mujoco.MjvPerturb, ux_state: ux.UxState, move_camera: bool = True) -> None:
    """Handles mouse events.

  Args:
    model: The MuJoCo model.
    data: The MuJoCo data.
    camera: The MuJoCo camera object.
    vis_options: The MuJoCo visualization options.
    perturb: The MuJoCo perturbation object.
    ux_state: The UX state object.
    move_camera: Whether drags and scrolls move the camera. False when the
      viewer's client moves it itself (Viewer.client_camera); picking and
      perturbation are handled either way.
  """
