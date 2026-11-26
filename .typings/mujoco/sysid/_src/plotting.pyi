import mujoco
import numpy as np
from collections.abc import Sequence

def render_rollout(model: mujoco.MjModel | Sequence[mujoco.MjModel], data: mujoco.MjData, state: np.ndarray, framerate: int, camera: str | int = -1, width: int = 640, height: int = 480, light_pos: Sequence[float] | None = None) -> list[np.ndarray]:
    """Renders a rollout or batch of rollouts.

  Args:
    model: Single model or list of models (one per batch).
    data: MjData scratch object.
    state: State array of shape (nbatch, nsteps, nstate).
    framerate: Frames per second to render.
    camera: Camera name or ID.
    width: Image width.
    height: Image height.
    light_pos: Optional light position [x, y, z] to add a spotlight.

  Returns:
    List of rendered frames (numpy arrays).
  """
