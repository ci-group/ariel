import mujoco
from mujoco.studio import messages as messages, viewer_protocol as viewer_protocol

def send_state(viewer: viewer_protocol.Viewer, model: mujoco.MjModel, data: mujoco.MjData, state_sig: int) -> None:
    """Sends the selected components of the viewer's state to the sim thread."""
def apply_perturb(viewer: viewer_protocol.Viewer, model: mujoco.MjModel | None, data: mujoco.MjData | None, is_paused: bool = False) -> None:
    """Apply perturbation to the model and send updated forces to the sim thread."""
