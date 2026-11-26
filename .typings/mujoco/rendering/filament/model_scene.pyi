import mujoco
from mujoco import mjrf as mjrf

class ModelScene:
    """Manages Lights and Renderables in an Scene based on mjModel/mjData."""
    def __init__(self, ctx: mjrf.Context, model: mujoco.MjModel, scene: mjrf.Scene | None = None) -> None: ...
    def update(self, data: mujoco.MjData):
        """Updates the lights and renderables in the scene from the mjData."""
    @property
    def scene(self) -> mjrf.Scene:
        """Returns the scene."""
    def get_light(self, index: int) -> mjrf.Light:
        """Returns the light with the given index in the mjModel."""
    def get_renderable(self, obj_type: mujoco.mjtObj, obj_index: int, *, sub_index: int = -1) -> mjrf.Renderable:
        """Returns the renderable corresponding to the given mjModel object."""
