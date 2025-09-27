import mujoco._render
import numpy
from typing import ClassVar

class Mutex:
    def __init__(self, *args, **kwargs) -> None:
        """Initialize self.  See help(type(self)) for accurate signature."""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def __enter__(self) -> None:
        """__enter__(self: mujoco._simulate.Mutex) -> None"""
    def __exit__(self, arg0: object, arg1: object, arg2: object) -> None:
        """__exit__(self: mujoco._simulate.Mutex, arg0: object, arg1: object, arg2: object) -> None"""

class Simulate:
    MAX_GEOM: ClassVar[int] = ...  # read-only
    droploadrequest: int
    load_error: str
    measured_slowdown: float
    speed_changed: bool
    ui0_enable: int
    ui1_enable: int
    def __init__(self, arg0: object, arg1: object, arg2: object, arg3: object, arg4: bool, arg5: object) -> None:
        """__init__(self: mujoco._simulate.Simulate, arg0: object, arg1: object, arg2: object, arg3: object, arg4: bool, arg5: object) -> None"""
    def _pybind11_conduit_v1_(self, *args, **kwargs): ...
    def add_to_history(self) -> None:
        """add_to_history(self: mujoco._simulate.Simulate) -> None"""
    def clear_figures(self) -> None:
        """clear_figures(self: mujoco._simulate.Simulate) -> None"""
    def clear_images(self) -> None:
        """clear_images(self: mujoco._simulate.Simulate) -> None"""
    def clear_texts(self) -> None:
        """clear_texts(self: mujoco._simulate.Simulate) -> None"""
    def destroy(self) -> None:
        """destroy(self: mujoco._simulate.Simulate) -> None"""
    def exit(self) -> None:
        """exit(self: mujoco._simulate.Simulate) -> None"""
    def load(self, arg0: object, arg1: object, arg2: str) -> None:
        """load(self: mujoco._simulate.Simulate, arg0: object, arg1: object, arg2: str) -> None"""
    def load_message(self, arg0: str) -> None:
        """load_message(self: mujoco._simulate.Simulate, arg0: str) -> None"""
    def load_message_clear(self) -> None:
        """load_message_clear(self: mujoco._simulate.Simulate) -> None"""
    def lock(self) -> Mutex:
        """lock(self: mujoco._simulate.Simulate) -> mujoco._simulate.Mutex"""
    def render_loop(self) -> None:
        """render_loop(self: mujoco._simulate.Simulate) -> None"""
    def set_figures(self, viewports_figures: list[tuple[mujoco._render.MjrRect, object]]) -> None:
        """set_figures(self: mujoco._simulate.Simulate, viewports_figures: list[tuple[mujoco._render.MjrRect, object]]) -> None"""
    def set_images(self, viewports_images: list[tuple[mujoco._render.MjrRect, numpy.ndarray]]) -> None:
        """set_images(self: mujoco._simulate.Simulate, viewports_images: list[tuple[mujoco._render.MjrRect, numpy.ndarray]]) -> None"""
    def set_texts(self, overlay_texts: list[tuple[int, int, str, str]]) -> None:
        """set_texts(self: mujoco._simulate.Simulate, overlay_texts: list[tuple[int, int, str, str]]) -> None"""
    def sync(self) -> None:
        """sync(self: mujoco._simulate.Simulate) -> None"""
    def uiloadrequest_decrement(self) -> None:
        """uiloadrequest_decrement(self: mujoco._simulate.Simulate) -> None"""
    def update_hfield(self, arg0: int) -> None:
        """update_hfield(self: mujoco._simulate.Simulate, arg0: int) -> None"""
    def update_mesh(self, arg0: int) -> None:
        """update_mesh(self: mujoco._simulate.Simulate, arg0: int) -> None"""
    def update_texture(self, arg0: int) -> None:
        """update_texture(self: mujoco._simulate.Simulate, arg0: int) -> None"""
    @property
    def busywait(self) -> int: ...
    @property
    def ctrl_noise_rate(self) -> float: ...
    @property
    def ctrl_noise_std(self) -> float: ...
    @property
    def d(self) -> object: ...
    @property
    def dropfilename(self) -> str: ...
    @property
    def exitrequest(self) -> int: ...
    @property
    def filename(self) -> str: ...
    @property
    def m(self) -> object: ...
    @property
    def real_time_index(self) -> int: ...
    @property
    def refresh_rate(self) -> int: ...
    @property
    def run(self) -> int: ...
    @property
    def uiloadrequest(self) -> int: ...
    @property
    def viewport(self) -> mujoco._render.MjrRect: ...

def set_glfw_dlhandle(arg0: int) -> None:
    """set_glfw_dlhandle(arg0: int) -> None"""
