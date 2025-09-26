import abc
import mujoco
import numpy as np
import queue
from _typeshed import Incomplete
from mujoco import _simulate as _simulate
from typing import Callable

PERCENT_REALTIME: Incomplete
MAX_SYNC_MISALIGN: float
SIM_REFRESH_FRACTION: float
CallbackType: Incomplete
LoaderType: Incomplete
KeyCallbackType = Callable[[int], None]
_LoaderWithPathType: Incomplete
_InternalLoaderType = LoaderType | _LoaderWithPathType
_Simulate: Incomplete

class Handle:
    """A handle for interacting with a MuJoCo viewer."""
    _sim: Incomplete
    _cam: Incomplete
    _opt: Incomplete
    _pert: Incomplete
    _user_scn: Incomplete
    def __init__(self, sim: _Simulate, cam: mujoco.MjvCamera, opt: mujoco.MjvOption, pert: mujoco.MjvPerturb, user_scn: mujoco.MjvScene | None) -> None: ...
    @property
    def cam(self): ...
    @property
    def opt(self): ...
    @property
    def perturb(self): ...
    @property
    def user_scn(self): ...
    @property
    def m(self): ...
    @property
    def d(self): ...
    @property
    def viewport(self): ...
    def set_figures(self, viewports_figures: tuple[mujoco.MjrRect, mujoco.MjvFigure] | list[tuple[mujoco.MjrRect, mujoco.MjvFigure]]):
        """Overlay figures on the viewer.

    Args:
      viewports_figures: Single tuple or list of tuples of (viewport, figure)
        viewport: Rectangle defining position and size of the figure
        figure: MjvFigure object containing the figure data to display
    """
    def clear_figures(self) -> None: ...
    def set_texts(self, texts: tuple[int | None, int | None, str | None, str | None] | list[tuple[int | None, int | None, str | None, str | None]]):
        """Overlay text on the viewer.

    Args:
      texts: Single tuple or list of tuples of (font, gridpos, text1, text2)
        font: Font style from mujoco.mjtFontScale
        gridpos: Position of text box from mujoco.mjtGridPos
        text1: Left text column, defaults to empty string if None
        text2: Right text column, defaults to empty string if None
    """
    def clear_texts(self) -> None: ...
    def set_images(self, viewports_images: tuple[mujoco.MjrRect, np.ndarray] | list[tuple[mujoco.MjrRect, np.ndarray]]):
        """Overlay images on the viewer.

    Args:
      viewports_images: Single tuple or list of tuples of (viewport, image)
        viewport: Rectangle defining position and size of the image
        image: RGB image with shape (height, width, 3)
    """
    def clear_images(self) -> None: ...
    def close(self) -> None: ...
    def _get_sim(self) -> _Simulate | None: ...
    def is_running(self) -> bool: ...
    def lock(self): ...
    def sync(self) -> None: ...
    def update_hfield(self, hfieldid: int): ...
    def update_mesh(self, meshid: int): ...
    def update_texture(self, texid: int): ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...

class _MjPythonBase(metaclass=abc.ABCMeta):
    def launch_on_ui_thread(self, model: mujoco.MjModel, data: mujoco.MjData, handle_return: queue.Queue[Handle] | None, key_callback: KeyCallbackType | None): ...

_MJPYTHON: _MjPythonBase | None

def _file_loader(path: str) -> _LoaderWithPathType:
    """Loads an MJCF model from file path."""
def _reload(simulate: _Simulate, loader: _InternalLoaderType, notify_loaded: Callable[[], None] | None = None) -> tuple[mujoco.MjModel, mujoco.MjData] | None:
    """Internal function for reloading a model in the viewer."""
def _physics_loop(simulate: _Simulate, loader: _InternalLoaderType | None):
    """Physics loop for the GUI, to be run in a separate thread."""
def _launch_internal(model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None, *, run_physics_thread: bool, loader: _InternalLoaderType | None = None, handle_return: queue.Queue[Handle] | None = None, key_callback: KeyCallbackType | None = None, show_left_ui: bool = True, show_right_ui: bool = True) -> None:
    """Internal API, so that the public API has more readable type annotations."""
def launch(model: mujoco.MjModel | None = None, data: mujoco.MjData | None = None, *, loader: LoaderType | None = None, show_left_ui: bool = True, show_right_ui: bool = True) -> None:
    """Launches the Simulate GUI."""
def launch_from_path(path: str) -> None:
    """Launches the Simulate GUI from file path."""
def launch_passive(model: mujoco.MjModel, data: mujoco.MjData, *, key_callback: KeyCallbackType | None = None, show_left_ui: bool = True, show_right_ui: bool = True) -> Handle:
    """Launches a passive Simulate GUI without blocking the running thread."""
