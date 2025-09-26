import numpy as np
from _typeshed import Incomplete
from mujoco import _enums as _enums, _functions as _functions, _render as _render, _structs as _structs, gl_context as gl_context

class Renderer:
    """Renders MuJoCo scenes."""
    _width: Incomplete
    _height: Incomplete
    _model: Incomplete
    _scene: Incomplete
    _scene_option: Incomplete
    _rect: Incomplete
    _gl_context: Incomplete
    _mjr_context: Incomplete
    _depth_rendering: bool
    _segmentation_rendering: bool
    def __init__(self, model: _structs.MjModel, height: int = 240, width: int = 320, max_geom: int = 10000) -> None:
        """Initializes a new `Renderer`.

    Args:
      model: an MjModel instance.
      height: image height in pixels.
      width: image width in pixels.
      max_geom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. If None this will be chosen automatically
        based on the estimated maximum number of renderable geoms in the model.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    @property
    def model(self): ...
    @property
    def scene(self) -> _structs.MjvScene: ...
    @property
    def height(self): ...
    @property
    def width(self): ...
    def enable_depth_rendering(self) -> None: ...
    def disable_depth_rendering(self) -> None: ...
    def enable_segmentation_rendering(self) -> None: ...
    def disable_segmentation_rendering(self) -> None: ...
    def render(self, *, out: np.ndarray | None = None) -> np.ndarray:
        """Renders the scene as a numpy array of pixel values.

    Args:
      out: Alternative output array in which to place the resulting pixels. It
        must have the same shape as the expected output but the type will be
        cast if necessary. The expted shape depends on the value of
        `self._depth_rendering`: when `True`, we expect `out.shape == (width,
        height)`, and `out.shape == (width, height, 3)` when `False`.

    Returns:
      A new numpy array holding the pixels with shape `(H, W)` or `(H, W, 3)`,
      depending on the value of `self._depth_rendering` unless
      `out is None`, in which case a reference to `out` is returned.

    Raises:
      RuntimeError: if this method is called after the close method.
    """
    def update_scene(self, data: _structs.MjData, camera: int | str | _structs.MjvCamera = -1, scene_option: _structs.MjvOption | None = None):
        """Updates geometry used for rendering.

    Args:
      data: An instance of `MjData`.
      camera: An instance of `MjvCamera`, a string or an integer
      scene_option: A custom `MjvOption` instance to use to render the scene
        instead of the default.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if camera does
        not exist.
    """
    def close(self) -> None:
        """Frees the resources used by the renderer.

    This method can be used directly:

    ```python
    renderer = Renderer(...)
    # Use renderer.
    renderer.close()
    ```

    or via a context manager:

    ```python
    with Renderer(...) as renderer:
      # Use renderer.
    ```
    """
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def __del__(self) -> None: ...
