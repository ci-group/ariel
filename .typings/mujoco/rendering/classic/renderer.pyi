import mujoco
import numpy as np
import types
from _typeshed import Incomplete
from mujoco.rendering.classic import gl_context as gl_context

GLContext: Incomplete

class Renderer:
    """Renders MuJoCo scenes."""
    def __init__(self, model: mujoco.MjModel, height: int = 240, width: int = 320, max_geom: int = 10000, font_scale: mujoco.mjtFontScale = ...) -> None:
        """Initializes a new `Renderer`.

    Args:
      model: an MjModel instance.
      height: image height in pixels.
      width: image width in pixels.
      max_geom: Optional integer specifying the maximum number of geoms that can
        be rendered in the same scene. If None this will be chosen automatically
        based on the estimated maximum number of renderable geoms in the model.
      font_scale: Optional enum specifying the font scale for text.

    Raises:
      ValueError: If `camera_id` is outside the valid range, or if `width` or
        `height` exceed the dimensions of MuJoCo's offscreen framebuffer.
    """
    @property
    def model(self): ...
    @property
    def scene(self) -> mujoco.MjvScene: ...
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
        cast if necessary. The expected shape depends on the rendering mode:
        `(height, width)` when `self._depth_rendering` is `True`,
        `(height, width, 2)` when `self._segmentation_rendering` is `True`, and
        `(height, width, 3)` otherwise.

    Returns:
      A new numpy array holding the pixels with shape `(H, W)`, `(H, W, 2)`, or
      `(H, W, 3)` depending on the rendering mode unless `out is not None`, in
      which case a reference to `out` is returned.

    Raises:
      RuntimeError: if this method is called after the close method.
    """
    def update_scene(self, data: mujoco.MjData, camera: int | str | mujoco.MjvCamera = -1, scene_option: mujoco.MjvOption | None = None):
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
