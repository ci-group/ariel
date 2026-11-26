import dataclasses
import mujoco
from collections.abc import Buffer
from mujoco import mjrf as mjrf
from typing import Any, Protocol

class SupportsDLPack(Protocol):
    """Protocol for objects supporting the DLPack tensor exchange protocol."""
    def __dlpack__(self, /, *, stream: Any = None) -> Any: ...
    def __dlpack_device__(self, /) -> tuple[int, int]: ...
BufferType = Buffer | SupportsDLPack

@dataclasses.dataclass
class View:
    """A representation of an mjrf.RenderRequest that works with the Renderer."""
    scene: str
    target: str
    viewport: tuple[float, float, float, float] = ...
    camera: mjrf.Camera | mujoco.MjvGLCamera | None = ...
    draw_mode: mjrf.DrawMode = ...
    post_processing: bool = ...
    reflections: bool = ...
    shadows: bool = ...
    skybox: bool = ...
    haze: bool = ...
    headlight: bool = ...
    headlight_color: tuple[float, float, float] = ...
    headlight_intensity: float = ...

class Renderer:
    """Coordinates the rendering of scenes to render targets.

  Internally, the Renderer manages the mjrf objects (i.e. Scene, Camera, and
  RenderTarget) that are needed to render a MuJoCo model. The Scene defines
  the entities from the model that are to be rendered. The Camera defines the
  viewpoint from which to render the model. And, finally, the RenderTarget
  defines the memory buffers into which the final rendered image will be
  written.

  Each object owned by the Renderer is associated with a unique name. This name
  is used to identify the object when making rendering calls.

  The Renderer does not specify the contents of the Scene; it is up to the user
  to populate the Scene with the entities that they want to render.

  The Renderer is not limited to rendering a single scene to a single target.
  You can configure the Renderer to manage multiple scenes from multiple
  cameras into multiple targets. Each such render operation is specified using
  the view() method.
  """
    def __init__(self, ctx: mjrf.Context) -> None: ...
    @property
    def context(self) -> mjrf.Context:
        """Returns the underlying mjrf.Context."""
    def scene(self, name: str) -> mjrf.Scene:
        """Returns the scene with the given name, creating it if needed."""
    def target(self, name: str, buffer: BufferType, pixel_format=...):
        """Defines a render target with the given name and destination buffer."""
    def render(self, views: list[View]):
        """Renders all active views."""
    @dataclasses.dataclass
    class _Read:
        """Internal wrapper around mjrf.ReadRequest."""
        size: tuple[int, int]
        target: mjrf.RenderTarget
        format: mjrf.PixelFormat
        request: mjrf.ReadRequest
        buffer: tuple[BufferType, Buffer]
