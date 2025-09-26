from _typeshed import Incomplete
from mujoco.cgl import cgl as cgl

_ATTRIB = cgl.CGLPixelFormatAttribute
_PROFILE = cgl.CGLOpenGLProfile

class GLContext:
    """An EGL context for headless accelerated OpenGL rendering on GPU devices."""
    _pix: Incomplete
    _context: Incomplete
    def __init__(self, max_width, max_height) -> None: ...
    def make_current(self) -> None: ...
    def free(self) -> None:
        """Frees resources associated with this context."""
    def __del__(self) -> None: ...
