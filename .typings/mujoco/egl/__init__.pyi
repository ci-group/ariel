from _typeshed import Incomplete

PYOPENGL_PLATFORM: Incomplete

def create_initialized_egl_device_display():
    """Creates an initialized EGL display directly on a device."""

EGL_DISPLAY: Incomplete
EGL_ATTRIBUTES: Incomplete

class GLContext:
    """An EGL context for headless accelerated OpenGL rendering on GPU devices."""
    _context: Incomplete
    def __init__(self, max_width, max_height) -> None: ...
    def make_current(self) -> None: ...
    def free(self) -> None:
        """Frees resources associated with this context."""
    def __del__(self) -> None: ...
