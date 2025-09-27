from _typeshed import Incomplete

PYOPENGL_PLATFORM: Incomplete
_DEPTH_BITS: int
_STENCIL_BITS: int
_ACCUM_BITS: int

class GLContext:
    """An OSMesa context for software-based OpenGL rendering."""
    _context: Incomplete
    _height: Incomplete
    _width: Incomplete
    _buffer: Incomplete
    def __init__(self, max_width, max_height) -> None:
        """Initializes this OSMesa context."""
    def make_current(self) -> None: ...
    def free(self) -> None:
        """Frees resources associated with this context."""
    def __del__(self) -> None: ...
