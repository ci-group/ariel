import contextlib
from _typeshed import Incomplete
from collections.abc import Generator

PYOPENGL_PLATFORM: Incomplete

class GLContext:
    """An OSMesa context for software-based OpenGL rendering."""
    def __init__(self, max_width, max_height) -> None:
        """Initializes this OSMesa context."""
    def make_current(self) -> None: ...
    @contextlib.contextmanager
    def as_current(self) -> Generator[None]:
        """Temporarily makes this context current, restoring any prior context."""
    def free(self) -> None:
        """Frees resources associated with this context."""
    def __del__(self) -> None: ...
