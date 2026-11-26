import contextlib
from collections.abc import Generator
from mujoco.cgl import cgl as cgl

class GLContext:
    """An EGL context for headless accelerated OpenGL rendering on GPU devices."""
    def __init__(self, max_width, max_height) -> None: ...
    def make_current(self) -> None: ...
    @contextlib.contextmanager
    def as_current(self) -> Generator[None]:
        """Temporarily makes this context current, restoring any prior context."""
    def free(self) -> None:
        """Frees resources associated with this context."""
    def __del__(self) -> None: ...
