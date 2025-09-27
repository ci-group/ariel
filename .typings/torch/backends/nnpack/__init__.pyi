from collections.abc import Generator
from contextlib import contextmanager

__all__ = ['is_available', 'flags', 'set_flags']

def is_available():
    """Return whether PyTorch is built with NNPACK support."""
def set_flags(_enabled):
    """Set if nnpack is enabled globally"""
@contextmanager
def flags(enabled: bool = False) -> Generator[None]:
    """Context manager for setting if nnpack is enabled globally"""
