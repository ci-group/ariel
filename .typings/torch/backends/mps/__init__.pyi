from functools import lru_cache as _lru_cache

__all__ = ['is_built', 'is_available', 'is_macos13_or_newer', 'is_macos_or_newer']

def is_built() -> bool:
    """Return whether PyTorch is built with MPS support.

    Note that this doesn't necessarily mean MPS is available; just that
    if this PyTorch binary were run a machine with working MPS drivers
    and devices, we would be able to use it.
    """
@_lru_cache
def is_available() -> bool:
    """Return a bool indicating if MPS is currently available."""
@_lru_cache
def is_macos_or_newer(major: int, minor: int) -> bool:
    """Return a bool indicating whether MPS is running on given MacOS or newer."""
@_lru_cache
def is_macos13_or_newer(minor: int = 0) -> bool:
    """Return a bool indicating whether MPS is running on MacOS 13 or newer."""
