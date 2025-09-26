__all__ = ['version', 'is_available', 'get_max_alg_id']

def version() -> int | None:
    """Return the version of cuSPARSELt"""
def is_available() -> bool:
    """Return a bool indicating if cuSPARSELt is currently available."""
def get_max_alg_id() -> int | None: ...
