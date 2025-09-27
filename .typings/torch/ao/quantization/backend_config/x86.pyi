from .backend_config import BackendConfig

__all__ = ['get_x86_backend_config']

def get_x86_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native x86 backend.
    """
