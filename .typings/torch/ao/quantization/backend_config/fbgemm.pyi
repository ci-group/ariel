from .backend_config import BackendConfig

__all__ = ['get_fbgemm_backend_config']

def get_fbgemm_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native FBGEMM backend.
    """
