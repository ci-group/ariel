from .backend_config import BackendConfig

__all__ = ['get_onednn_backend_config']

def get_onednn_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native ONEDNN backend.
    """
