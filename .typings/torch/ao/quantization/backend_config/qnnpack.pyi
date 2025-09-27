from .backend_config import BackendConfig

__all__ = ['get_qnnpack_backend_config']

def get_qnnpack_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch's native QNNPACK backend.
    """
