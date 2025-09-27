from .backend_config import BackendConfig

__all__ = ['get_executorch_backend_config']

def get_executorch_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for backends PyTorch lowers to through the Executorch stack.
    """
