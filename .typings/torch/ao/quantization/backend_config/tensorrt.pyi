from .backend_config import BackendConfig

__all__ = ['get_tensorrt_backend_config', 'get_tensorrt_backend_config_dict']

def get_tensorrt_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for the TensorRT backend.
    NOTE: Current api will change in the future, it's just to unblock experimentation for
    new backends, please don't use it right now.
    TODO: add a README when it's more stable
    """
def get_tensorrt_backend_config_dict():
    """
    Return the `BackendConfig` for the TensorRT backend in dictionary form.
    """
