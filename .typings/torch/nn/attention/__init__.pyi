import contextlib
from torch._C import _SDPBackend as SDPBackend

__all__ = ['SDPBackend', 'sdpa_kernel', 'WARN_FOR_UNFUSED_KERNELS']

WARN_FOR_UNFUSED_KERNELS: bool
SDPBackend = SDPBackend

@contextlib.contextmanager
def sdpa_kernel(backends: list[SDPBackend] | SDPBackend, set_priority: bool = False):
    """
    Context manager to select which backend to use for scaled dot product attention.

    .. warning:: This function is beta and subject to change.

    Args:
        backends (Union[List[SDPBackend], SDPBackend]): A backend or list of backends for scaled dot product attention.
        set_priority_order (bool=False): Whether the ordering of the backends is interpreted as their priority order.

    Example:

    .. code-block:: python

        from torch.nn.functional import scaled_dot_product_attention
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # Only enable flash attention backend
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            scaled_dot_product_attention(...)

        # Enable the Math or Efficient attention backends
        with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
            scaled_dot_product_attention(...)

    This context manager can be used to select which backend to use for scaled dot product attention.
    Upon exiting the context manager, the previous state of the flags will be restored, enabling all backends.
    """
