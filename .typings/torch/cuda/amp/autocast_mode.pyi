import torch
from _typeshed import Incomplete
from typing import Any

__all__ = ['autocast', 'custom_fwd', 'custom_bwd']

class autocast(torch.amp.autocast_mode.autocast):
    '''See :class:`torch.autocast`.

    ``torch.cuda.amp.autocast(args...)`` is deprecated. Please use ``torch.amp.autocast("cuda", args...)`` instead.
    '''
    _enabled: Incomplete
    device: str
    fast_dtype: Incomplete
    def __init__(self, enabled: bool = True, dtype: torch.dtype = ..., cache_enabled: bool = True) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...
    def __call__(self, func): ...

def custom_fwd(fwd=None, *, cast_inputs=None):
    """
    ``torch.cuda.amp.custom_fwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_fwd(args..., device_type='cuda')`` instead.
    """
def custom_bwd(bwd):
    """
    ``torch.cuda.amp.custom_bwd(args...)`` is deprecated. Please use
    ``torch.amp.custom_bwd(args..., device_type='cuda')`` instead.
    """
