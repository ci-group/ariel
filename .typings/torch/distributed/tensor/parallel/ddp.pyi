import torch.nn as nn
from _typeshed import Incomplete
from torch.distributed.tensor.parallel._data_parallel_utils import _flatten_tensor as _flatten_tensor, _unflatten_tensor as _unflatten_tensor
from typing import Any

__all__: Incomplete

def _get_submodule_n_params(module: nn.Module, path: str):
    """
    Get submodule and the direct path of parameter from the module
    """
def _update_module_param(param_list: list[tuple[nn.Module, str, nn.Parameter]]):
    """
    Update parameters within the module
    """
def _reconstruct_dtensor(module: nn.Module, _input: Any):
    """
    Recontruct DTensor parameters from local tensors
    """
def _localize_dtensor(module: nn.Module, *_: Any, ignored_params: set[nn.Parameter] | None = None):
    """
    Convert DTensor parameters to local tensors
    """
def _pre_dp_module_transform(module: nn.Module):
    '''
    Enable the composability between Tensor Parallelism (TP) and Data
    Parallelism(DP) in PyTorch when using DDP. We need to convert Parameters which
    are DTensors to local tensors before wrapping with data parallelism API.
    We then register two hooks, one for converting local tensors back to DTensor
    preforward and one to convert DTensors back to tensors after Forward. By
    integrating this way, we avoid any special handling of DTensor parameters by DDP
    and get DTensor\'s gradients propagated back to DP, e.g. gradient buckets of DDP.

    For now, this API only works with ``DistributedDataParallel``. It will later support
    other DP methods such as FSDP.

    Args:
        module (:class:`nn.Module`):
            Module which has been applied TP on.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, PairwiseParallel
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.tensor.parallel.ddp import pre_dp_module_transform
        >>>
        >>> # Define the module.
        >>> m = module(...)
        >>> parallelize_module(m, PairwiseParallel())
        >>> m = pre_dp_module_transform(m)
        >>> m = DDP(m)
        >>>
    '''
