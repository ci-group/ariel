import torch
import torch.distributed as dist
from _typeshed import Incomplete
from collections import OrderedDict
from collections.abc import Container
from torch import nn as nn
from torch.nn.utils.rnn import PackedSequence as PackedSequence
from typing import Any, Callable, TypeVar, overload

__all__: Incomplete

def _pack_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    '''
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite).

    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        gives both positional args and kwarg values, where the positional args
        proceed kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element\'s length is at most the first tuple element\'s length.
    '''
def _cast_forward_inputs(dtype: torch.dtype | None, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
def _unpack_kwargs(flat_args: tuple[Any, ...], kwarg_keys: tuple[str, ...]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """See _pack_kwargs."""
S = TypeVar('S', dict, list, tuple)
T = TypeVar('T', torch.Tensor, PackedSequence)

@overload
def _recursive_to(inputs: S, target_device: torch.device, use_side_stream_for_tensor_copies: bool) -> list[S]: ...
@overload
def _recursive_to(inputs: T, target_device: torch.device, use_side_stream_for_tensor_copies: bool) -> tuple[T]: ...
def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
Q = TypeVar('Q')
R = TypeVar('R', dict, list, tuple, set, OrderedDict, PackedSequence, Any)

@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Q], container: torch.Tensor) -> Q: ...
@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R: ...
def _to_kwargs(inputs: tuple[Any, ...], kwargs: dict[str, Any] | None, target_device: torch.device, use_side_stream_for_tensor_copies: bool) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]: ...
def _verify_param_shape_across_processes(process_group: dist.ProcessGroup, tensors: list[torch.Tensor], logger: dist.Logger | None = None): ...
def _sync_module_states(module: nn.Module, process_group: dist.ProcessGroup, broadcast_bucket_size: int, src: int, params_and_buffers_to_ignore: Container[str], broadcast_buffers: bool = True) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
def _sync_params_and_buffers(process_group: dist.ProcessGroup, module_states: list[torch.Tensor], broadcast_bucket_size: int, src: int) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
def _replace_by_prefix(state_dict: dict[str, Any], old_prefix: str, new_prefix: str) -> None:
    '''
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    '''
def _data_ptr_allocated(tensor: torch.Tensor) -> bool: ...
def _get_root_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """
    Returns the modules in ``modules`` that are root modules (i.e.
    parent-less) with respect to the set ``modules``. In other words, these
    are the modules in ``modules`` that are the not child of any other
    module in ``modules``.
    """
