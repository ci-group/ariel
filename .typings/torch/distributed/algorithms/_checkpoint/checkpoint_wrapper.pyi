import abc
import torch
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from torch.autograd.graph import save_on_cpu as save_on_cpu
from torch.distributed.utils import _pack_kwargs as _pack_kwargs, _replace_by_prefix as _replace_by_prefix, _unpack_kwargs as _unpack_kwargs
from typing import Any, Callable

_CHECKPOINT_WRAPPED_MODULE: str
_CHECKPOINT_PREFIX: Incomplete

class CheckpointImpl(Enum):
    REENTRANT = ...
    NO_REENTRANT = ...

class ActivationWrapper(torch.nn.Module, ABC, metaclass=abc.ABCMeta):
    """
    Base class for Activation Checkpoint and Activation Offload.

    Not meant to be instantiated directly.
    """
    _checkpoint_wrapped_module: Incomplete
    def __init__(self, mod) -> None: ...
    @abstractmethod
    def forward(self, *args, **kwargs): ...
    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to wrapped module."""
    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is a nn.Sequential."""
    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """
        Override :meth:`named_parameters()` to intercept parameter names.

        remove all occurrences of ``_CHECKPOINT_PREFIX``.
        """
    @staticmethod
    def _post_state_dict_hook(module: nn.Module, state_dict: dict[str, Any], prefix: str, *args: Any) -> dict[str, Any]:
        """
        _post_state_dict_hook() is called after the state_dict() of this FSDP module is executed.

        For ``checkpoint_wrapper``, it will strip checkpoint-wrapped module prefix,
        so that this module can be loaded into non-checkpointed modules.
        It would still be able to be loaded into checkpoint-wrapped modules as this class,
        adds the prefix back before loading the state_dict.
        """
    @staticmethod
    def _pre_load_state_dict_hook(module: nn.Module, state_dict: dict[str, Any], prefix: str, *args: Any) -> None:
        """
        ``_pre_state_dict_hook` is called before ``self._load_from_state_dict()`` is called.

        For ``checkpoint_wrapper``, it will add back the module
        prefix so that non-checkpointed modules can be loaded into
        checkpoint_wrapper modules properly.
        """

class OffloadWrapper(ActivationWrapper):
    def __init__(self, mod) -> None: ...
    def forward(self, *args, **kwargs): ...

class CheckpointWrapper(ActivationWrapper):
    """
    An ``nn.Module`` that wraps another ``nn.Module`` with checkpointing.

    Note that this module is not meant to be used directly but instead,
    it is to be used through the ``checkpoint_wrapper`` function.
    """
    checkpoint_impl: Incomplete
    checkpoint_fn: Incomplete
    def __init__(self, mod: torch.nn.Module, checkpoint_impl: CheckpointImpl = ..., checkpoint_fn=None, **checkpoint_fn_kwargs) -> None: ...
    def forward(self, *args, **kwargs): ...

def offload_wrapper(module: torch.nn.Module) -> torch.nn.Module:
    """
    Wrap a module for activation offloading to CPU.

    Offloads intermediate activations to the CPU for modules wrapped with this function.
    Wrappers with activation offload can be composed with ones that do recomputation-based
    checkpoint to trade off increased compute versus increased CPU
    memory usage and additional H2D transfers.

    Usage::
        offloaded_module = offload_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
    Returns:
        (nn.Module):
            Wrapped module
    """
def checkpoint_wrapper(module: torch.nn.Module, checkpoint_impl: CheckpointImpl = ..., checkpoint_fn=None, **checkpoint_fn_kwargs) -> torch.nn.Module:
    """
    Wrap a module for activation checkpointing.

    If the module is wrapped with this function, all subsequent calls to the module will,
    automatically perform checkpointing without the user having to explicitly call ``checkpoint`` function.

    Usage::
        checkpointed_module = checkpoint_wrapper(module)
        outputs = checkpointed_module(inputs)
    Args:
        module (nn.Module):
            The module to be wrapped
        checkpoint_impl (Optional[CheckpointImpl]):
            The checkpointing implementation to use. Note that this will only
            be passed into the ``torch.utils.checkpoint.checkpoint``
            implementation, and is ignored if a custom ``checkpoint_fn`` is
            specified. Note that for implementations using reentrant checkpoint
            from ``torch.utils.checkpoint``, keyword arguments will only be
            supported if ``checkpoint_impl`` is passed as ``CheckpointImpl.REENTRANT`.
        checkpoint_fn (Optional[Callable]):
            Functional checkpoint implementation to use. If this is specified,
            it will be used over the default ``torch.utils.checkpoint.checkpoint``
            implementation and the `checkpoint_impl` argument will be ignored.
        **checkpoint_fn_kwargs: (Dict[str, Any]): Keyword arguments to pass into `checkpoint_fn`.

    Returns:
        (nn.Module):
            Wrapped module
    """
def apply_activation_checkpointing(model, checkpoint_wrapper_fn=..., check_fn=..., auto_wrap_policy: Callable[[nn.Module, bool, int], bool] | None = None):
    """
    Apply :func:`checkpoint_wrapper` to modules within `model` based on a user-defined configuration.

    For each module within `model`, the `check_fn` is used to decide
    whether `module` should be wrapped with :func:`checkpoint_wrapper` or not.

    Note::
        This function modifies `model` in place and replaces appropriate layers with
        their checkpoint-wrapped modules.
    Note::
        This function will not wrap the overall root module. If this is needed, please directly use
        :func:`checkpoint_wrapper` or :func:`offload_wrapper`.
    Usage::
        model = nn.Sequential(
            nn.Linear(10, 10), nn.Linear(10, 10), nn.Linear(10, 10)
        )
        check_fn = lambda l: isinstance(l, nn.Linear)
        # checkpoint activations
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
        # Or offload activations to CPU
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=offload_wrapper, check_fn=check_fn)
    Args:
        model (nn.Module):
            The model whose submodules should be wrapped with activation checkpointing.
        checkpoint_wrapper_fn (Optional[Callable[nn.Module]])
            A ``Callable`` which will wrap modules
        check_fn (Optional[Callable[nn.Module, nn.Module]])
            A lambda function which will be passed each child submodule of ``model`` and returns
            ``True`` or ``False`` depending on whether the submodule should be wrapped.
        auto_wrap_policy (Optional[Callable[[nn.Module, bool, int], bool]]): A policy to wrap model's
            submodules with AC. Note that if this is specified, it takes precedence over ``check_fn``.
    Returns: None (`model` is modified inplace)
    """
