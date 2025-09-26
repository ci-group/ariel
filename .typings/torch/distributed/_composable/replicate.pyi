import torch
import torch.nn as nn
import weakref
from .contract import _get_registry as _get_registry, contract as contract
from _typeshed import Incomplete
from collections.abc import Iterable
from torch.distributed._composable_state import _State as _State
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from typing import Any, NoReturn

_ROOT_MODULE_PREFIX: str

class _ReplicateState(_State):
    _ddp_weakref: weakref.ref
    module: nn.Module
    has_initialized: bool
    _param_list: nn.ParameterList
    _orig_module: Incomplete
    _param_names: list[str]
    _no_sync: bool
    _init_args: tuple[Any, ...] | None
    _init_kwargs: dict[str, Any]
    _comm_hook_args: list[Any]
    def __init__(self) -> None: ...
    def _collect_params(self, module: nn.Module, ignored_modules: set[nn.Module], ignored_params: set[nn.Parameter], prefix: str = ...) -> None: ...
    def lazy_init(self) -> None: ...
    _ddp: Incomplete
    def init(self, module: nn.Module, ignored_modules: set[nn.Module], **kwargs) -> None: ...
    def register_comm_hook(self) -> None: ...
    def record_init_args(self, *args, **kwargs) -> None: ...
    def forward_pre_hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...
    def forward_post_hook(self, module: nn.Module, input: tuple[torch.Tensor], output: torch.Tensor) -> torch.Tensor: ...

def unimplemented_deepcopy(*args: Any, **kwargs: Any) -> NoReturn: ...

class DDP:
    def __new__(cls, *args, **kwargs):
        """
        Override ``__new__`` to remove the DDP class and directly construct
        the original class for cases like indexing into a container module.
        """
    def set_requires_gradient_sync(self, requires_gradient_sync: bool) -> None:
        """
        Sets if the module should sync gradients. This can be used to implement
        gradient accumulation without communication.

        Args:
            requires_gradient_sync (bool): Whether to reduce gradients for the
                module's parameters.
        """
    def register_comm_hook(self, *args, **kwargs) -> None: ...

def replicate(module: nn.Module, ignored_modules: Iterable[torch.nn.Module] | None = None, **kwargs) -> nn.Module:
    """Replicates a module

    Args:
        module (torch.nn.Module): module to replicate

    Example::
        >>> # xdoctest: +REQUIRES(module:torch._C._distributed_c10d)
        >>> module = nn.Linear(3, 3)
        >>> replicate(module)
    """
def _is_fully_sharded(module: nn.Module) -> bool:
    """Check if module is marked with fully_shard."""
