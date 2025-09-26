import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook as allreduce_hook
from torch.distributed.algorithms.ddp_comm_hooks.optimizer_overlap_hooks import _OptimizerHookState as _OptimizerHookState, _hook_then_optimizer as _hook_then_optimizer
from torch.distributed.fsdp import FullyShardedDataParallel as FullyShardedDataParallel
from torch.distributed.optim import as_functional_optim as as_functional_optim
from torch.nn.parallel import DistributedDataParallel as DistributedDataParallel
from torch.optim import Optimizer as Optimizer

_registered_overlapped_optims: dict[type, type]

def register_overlapped(optim_cls): ...

class OverlappedOptimizer(ABC, metaclass=abc.ABCMeta):
    optim_cls: Incomplete
    def __init__(self, optim_cls: type) -> None:
        """
        Initialize the OverlappedOptimizer.

        Overlappedoptimizer is a base class that child classes can implement to
        specify how different optimizers will register themselves with DDP.
        """
    @abstractmethod
    def register_ddp(self, ddp: DistributedDataParallel) -> None:
        """Registers the overlapped optimizer with DDP."""
    @abstractmethod
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Registers the overlapped optimizer with FSDP."""

class _OverlappedStandardOptimizer(OverlappedOptimizer):
    """Overlaps a regular ``Optimizer``."""
    _opt_hook_state: Incomplete
    def __init__(self, optim_cls: type, params, *optim_args, **optim_kwargs) -> None: ...
    def register_ddp(self, ddp_inst: DistributedDataParallel): ...
    def register_fsdp(self, fsdp: FullyShardedDataParallel) -> None:
        """Register the overlapped optimizer with FSDP."""

def _as_overlapped_optim(optim_cls: type, params, *args, **kwargs):
    """Return a new ``OverlappedOptimizer`` instance that supports ``optim_cls``."""
