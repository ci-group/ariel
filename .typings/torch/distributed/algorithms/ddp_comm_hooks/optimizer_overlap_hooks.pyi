import torch
import torch.distributed as dist
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.autograd import Variable as Variable
from typing import Any, Callable, no_type_check

__all__: list[str]
_FUNCTIONAL_OPTIM_STEP_METHOD_NAME: str

class _OptimizerHookState:
    """
    Holds state for running optimizer in-line after DDP communication hook.

    Currently contains only optimizer class which must have a method `step_param`.
    """
    __slots__: Incomplete
    functional_optimizer: Incomplete
    def __init__(self, functional_optim, params=None) -> None: ...
    params_to_optimize: Incomplete
    def _set_params_to_optimize(self, params) -> None: ...
    def _check_valid_functional_optim(self) -> None: ...

@dataclass
class _OptimInBackwardHookState:
    optim_stream: torch.Stream
    wait_for_optim_stream_enqueued: bool

@no_type_check
def _apply_optim_in_backward_hook(gradient_is_bucket_view):
    """
    Register hook to apply the optimizer in backward.

    If torch.distributed.optim._apply_optimizer_in_backward is used to overlap
    optimizer with backward pass, DDP will run the below hook to run optimizer
    step for parameters after gradient communication has taken place.
    """
def _hook_then_optimizer(hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]], optimizer_state: _OptimizerHookState) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """Run optimizer in a functional fashion after DDP communication hook."""
