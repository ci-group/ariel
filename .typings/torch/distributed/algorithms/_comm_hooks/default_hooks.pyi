import torch
import torch.distributed as dist
from _typeshed import Incomplete

class DefaultState:
    """
    Stores state needed to perform the default communication algorithm within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """
    __slots__: Incomplete
    process_group: Incomplete
    world_size: Incomplete
    gradient_predivide_factor: Incomplete
    gradient_postdivide_factor: Incomplete
    def __init__(self, process_group: dist.ProcessGroup) -> None: ...
    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float: ...

class LowPrecisionState(DefaultState):
    """
    Stores state needed to perform gradient communication in a lower precision within a communication hook.

    Communication hook will cast gradients back to the original
    parameter precision specified by ``parameter_type`` (default: torch.float32).
    Builds on top of the :class:`DefaultState`.

    Args:
        parameter_type (torch.dtype): The precision of model's parameters.
        Required for a hook to cast gradients back to a parameter's precision.
    """
    __slots__: Incomplete
    parameter_type: Incomplete
    def __init__(self, process_group, parameter_type=...) -> None: ...

def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    """
    Casts gradients back to full parameter precision so that further computation happens in full precision.
    """
def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    """
    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    """
    Implement the  FSDP communication hook for ``reduce_scatter`` algorithm.

    For sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be
        communicated across ranks.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
def _low_precision_hook(prec: torch.dtype, state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None): ...
def fp16_compress_hook(state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None):
    """
    Implement FSDP communication hook for a simple gradient compression approach.
    Casts ``grad`` to half-precision floating-point format (``torch.float16``).

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
def bf16_compress_hook(state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None):
    """
    Implement FSDP communication hook for a simple gradient compression approach .
    Casts ``grad`` to half-precision floating-point format.

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
