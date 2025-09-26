import torch
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel.distributed import DistributedDataParallel
from typing import Any, Callable

__all__ = ['hook_with_zero_step', 'hook_with_zero_step_interleaved']

def hook_with_zero_step(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool = False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

    This approach overlaps the optimizer computation and communication with the
    backward communication. In particular, the backward computation proceeds
    contiguously, and the optimizer computation follows, overlapping with
    outstanding backward communication (i.e. all-reduces) and possibly other
    optimizer communication (i.e. broadcasts).
    The optimizer step computation begins after the last gradient bucket computation has finished.

    This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
    if communication is relatively slow compared to computation.

    Arguments:
        hook (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
            to modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL/HCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
def hook_with_zero_step_interleaved(hook: Callable[[Any, dist.GradBucket], torch.futures.Future], ddp: DistributedDataParallel, zero: ZeroRedundancyOptimizer, shard_buckets: bool = False) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Modify ``hook`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass

    This approach overlaps the optimizer computation and communication with the
    backward computation and communication. In particular, once a bucket's
    gradients have been computed, the optimizer computation using those
    gradients is launched (though the actual computation must wait for the
    bucket's all-reduce to complete). This yields an interleaving of all-
    reduces and broadcasts in the communication stream.

    This approach may be preferred over :meth:`hook_with_zero_step` if
    communication is relatively fast compared to computation.

    Arguments:
        hook (Any * dist.GradBucket -> torch.futures.Future): the hook to
            modify.
        ddp (DistributedDataParallel): the :class:`DistributedDataParallel`
            instance to use.
        zero (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
            instance to use.
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity; if
            ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank).

    Returns:
        The modified hook.

    Raises:
        ValueError: if ``zero`` was constructed with ``overlap_with_ddp=False``.
        RuntimeError: if using any backend other than NCCL since currently
            Gloo may hang.

    .. warning::
        Given the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``.
    """
