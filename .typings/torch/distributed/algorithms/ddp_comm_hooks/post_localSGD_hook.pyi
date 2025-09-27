import torch
import torch.distributed as dist
from _typeshed import Incomplete

logger: Incomplete

class PostLocalSGDState:
    """
    Store state for all-reducing gradients globally until given step, then locally after.

    Stores the state for all-reducing gradients globally using ``process_group`` until step ``start_localSGD_iter``,
    and all-reducing gradients locally using ``subgroup`` afterwards.

    If ``process_group`` is ``None``, the global process group will be used.
    If ``subgroup`` is ``None``, the intra-node process group on each machine will be used.

    Additionally, ``post_local_gradient_allreduce`` may be worth tuning,
    because both true and false may give a faster convergence.
    """
    __slots__: Incomplete
    process_group: Incomplete
    subgroup: Incomplete
    start_localSGD_iter: Incomplete
    post_local_gradient_allreduce: Incomplete
    iter: int
    def __init__(self, process_group, subgroup, start_localSGD_iter, post_local_gradient_allreduce: bool = True) -> None:
        """Initialize state object with given parameters and log when localSGD start."""
    def maybe_increase_iter(self, bucket) -> None:
        """Track iterations and trigger log message at start of local SGD."""

def post_localSGD_hook(state: PostLocalSGDState, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Run post-localSGD algorithm.

    This DDP communication hook is used for running post-localSGD algorithm,
    by combining with a model averaging component (e.g.,
    :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`)
    that runs after the optimizer step.

    Args:
        state (PostLocalSGDState): State information to run post-localSGD.
            Users mainly need to tune ``start_localSGD_iter`` to determine when to start local SGD.
        bucket (dist.GradBucket): Bucket that stores a 1D flattened gradient tensor that batches multiple per-variable tensors.
            Note that since DDP comm hook only supports single process single device mode,
            only exactly one tensor is stored in this bucket.

    Returns:
        Future handler of the communication, which updates the gradients in place.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PostLocalSGDState(process_group=process_group, subgroup=subgroup,
                                  start_localSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, post_localSGD_hook)
        >>> # Also need to establish a model averaging module and run model averaging after ``optimizer.step()``.
        >>> # Please refer to the examples in ``torch.distributed.algorithms.model_averaging.averagers`` module.
    """
