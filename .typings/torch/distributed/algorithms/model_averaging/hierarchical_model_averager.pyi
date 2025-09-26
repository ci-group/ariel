import torch
import torch.distributed.algorithms.model_averaging.averagers as averagers
from _typeshed import Incomplete
from collections.abc import Iterable

logger: Incomplete

class HierarchicalModelAverager(averagers.ModelAverager):
    '''
    Runs hierarchical model averaging (`hierarchical SGD <https://arxiv.org/pdf/2010.12998.pdf>`_).

    Process groups of different sizes are organized in a hierarchy, and they average parameters
    by using different periods concurrently after the warm-up stage.
    This is an extension of :class:`~torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager`
    that supports `post-local SGD <https://arxiv.org/abs/1808.07217>`_, which essentially only supports
    a two-level hierarchy: the intra-machine level and the global level, where the intra-machine
    level is usually embedded in :meth:`~torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook`.
    Similarly, the process groups within this class do not have such an intra-machine process
    subgroup, which should be embedded by the post-local SGD communication hook instead.

    Args:
        period_group_size_dict: An ordered dict mapping keys of model averaging period to
                                process group size, used for initializing process groups of
                                different sizes in a hierarchy to average parameters concurrently.
                                Particularly, at each iteration, there will be at most a single
                                process group that runs averaging -- the period of such group should
                                have the largest period which the current step can be divided by.
                                For example, if the dict has three keys: 2, 4, and 8,
                                then this means totally three process groups will be created to
                                average parameters every 2, 4, and 8 iterations, respectively.
                                At the 4th iteration, only the second process group will run
                                averaging, because the first process group should be a
                                subset of the second process group, and no need to execute the first
                                process group redundantly.
                                On the other hand, the third process group can only be triggered
                                every 8 iterations, so it will not be triggered at the 4th iteration.
        warmup_steps (int): The number of warm-up steps. During this stage, model averaging is skipped.
        process_group (ProcessGroup, optional): The overall process group containing all the processes that runs model averaging.
                                                If ``None``, the default process group, which is created
                                                by :func:`torch.distributed.init_process_group`, will be used.
                                                (default: ``None``)

    Example::
        >>> # xdoctest: +SKIP(\'undefined rank\')
        >>> from collections import OrderedDict
        >>> import torch
        >>> import torch.distributed as dist
        >>> from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (
        >>>     PostLocalSGDState,
        >>>     post_localSGD_hook,
        >>> )
        >>> import torch.distributed.algorithms.model_averaging.hierarchical_model_averager as hierarchicalSGD
        >>> import torch.nn as nn
        >>>
        >>> dist.init_process_group("nccl", rank=rank, world_size=16)
        >>> torch.cuda.set_device(rank)
        >>> module = nn.Linear(1, 1, bias=False).to(rank)
        >>> model = nn.parallel.DistributedDataParallel(
        >>>    module, device_ids=[rank], output_device=rank
        >>> )
        >>> # Register a post-localSGD communication hook.
        >>> # Assume that each machine has 4 GPUs, then each intra-machine subgroup has a size of 4.
        >>> subgroup, _ = dist.new_subgroups()
        >>> state = PostLocalSGDState(process_group=None, subgroup=subgroup, start_localSGD_iter=100)
        >>> model.register_comm_hook(state, post_localSGD_hook)
        >>>
        >>> # Average parameters among each group of 8 processes every 4 iterations, and among all
        >>> # the 16 processes every 16 iterations.
        >>> averager = hierarchicalSGD.HierarchicalModelAverager(
        >>>     period_group_size_dict=OrderedDict([(4, 8), (16, 16)]), warmup_steps=100)
        >>> # Note that ``warmup_steps`` must be the same as ``start_localSGD_iter`` used in ``PostLocalSGDState``.
        >>> # In the first 100 steps, run global gradient averaging like normal DDP at every step.
        >>> # After 100 steps, run model averaging at two levels.
        >>> for step in range(0, 200):
        >>>    optimizer.zero_grad()
        >>>    loss = loss_fn(output, labels)
        >>>    loss.backward()
        >>>    optimizer.step()
        >>>    # Average parameters after ``optimizer.step()``.
        >>>    # Thus, the inter-node communication only occurs periodically after ``warmup_steps``.
        >>>    averager.average_parameters(model.parameters())

    .. warning ::
        The last group size in the dict must be the size of the provided ``process_group``,
        which indicates model averaging at the highest level of the hierarchy.
        If ``process_group`` is not provided, then the last group size should be equal to the world size.

    .. warning ::
        `HierarchicalModelAverager` is experimental and subject to change.
    '''
    _periods: Incomplete
    period_process_group_dict: Incomplete
    warmup_steps: Incomplete
    def __init__(self, period_group_size_dict=None, warmup_steps: int = 0, process_group=None) -> None: ...
    def _find_process_group(self):
        """
        Return a process group as the value of an ``period_process_group_dict`` entry.

        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        then the returned process group is the one corresponding to the largest period,
        since this process group will be used for averaging parameters at this ``step``.
        Returns ``None`` if not found.
        """
    def average_parameters(self, params: Iterable[torch.nn.Parameter] | Iterable[dict[str, torch.nn.Parameter]]):
        """
        Averages parameters or parameter groups of an optimizer.

        Averaging only occurs if ``step`` is no less than ``warmup_steps``
        and it can be divided by a period in the keys of ``period_process_group_dict``,
        where ``step`` is increased by 1 at each iteration in the training loop.
        If ``step`` can be divided by multiple periods in the keys of ``period_process_group_dict``,
        only the largest period is used, and the corresponding process group is used for averaging parameters.
        Args:
            params: The parameters of a model or parameter groups of an optimizer.
        """
