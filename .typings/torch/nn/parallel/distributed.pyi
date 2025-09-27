import torch
import torch.distributed as dist
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from torch.autograd import Function
from torch.distributed.algorithms.join import JoinHook, Joinable
from torch.nn.modules import Module
from torch.utils.hooks import RemovableHandle
from typing import Any, Callable

__all__ = ['DistributedDataParallel']

@dataclass
class _MixedPrecision:
    """
    This configures DDP-native mixed precision training.

    Attributes:
        param_dtype (torch.dtype): This specifies the dtype for model
            parameters, inputs (when ``cast_forward_inputs`` is set to
            ``True``), and therefore the dtype for computation.
            However, outside the forward and backward passes, parameters are in
            full precision. Model checkpointing always happens in full
            precision.
        reduce_dtype (torch.dtype): This specifies the dtype for gradient
            reduction, which is permitted to differ from ``param_dtype``.
        buffer_dtype (torch.dtype): This specifies the dtype for buffers.

    .. note:: This API is experimental and subject to change.

    .. note:: Only floating point tensors are cast to their specified dtypes.

    .. note:: ``state_dict`` checkpoints parameters and buffers in full
        precision.

    .. note:: Each low precision dtype must be specified explicitly. For
        example, ``_MixedPrecision(reduce_dtype=torch.float16)`` only specifies
        the reduction dtype to be low precision, and DDP will not cast
        parameters or buffers.

    .. note:: If a ``reduce_dtype`` is not specified, then gradient reduction
        happens in ``param_dtype`` if specified or the original parameter dtype
        otherwise. For example, ``_MixedPrecision(param_dtype=torch.float16)``
        would result in communication occurring in fp16.
    """
    param_dtype: torch.dtype | None = ...
    reduce_dtype: torch.dtype | None = ...
    buffer_dtype: torch.dtype | None = ...

class _BufferCommHookLocation(Enum):
    PRE_FORWARD = ...
    POST_FORWARD = ...

@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation

class _DDPSink(Function):
    @staticmethod
    def forward(ctx, ddp_weakref, *inputs): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class _DDPJoinHook(JoinHook):
    ddp: Incomplete
    def __init__(self, ddp, divide_by_initial_world_size) -> None:
        """Set config variables for internal usage."""
    def main_hook(self) -> None:
        """Shadow the DDP collective communication operations in the forward and backward passes."""
    def post_hook(self, is_last_joiner: bool):
        """Sync the final model to ensure that the model is the same across all processes."""

class DistributedDataParallel(Module, Joinable):
    '''Implement distributed data parallelism based on ``torch.distributed`` at module level.

    This container provides data parallelism by synchronizing gradients
    across each model replica. The devices to synchronize across are
    specified by the input ``process_group``, which is the entire world
    by default. Note that ``DistributedDataParallel`` does not chunk or
    otherwise shard the input across participating GPUs; the user is
    responsible for defining how to do so, for example through the use
    of a :class:`DistributedSampler`.

    See also: :ref:`distributed-basics` and :ref:`cuda-nn-ddp-instead`.
    The same constraints on input as in :class:`torch.nn.DataParallel` apply.

    Creation of this class requires that ``torch.distributed`` to be already
    initialized, by calling :func:`torch.distributed.init_process_group`.

    ``DistributedDataParallel`` is proven to be significantly faster than
    :class:`torch.nn.DataParallel` for single-node multi-GPU data
    parallel training.

    To use ``DistributedDataParallel`` on a host with N GPUs, you should spawn
    up ``N`` processes, ensuring that each process exclusively works on a single
    GPU from 0 to N-1. This can be done by either setting
    ``CUDA_VISIBLE_DEVICES`` for every process or by calling:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.cuda.set_device(i)

    where i is from 0 to N-1. In each process, you should refer the following
    to construct this module:

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(
        >>>     backend=\'nccl\', world_size=N, init_method=\'...\'
        >>> )
        >>> model = DistributedDataParallel(model, device_ids=[i], output_device=i)

    In order to spawn up multiple processes per node, you can use either
    ``torch.distributed.launch`` or ``torch.multiprocessing.spawn``.

    .. note::
        Please refer to `PyTorch Distributed Overview <https://pytorch.org/tutorials/beginner/dist_overview.html>`__
        for a brief introduction to all features related to distributed training.

    .. note::
        ``DistributedDataParallel`` can be used in conjunction with
        :class:`torch.distributed.optim.ZeroRedundancyOptimizer` to reduce
        per-rank optimizer states memory footprint. Please refer to
        `ZeroRedundancyOptimizer recipe <https://pytorch.org/tutorials/recipes/zero_redundancy_optimizer.html>`__
        for more details.

    .. note:: ``nccl`` backend is currently the fastest and highly recommended
        backend when using GPUs. This applies to both single-node and
        multi-node distributed training.

    .. note:: This module also supports mixed-precision distributed training.
        This means that your model can have different types of parameters such
        as mixed types of ``fp16`` and ``fp32``, the gradient reduction on these
        mixed types of parameters will just work fine.

    .. note:: If you use ``torch.save`` on one process to checkpoint the module,
        and ``torch.load`` on some other processes to recover it, make sure that
        ``map_location`` is configured properly for every process. Without
        ``map_location``, ``torch.load`` would recover the module to devices
        where the module was saved from.

    .. note:: When a model is trained on ``M`` nodes with ``batch=N``, the
        gradient will be ``M`` times smaller when compared to the same model
        trained on a single node with ``batch=M*N`` if the loss is summed (NOT
        averaged as usual) across instances in a batch (because the gradients
        between different nodes are averaged). You should take this into
        consideration when you want to obtain a mathematically equivalent
        training process compared to the local training counterpart. But in most
        cases, you can just treat a DistributedDataParallel wrapped model, a
        DataParallel wrapped model and an ordinary model on a single GPU as the
        same (E.g. using the same learning rate for equivalent batch size).

    .. note::
        Parameters are never broadcast between processes. The module performs
        an all-reduce step on gradients and assumes that they will be modified
        by the optimizer in all processes in the same way. Buffers
        (e.g. BatchNorm stats) are broadcast from the module in process of rank
        0, to all other replicas in the system in every iteration.

    .. note::
        If you are using DistributedDataParallel in conjunction with the
        :ref:`distributed-rpc-framework`, you should always use
        :meth:`torch.distributed.autograd.backward` to compute gradients and
        :class:`torch.distributed.optim.DistributedOptimizer` for optimizing
        parameters.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> import torch.distributed.autograd as dist_autograd
            >>> from torch.nn.parallel import DistributedDataParallel as DDP
            >>> import torch
            >>> from torch import optim
            >>> from torch.distributed.optim import DistributedOptimizer
            >>> import torch.distributed.rpc as rpc
            >>> from torch.distributed.rpc import RRef
            >>>
            >>> t1 = torch.rand((3, 3), requires_grad=True)
            >>> t2 = torch.rand((3, 3), requires_grad=True)
            >>> rref = rpc.remote("worker1", torch.add, args=(t1, t2))
            >>> ddp_model = DDP(my_model)
            >>>
            >>> # Setup optimizer
            >>> optimizer_params = [rref]
            >>> for param in ddp_model.parameters():
            >>>     optimizer_params.append(RRef(param))
            >>>
            >>> dist_optim = DistributedOptimizer(
            >>>     optim.SGD,
            >>>     optimizer_params,
            >>>     lr=0.05,
            >>> )
            >>>
            >>> with dist_autograd.context() as context_id:
            >>>     pred = ddp_model(rref.to_here())
            >>>     loss = loss_func(pred, target)
            >>>     dist_autograd.backward(context_id, [loss])
            >>>     dist_optim.step(context_id)

    .. note::
        DistributedDataParallel currently offers limited support for gradient
        checkpointing with :meth:`torch.utils.checkpoint`.
        If the checkpoint is done with use_reentrant=False (recommended), DDP
        will work as expected without any limitations.
        If, however, the checkpoint is done with use_reentrant=True (the default),
        DDP will work as expected when there are no unused parameters in the model
        and each layer is checkpointed at most once (make sure you are not passing
        `find_unused_parameters=True` to DDP). We currently do not support the
        case where a layer is checkpointed multiple times, or when there unused
        parameters in the checkpointed model.

    .. note::
        To let a non-DDP model load a state dict from a DDP model,
        :meth:`~torch.nn.modules.utils.consume_prefix_in_state_dict_if_present`
        needs to be applied to strip the prefix "module." in the DDP state dict before loading.

    .. warning::
        Constructor, forward method, and differentiation of the output (or a
        function of the output of this module) are distributed synchronization
        points. Take that into account in case different processes might be
        executing different code.

    .. warning::
        This module assumes all parameters are registered in the model by the
        time it is created. No parameters should be added nor removed later.
        Same applies to buffers.

    .. warning::
        This module assumes all parameters are registered in the model of each
        distributed processes are in the same order. The module itself will
        conduct gradient ``allreduce`` following the reverse order of the
        registered parameters of the model. In other words, it is users\'
        responsibility to ensure that each distributed process has the exact
        same model and thus the exact same parameter registration order.

    .. warning::
        This module allows parameters with non-rowmajor-contiguous strides.
        For example, your model may contain some parameters whose
        :class:`torch.memory_format` is ``torch.contiguous_format``
        and others whose format is ``torch.channels_last``.  However,
        corresponding parameters in different processes must have the
        same strides.

    .. warning::
        This module doesn\'t work with :func:`torch.autograd.grad` (i.e. it will
        only work if gradients are to be accumulated in ``.grad`` attributes of
        parameters).

    .. warning::
        If you plan on using this module with a ``nccl`` backend or a ``gloo``
        backend (that uses Infiniband), together with a DataLoader that uses
        multiple workers, please change the multiprocessing start method to
        ``forkserver`` (Python 3 only) or ``spawn``. Unfortunately
        Gloo (that uses Infiniband) and NCCL2 are not fork safe, and you will
        likely experience deadlocks if you don\'t change this setting.

    .. warning::
        You should never try to change your model\'s parameters after wrapping
        up your model with ``DistributedDataParallel``. Because, when
        wrapping up your model with ``DistributedDataParallel``, the constructor
        of ``DistributedDataParallel`` will register the additional gradient
        reduction functions on all the parameters of the model itself at the
        time of construction. If you change the model\'s parameters afterwards,
        gradient reduction functions no longer match the correct set of
        parameters.

    .. warning::
        Using ``DistributedDataParallel`` in conjunction with the
        :ref:`distributed-rpc-framework` is experimental and subject to change.

    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices.
                   1) For single-device modules, ``device_ids`` can
                   contain exactly one device id, which represents the only
                   CUDA device where the input module corresponding to this process resides.
                   Alternatively, ``device_ids`` can also be ``None``.
                   2) For multi-device modules and CPU modules,
                   ``device_ids`` must be ``None``.

                   When ``device_ids`` is ``None`` for both cases,
                   both the input data for the forward pass and the actual module
                   must be placed on the correct device.
                   (default: ``None``)
        output_device (int or torch.device): Device location of output for
                      single-device CUDA modules. For multi-device modules and
                      CPU modules, it must be ``None``, and the module itself
                      dictates the output location. (default: ``device_ids[0]``
                      for single-device modules)
        broadcast_buffers (bool): Flag that enables syncing (broadcasting)
                          buffers of the module at beginning of the ``forward``
                          function. (default: ``True``)
        init_sync (bool): Whether to sync during initialization to verify param
                          shapes and broadcast parameters and buffers.
                          WARNING: if this is set to False the user is required
                          to ensure themselves that the weights are the same on
                          all ranks.
                          (default: ``True``)
        process_group: The process group to be used for distributed data
                       all-reduction. If ``None``, the default process group, which
                       is created by :func:`torch.distributed.init_process_group`,
                       will be used. (default: ``None``)
        bucket_cap_mb: ``DistributedDataParallel`` will bucket parameters into
                       multiple buckets so that gradient reduction of each
                       bucket can potentially overlap with backward computation.
                       :attr:`bucket_cap_mb` controls the bucket size in
                       MebiBytes (MiB). If ``None``, a default size of 25 MiB
                       will be used. (default: ``None``)
        find_unused_parameters (bool): Traverse the autograd graph from all
                               tensors contained in the return value of the
                               wrapped module\'s ``forward`` function. Parameters
                               that don\'t receive gradients as part of this
                               graph are preemptively marked as being ready to
                               be reduced. In addition, parameters that may have
                               been used in the wrapped module\'s ``forward``
                               function but were not part of loss computation and
                               thus would also not receive gradients are
                               preemptively marked as ready to be reduced.
                               (default: ``False``)
        check_reduction: This argument is deprecated.
        gradient_as_bucket_view (bool): When set to ``True``, gradients will be views
                      pointing to different offsets of ``allreduce`` communication
                      buckets. This can reduce peak memory usage, where the
                      saved memory size will be equal to the total gradients
                      size. Moreover, it avoids the overhead of copying between
                      gradients and ``allreduce`` communication buckets. When
                      gradients are views, ``detach_()`` cannot be called on the
                      gradients. If hitting such errors, please fix it by
                      referring to the :meth:`~torch.optim.Optimizer.zero_grad`
                      function in ``torch/optim/optimizer.py`` as a solution.
                      Note that gradients will be views after first iteration, so
                      the peak memory saving should be checked after first iteration.
        static_graph (bool): When set to ``True``, DDP knows the trained graph is
                     static. Static graph means 1) The set of used and unused
                     parameters will not change during the whole training loop; in
                     this case, it does not matter whether users set
                     ``find_unused_parameters = True`` or not. 2) How the graph is trained
                     will not change during the whole training loop (meaning there is
                     no control flow depending on iterations).
                     When static_graph is set to be ``True``, DDP will support cases that
                     can not be supported in the past:
                     1) Reentrant backwards.
                     2) Activation checkpointing multiple times.
                     3) Activation checkpointing when model has unused parameters.
                     4) There are model parameters that are outside of forward function.
                     5) Potentially improve performance when there are unused parameters,
                     as DDP will not search graph in each iteration to detect unused
                     parameters when static_graph is set to be ``True``.
                     To check whether you can set static_graph to be ``True``, one way is to
                     check ddp logging data at the end of your previous model training,
                     if ``ddp_logging_data.get("can_set_static_graph") == True``, mostly you
                     can set ``static_graph = True`` as well.

                     Example::
                         >>> # xdoctest: +SKIP("undefined variables")
                         >>> model_DDP = torch.nn.parallel.DistributedDataParallel(model)
                         >>> # Training loop
                         >>> ...
                         >>> ddp_logging_data = model_DDP._get_ddp_logging_data()
                         >>> static_graph = ddp_logging_data.get("can_set_static_graph")
        delay_all_reduce_named_params (list of tuple of str and torch.nn.Parameter): a list
                    of named parameters whose all reduce will be delayed when the gradient of
                    the parameter specified in ``param_to_hook_all_reduce`` is ready. Other
                    arguments of DDP do not apply to named params specified in this argument
                    as these named params will be ignored by DDP reducer.
        param_to_hook_all_reduce (torch.nn.Parameter): a parameter to hook delayed all reduce
                    of parameters specified in ``delay_all_reduce_named_params``.
        skip_all_reduce_unused_params: When set to True, DDP will skip reducing unused parameters.
                    This requires that unused parameters remain the same across all ranks throughout
                    the entire training process. If this condition is not met, it may cause
                    desynchronization and result in training hang.


    Attributes:
        module (Module): the module to be parallelized.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> torch.distributed.init_process_group(backend=\'nccl\', world_size=4, init_method=\'...\')
        >>> net = torch.nn.parallel.DistributedDataParallel(model)
    '''
    _active_ddp_module: DistributedDataParallel | None
    _use_python_reducer: Incomplete
    logger: dist.Logger | None
    process_group: Incomplete
    device_mesh: Incomplete
    _delay_all_reduce_params: Incomplete
    parameters_to_ignore: Incomplete
    _module_parameters: Incomplete
    is_multi_device_module: Incomplete
    device_type: Incomplete
    device_ids: Incomplete
    output_device: Incomplete
    static_graph: bool
    dim: Incomplete
    module: Incomplete
    device: Incomplete
    broadcast_buffers: Incomplete
    find_unused_parameters: Incomplete
    require_backward_grad_sync: bool
    require_forward_param_sync: bool
    gradient_as_bucket_view: Incomplete
    mixed_precision: Incomplete
    broadcast_bucket_size: Incomplete
    bucket_bytes_cap_default: bool
    bucket_bytes_cap: Incomplete
    use_side_stream_for_tensor_copies: Incomplete
    _delay_grad_buffer: torch.Tensor | None
    _delay_grad_views: list[torch.Tensor]
    _delay_all_reduce_all_params: bool
    skip_all_reduce_unused_params: Incomplete
    _comm_hooks: list[tuple[Callable, object]]
    _mp_stream: Incomplete
    _submodule_to_event: Incomplete
    _has_rebuilt_buckets: bool
    _lazy_init_ran: bool
    _accum_grad_hooks: list[RemovableHandle]
    _ddp_sink_clone: bool
    def __init__(self, module, device_ids=None, output_device=None, dim: int = 0, broadcast_buffers: bool = True, init_sync: bool = True, process_group=None, bucket_cap_mb=None, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False, delay_all_reduce_named_params=None, param_to_hook_all_reduce=None, mixed_precision: _MixedPrecision | None = None, device_mesh=None, skip_all_reduce_unused_params: bool = False) -> None: ...
    def _register_accum_grad_hook(self) -> None: ...
    def _delayed_all_reduce_hook(self, grad): ...
    def _register_delay_all_reduce_hook(self, bucket_cap_mb, param_to_hook_all_reduce, device_ids) -> None: ...
    def _setup_in_backward_optimizers(self) -> None: ...
    def _fire_reducer_autograd_hook(self, idx, *unused) -> None:
        """
        Fire the reducer's autograd hook to allreduce params in a Reducer bucket.

        Note that this is only used during mixed precision training as the
        Reducer's hooks installed during construction time would not be called
        as we're working in the low precision parameter setting.
        """
    def _root_copy_hook(self, *args: Any, **kwargs: Any) -> None:
        """
        For DDP mixed precision, put low precision copies on separate stream and create events to wait for them.

        When training with DDP mixed precision, this root pre-forward hook kicks
        off low precision copies on a separate stream and creates respective
        events to wait for them.
        """
    def _module_wait_for_copy_hook(self, module, *args: Any, **kwargs: Any) -> None:
        """Before carrying out computation, wait on the appropriate event to ensure low precision copies have finished."""
    def _log_and_throw(self, err_type, err_msg) -> None: ...
    reducer: Incomplete
    def _ddp_init_helper(self, parameters, expect_sparse_gradient, param_to_name_mapping, static_graph) -> None:
        """
        DDP init helper function to manage parameters, grad hooks, logging, and SyncBatchNorm.

        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def _build_params_for_reducer(self): ...
    modules_buffers: Incomplete
    named_module_buffers: Incomplete
    def _assign_modules_buffers(self) -> None:
        """
        Assign self.module.named_buffers to self.modules_buffers.

        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
    def _build_debug_param_to_name_mapping(self, parameters): ...
    def _get_parameters(self, m, recurse: bool = True) -> Generator[Incomplete, Incomplete]:
        """Return a generator of module parameters."""
    def _check_default_group(self) -> None: ...
    @contextmanager
    def no_sync(self) -> Generator[None]:
        '''
        Context manager to disable gradient synchronizations across DDP processes.

        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>     for input in inputs:
            >>>         ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads

        .. warning::
            The forward pass should be included inside the context manager, or
            else gradients will still be synchronized.
        '''
    @classmethod
    def _get_active_ddp_module(cls):
        """`TorchDynamo` requires DDP's status and module for cooperative optimization."""
    @contextmanager
    def _inside_ddp_forward(self) -> Generator[None]: ...
    def _run_ddp_forward(self, *inputs, **kwargs): ...
    def _clear_grad_buffer(self) -> None: ...
    def _lazy_init(self) -> None: ...
    def _pre_forward(self, *inputs, **kwargs): ...
    def _post_forward(self, output): ...
    def forward(self, *inputs, **kwargs): ...
    def scatter(self, inputs, kwargs, device_ids): ...
    def to_kwargs(self, inputs, kwargs, device_id): ...
    def gather(self, outputs, output_device): ...
    def train(self, mode: bool = True): ...
    def _check_global_requires_backward_grad_sync(self, is_joined_rank): ...
    def _check_and_sync_module_buffers(self) -> None: ...
    _authoritative_rank: Incomplete
    def _sync_final_model(self, is_last_joiner) -> None: ...
    def _match_all_reduce_for_bwd_pass(self) -> None: ...
    def _match_unused_params_allreduce(self) -> None: ...
    def join(self, divide_by_initial_world_size: bool = True, enable: bool = True, throw_on_early_termination: bool = False):
        '''
        Context manager for training with uneven inputs across processes in DDP.

        This context manager will keep track of already-joined DDP processes,
        and "shadow" the forward and backward passes by inserting collective
        communication operations to match with the ones created by non-joined
        DDP processes. This will ensure each collective call has a corresponding
        call by already-joined DDP processes, preventing hangs or errors that
        would otherwise happen when training with uneven inputs across
        processes. Alternatively, if the flag ``throw_on_early_termination`` is
        specified to be ``True``, all trainers will throw an error once one rank
        runs out of inputs, allowing these errors to be caught and handled
        according to application logic.

        Once all DDP processes have joined, the context manager will broadcast
        the model corresponding to the last joined process to all processes to
        ensure the model is the same across all processes
        (which is guaranteed by DDP).

        To use this to enable training with uneven inputs across processes,
        simply wrap this context manager around your training loop. No further
        modifications to the model or data loading is required.

        .. warning::
            If the model or training loop this context manager is wrapped around
            has additional distributed collective operations, such as
            ``SyncBatchNorm`` in the model\'s forward pass, then the flag
            ``throw_on_early_termination`` must be enabled. This is because this
            context manager is not aware of non-DDP collective communication.
            This flag will cause all ranks to throw when any one rank
            exhausts inputs, allowing these errors to be caught and recovered
            from across all ranks.

        Args:
            divide_by_initial_world_size (bool): If ``True``, will divide
                gradients by the initial ``world_size`` DDP training was launched
                with. If ``False``, will compute the effective world size
                (number of ranks that have not depleted their inputs yet) and
                divide gradients by that during allreduce. Set
                ``divide_by_initial_world_size=True`` to ensure every input
                sample including the uneven inputs have equal weight in terms of
                how much they contribute to the global gradient. This is
                achieved by always dividing the gradient by the initial
                ``world_size`` even when we encounter uneven inputs. If you set
                this to ``False``, we divide the gradient by the remaining
                number of nodes. This ensures parity with training on a smaller
                ``world_size`` although it also means the uneven inputs would
                contribute more towards the global gradient. Typically, you
                would want to set this to ``True`` for cases where the last few
                inputs of your training job are uneven. In extreme cases, where
                there is a large discrepancy in the number of inputs, setting
                this to ``False`` might provide better results.
            enable (bool): Whether to enable uneven input detection or not. Pass
                in ``enable=False`` to disable in cases where you know that
                inputs are even across participating processes. Default is
                ``True``.
            throw_on_early_termination (bool): Whether to throw an error
                or continue training when at least one rank has exhausted
                inputs. If ``True``, will throw upon the first rank reaching end
                of data. If ``False``, will continue training with a smaller
                effective world size until all ranks are joined. Note that if
                this flag is specified, then the flag
                ``divide_by_initial_world_size`` would be ignored. Default
                is ``False``.


        Example::

            >>> # xdoctest: +SKIP("Distributed")
            >>> import torch
            >>> import torch.distributed as dist
            >>> import os
            >>> import torch.multiprocessing as mp
            >>> import torch.nn as nn
            >>> # On each spawned worker
            >>> def worker(rank):
            >>>     dist.init_process_group("nccl", rank=rank, world_size=2)
            >>>     torch.cuda.set_device(rank)
            >>>     model = nn.Linear(1, 1, bias=False).to(rank)
            >>>     model = torch.nn.parallel.DistributedDataParallel(
            >>>         model, device_ids=[rank], output_device=rank
            >>>     )
            >>>     # Rank 1 gets one more input than rank 0.
            >>>     inputs = [torch.tensor([1]).float() for _ in range(10 + rank)]
            >>>     with model.join():
            >>>         for _ in range(5):
            >>>             for inp in inputs:
            >>>                 loss = model(inp).sum()
            >>>                 loss.backward()
            >>>     # Without the join() API, the below synchronization will hang
            >>>     # blocking for rank 1\'s allreduce to complete.
            >>>     torch.cuda.synchronize(device=rank)
        '''
    def join_hook(self, **kwargs):
        """
        DDP join hook enables training on uneven inputs by mirroring communications in forward and backward passes.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        The hook supports the following keyword arguments:
            divide_by_initial_world_size (bool, optional):
                If ``True``, then gradients are divided by the initial world
                size that DDP was launched with.
                If ``False``, then gradients are divided by the effective world
                size (i.e. the number of non-joined processes), meaning that
                the uneven inputs contribute more toward the global gradient.
                Typically, this should be set to ``True`` if the degree of
                unevenness is small but can be set to ``False`` in extreme
                cases for possibly better results.
                Default is ``True``.
        """
    @property
    def join_device(self): ...
    @property
    def join_process_group(self): ...
    buffer_hook: Incomplete
    def _register_buffer_comm_hook(self, state, hook: Callable, comm_hook_location=...):
        """
        Allow custom registration of hooks that define how buffer are synchronized across ranks.

        The hook takes in an optional state and is passed in a Dict[str, Tensor]
        corresponding to buffer names and the buffers, and can run arbitrary reductions
        on buffers as opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                List[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
    def register_comm_hook(self, state: object, hook: Callable):
        """
        Register communication hook for user-defined DDP aggregation of gradients across multiple workers.

        This hook would be very useful for researchers to try out new ideas. For
        example, this hook can be used to implement several algorithms like GossipGrad
        and gradient compression which involve different communication strategies for
        parameter syncs while running Distributed DataParallel training.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in GossipGrad, etc.

                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable with the following signature:
                             ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``:

                             This function is called once the bucket is ready. The
                             hook can perform whatever processing is needed and return
                             a Future indicating completion of any async work (ex: allreduce).
                             If the hook doesn't perform any communication, it still
                             must return a completed Future. The Future should hold the
                             new value of grad bucket's tensors. Once a bucket is ready,
                             c10d reducer would call this hook and use the tensors returned
                             by the Future and copy grads to individual parameters.
                             Note that the future's return type must be a single tensor.

                             We also provide an API called ``get_future`` to retrieve a
                             Future associated with the completion of ``c10d.ProcessGroup.Work``.
                             ``get_future`` is currently supported for NCCL and also supported for most
                             operations on GLOO and MPI, except for peer to peer operations (send/recv).

        .. warning ::
            Grad bucket's tensors will not be predivided by world_size. User is responsible
            to divide by the world_size in case of operations like allreduce.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        .. warning ::
            The Future object that hook returns should contain a single tensor
            that has the same shape with the tensors inside grad bucket.

        .. warning ::
            ``get_future`` API supports NCCL, and partially GLOO and MPI backends (no support
            for peer-to-peer operations like send/recv) and will return a ``torch.futures.Future``.

        Example::
            Below is an example of a noop hook that returns the same tensor.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def noop(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     fut = torch.futures.Future()
            >>>     fut.set_result(bucket.buffer())
            >>>     return fut
            >>> ddp.register_comm_hook(state=None, hook=noop)

        Example::
            Below is an example of a Parallel SGD algorithm where gradients are encoded before
            allreduce, and then decoded after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> def encode_and_decode(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
            >>>     encoded_tensor = encode(bucket.buffer())  # encode gradients
            >>>     fut = torch.distributed.all_reduce(encoded_tensor).get_future()
            >>>     # Define the then callback to decode.
            >>>     def decode(fut):
            >>>         decoded_tensor = decode(fut.value()[0])  # decode gradients
            >>>         return decoded_tensor
            >>>     return fut.then(decode)
            >>> ddp.register_comm_hook(state=None, hook=encode_and_decode)
        """
    def _register_builtin_comm_hook(self, comm_hook_type) -> None:
        """
        Register a built-in communication hook that specifies how DDP aggregates gradients across multiple workers.

        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
    def _register_fused_optim(self, optim: type, *args, optim_params=None, **kwargs):
        '''
        Register an optimizer in DDP to optimize parameter immediately after its gradient reduction.

        Registers an optimizer with DDP such that the optimization for a
        parameter will run immediately when that parameter\'s gradient is
        finished with reduction, instead of waiting for all parameters\'
        gradients to finish reduction. This can result in a training speedup
        depending on your workload since the optimizer can run while gradient
        reduction for other parameters are still ongoing. In addition, this has
        the potential to reduce peak memory consumption during training, as it
        only needs to load the per-parameter optimizer states of a single
        parameter at a time, instead of loading all per-parameter optimizer
        states at once.

        Args:
            optim (Type): a ``torch.optim.Optimizer`` class to be registered
            as a fused optimizer.
            *args (Sequence[Any]): Arguments to forward to `optim`.
            optim_params (Optional[Iterable[torch.Tensor]]): Set of parameters
            to optimize, similar to `params` argument of traditional `torch.optim`
            Optimizers. If this is omitted, all DDP model parameters will be
            optimized.
            **kwargs: (Dict[str, Any]): Keyword arguments to forward to `optim`.

        .. warning ::
            _register_fused_optim should only be called once on a DDP instance,
            and registering multiple fused optimizers for the same DDP model
            is not currently supported. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            _register_fused_optim and register_comm_hook currently do not
            compose together, meaning that custom DDP communication hooks are
            not supported with overlapped optimizers. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            Gradient accumulation and DDP `no_sync` are currently not supported
            with overlapped optimizer. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        Example::

            >>> # xdoctest: +SKIP("No rendezvous handler")
            >>> torch.distributed.init_process_group(backend=\'nccl\', world_size=4, init_method=\'...\')
            >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> lr = 1e-2
            >>> betas = (0.9, 0.99)
            >>> eps = 1e-6
            >>> net._register_fused_optim(torch.optim.Adam, lr, betas=betas, eps=eps)
            >>> # Example with subset of parameters
            >>> params_to_opt = [list(net.parameters())[0]]
            >>> net._register_fused_optim(
            ...   torch.optim.Adam, lr, optim_params=params_to_opt,  betas=betas, eps=eps
            ... )
        '''
    def _distributed_broadcast_coalesced(self, tensors, buffer_size, authoritative_rank: int = 0) -> None: ...
    def _check_sync_bufs_post_fwd(self): ...
    def _check_sync_bufs_pre_fwd(self): ...
    def will_sync_module_buffers(self): ...
    def _find_common_rank(self, input_rank, rank_cond): ...
    def _sync_buffers(self) -> None: ...
    def _sync_module_buffers(self, authoritative_rank) -> None: ...
    def _default_broadcast_coalesced(self, bufs=None, bucket_size=None, authoritative_rank: int = 0) -> None:
        """
        Broadcasts buffers from rank 0 to rest of workers.

        If bufs, bucket_size are None, default values self.modules_buffers
        and self.broadcast_bucket_size are used instead.
        """
    def _passing_sync_batchnorm_handle(self, module) -> None: ...
    def _check_comm_hook(self, hook) -> None: ...
    @property
    def _distributed_rank(self): ...
    @staticmethod
    def _get_data_parallel_params(module, named_params: bool = False) -> Generator[Incomplete]:
        """Return a generator of parameters managed by a given DDP unit."""
    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(module, params_and_buffers_to_ignore) -> None:
        '''
        Set parameters and buffers to be ignored by DDP.

        Expected format for parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        '''
    def _get_ddp_logging_data(self):
        """
        Return a dictionary of logging data for debugging and analysis.

        This interface can be called after DistributedDataParallel() is
        constructed. It returns a dictionary of logging data. It could help
        for debugging and analysis. The logging data includes DistributedDataParallel
        constructor input parameters, some internal states of DistributedDataParallel
        and performance metrics. Simply print the dictionary and see what
        these metrics are.
        This is a prototype interface and subject to change in the future.
        """
    def _set_ddp_runtime_logging_sample_rate(self, sample_rate) -> None:
        '''
        Set sample_rate of collecting runtime stats.

        This interface allows users to set sample_rate of collecting
        runtime stats. The runtime stats will be recorded for the
        first 10 iterations, after 10 iterations runtime stats will be
        recorded once every "sample_rate" training iterations. In
        default, runtime stats are recorded for the first 10 iterations,
        after 10 iterations runtime stats are recorded once every
        "kDDPRuntimeLoggingSampleRate=100" training iterations.
        This is a prototype interface and subject to change in the future.
        '''
    _static_graph_delay_allreduce_enqueued: bool
    def _set_static_graph(self) -> None:
        """
        Set static graph for DDP.

        It is recommended to set static graph in the DDP constructor, which will
        call this private API internally.
        """
    def _remove_autograd_hooks(self) -> None:
        """Remove autograd hooks registered by the reducer on the model parameters."""
    def _check_reducer_finalized(self) -> None:
        """
        Check if the reducer has processed all buckets and finalized the backward appropriately.

        It is useful to call this method after calling .backward() in your training loop
        in order to avoid subsequent hard to debug errors down the road due to the
        reducer not finalizing backward.
        """
    def _set_sparse_metadata(self, global_unique_ids) -> None: ...
    def _update_process_group(self, new_process_group) -> None:
        """
        Dynamically updates the process group for DDP so that we can shrink/expand DDP
        world size without having to reinitialize DDP.

        NOTE: If you are using custom communications hooks via, register_comm_hook,
        you need to update the process groups for those hooks separately.
        """
    def _set_ddp_sink_clone(self, val: bool):
        """
        Sets whether or not DDPSink should clone the output tensors or not.
        The default is True since if the loss is modified in place we run
        into the view is modified in-place error.

        Although, cloning the tensors can add significant memory and
        performance hit if the number and size of tensors are large. As
        a result, this can be set to False if you are not modifying the
        loss in place.
        """
