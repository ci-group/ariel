import enum
import torch
import torch.distributed as dist
from _typeshed import Incomplete
from torch.distributed.algorithms.join import JoinHook, Joinable
from torch.optim import Optimizer
from typing import Any, Callable

__all__ = ['ZeroRedundancyOptimizer']

class _ZeROJoinHook(JoinHook):
    zero: Incomplete
    def __init__(self, zero) -> None: ...
    def main_hook(self) -> None:
        """
        Perform an optimizer step.

        This step updates the joined process's shard of
        the parameters and broadcasts those parameters.
        """

class _DDPBucketAssignment:
    """
    Represent a :class:`DistributedDataParallel` bucket assignment.

    This means that a (possibly non-strict) subset of the parameters corresponding to
    a DDP bucket assigned to a rank to update.

    Attributes:
        bucket_index (int): index of the bucket determined by the DDP gradient
            bucket all-reduce order.
        parameters (List[torch.Tensor]): model parameters in the bucket
            assigned to this rank.
        offset (int): offset into the :class:`GradBucket` 's :meth:`parameters`
            giving the index of the first element in the passed-in
            ``parameters``; this equivalently indexes into the
            :class:`GradBucket` 's :meth:`gradients`.
        device (torch.device): device on which the parameters are stored.
        tensor (torch.Tensor): flattened tensor giving the data of the
            parameter subset assigned to the rank.
    """
    bucket_index: Incomplete
    parameters: Incomplete
    offset: Incomplete
    device: torch.device
    tensor: torch.Tensor | None
    def __init__(self, bucket_index: int, parameters: list[torch.Tensor], offset: int) -> None: ...

class _OverlapStatus(enum.IntEnum):
    """
    Define possible statuses that :class:`ZeroRedundancyOptimizer` can be in when overlapping with :class:`DistributedDataParallel`.

    Attributes:
        ``UNINITIALIZED``: The ZeRO instance is effectively uninitialized and
            is waiting for DDP to finalize its bucketing.
        ``DDP_HAS_REBUILT_BUCKETS``: DDP has rebuilt its buckets, meaning that
            its bucketing is finalized. The ZeRO instance can now collect the
            necessary information about the DDP bucketing.
        ``INITIALIZED``: The ZeRO instance is fully initialized and can now
            optimize parameters.
    """
    UNINITIALIZED = 0
    DDP_HAS_REBUILT_BUCKETS = 1
    INITIALIZED = 2

class _OverlapInfo:
    """
    Information needed by :class:`ZeroRedundancyOptimizer` to overlap with :class:`DistributedDataParallel`.

    Arguments:
        world_size (int): world size of the process group being used.

    Attributes:
        shard_buckets (bool): if ``True``, then the assignment of each
            :class:`DistributedDataParallel` bucket is partitioned across
            possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
            across possibly multiple ranks) to approximate uniformity following
            a threshold given by the total parameter size divided by the world
            size; if ``False``, then each bucket is wholly assigned to a single
            :class:`ZeroRedundancyOptimizer` instance (i.e. to a single rank);
            this should be set to the value passed into the hook constructor.
        status (_OverlapStatus): current status; see :class:`_OverlapStatus`
            for more information.
        params_per_bucket (List[List[torch.Tensor]]): ``params_per_bucket[i]``
            gives the model parameters in the ``i``th bucket.
        params_per_rank (List[List[torch.Tensor]]): ``params_per_rank[i]``
            gives the model parameters assigned to the ``i``th rank, where the
            parameters are grouped by increasing bucket indices.
        offsets (Dict[int, int]): maps from bucket index to the offset in
            ``self.params_per_rank[rank]`` giving the index of the first
            parameter in that bucket, where ``rank`` is this process's own
            rank; the keys of this :class:`dict` are the bucket indices
            assigned to this rank.
        num_bucket_assignments (int): total number of bucket assignments across
            all ranks; this is equal to the number of
            :class:`DistributedDataParallel` gradient buckets if
            ``shard_buckets=False`` and possibly greater otherwise.
        total_size (int, optional): total size of all buckets (i.e. sum of
            ``param.numel()`` for all ``param`` across all buckets) if
            ``shard_buckets=True``; otherwise, ``None``.
        broadcast_handles (List[Work]): :class:`list` of async work handles for
            the parameter broadcasts.
        bucket_index_to_future (Dict[int, torch.futures.Future]):
            :class:`dict` mapping bucket index to the corresponding all-reduce
            future.
        bucket_index_to_bucket (Dict[int, dist.GradBucket]): :class:`dict`
            mapping bucket index to the corresponding bucket.
        bucket_indices_seen (List[int]): :class:`list` of the bucket indices
            seen on this iteration.
    """
    status: _OverlapStatus
    shard_buckets: bool
    params_per_bucket: list[list[torch.Tensor]]
    params_per_rank: list[list[torch.Tensor]]
    offsets: dict[int, int]
    assigned_ranks_per_bucket: list[set[int]]
    num_bucket_assignments: int
    total_size: int | None
    broadcast_handles: list[Any]
    bucket_indices_seen: list[int]
    bucket_index_to_future: dict[int, torch.futures.Future]
    bucket_index_to_bucket: dict[int, dist.GradBucket]
    def __init__(self, world_size) -> None: ...
    def wait_for_broadcasts(self) -> None:
        """
        Wait for all parameter broadcasts.

        This function should be called once all broadcasts have been scheduled,
        meaning ``self.broadcast_handles`` is filled. This clears ``self.broadcast_handles``
        in preparation for the next iteration.
        """
    def clear_per_iter_info(self) -> None:
        """
        Clear the data structures that are modified per-iteration.

        This function should be called at the end of an iteration.
        """

class ZeroRedundancyOptimizer(Optimizer, Joinable):
    """
    Wrap an arbitrary :class:`optim.Optimizer <torch.optim.Optimizer>` and shards its states across ranks in the group.

    The sharing is done as described by `ZeRO <https://arxiv.org/abs/1910.02054>`_.

    The local optimizer instance in each rank is only
    responsible for updating approximately ``1 / world_size`` parameters and
    hence only needs to keep ``1 / world_size`` optimizer states. After
    parameters are updated locally, each rank will broadcast its parameters to
    all other peers to keep all model replicas in the same state.
    ``ZeroRedundancyOptimizer`` can be used in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel` to reduce per-rank peak
    memory consumption.

    ``ZeroRedundancyOptimizer`` uses a sorted-greedy algorithm to pack a number
    of parameters at each rank. Each parameter belongs to a single rank and is
    not divided among ranks. The partition is arbitrary and might not match the
    the parameter registration or usage order.

    Arguments:
        params (``Iterable``): an ``Iterable`` of :class:`torch.Tensor` s
            or :class:`dict` s giving all parameters, which will be sharded
            across ranks.

    Keyword Args:
        optimizer_class (:class:`torch.nn.Optimizer`): the class of the local
            optimizer.
        process_group (``ProcessGroup``, optional): ``torch.distributed``
            ``ProcessGroup`` (default: ``dist.group.WORLD`` initialized by
            :meth:`torch.distributed.init_process_group`).
        parameters_as_bucket_view (bool, optional): if ``True``, parameters are
            packed into buckets to speed up communication, and ``param.data``
            fields point to bucket views at different offsets; if ``False``,
            each individual parameter is communicated separately, and each
            ``params.data`` stays intact (default: ``False``).
        overlap_with_ddp (bool, optional): if ``True``, :meth:`step` is
            overlapped with :class:`DistributedDataParallel` 's gradient
            synchronization; this requires (1) either a functional optimizer
            for the ``optimizer_class`` argument or one with a functional
            equivalent and (2) registering a DDP communication hook
            constructed from one of the functions in ``ddp_zero_hook.py``;
            parameters are packed into buckets matching those in
            :class:`DistributedDataParallel`, meaning that the
            ``parameters_as_bucket_view`` argument is ignored.
            If ``False``, :meth:`step` runs disjointly after the backward pass
            (per normal).
            (default: ``False``)
        **defaults: any trailing arguments, which are forwarded to the local
            optimizer.

    Example::

        >>> # xdoctest: +SKIP
        >>> import torch.nn as nn
        >>> from torch.distributed.optim import ZeroRedundancyOptimizer
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> model = nn.Sequential(*[nn.Linear(2000, 2000).to(rank) for _ in range(20)])
        >>> ddp = DDP(model, device_ids=[rank])
        >>> opt = ZeroRedundancyOptimizer(
        >>>     ddp.parameters(),
        >>>     optimizer_class=torch.optim.Adam,
        >>>     lr=0.01
        >>> )
        >>> ddp(inputs).sum().backward()
        >>> opt.step()

    .. warning::
        Currently, ``ZeroRedundancyOptimizer`` requires that all of the
        passed-in parameters are the same dense type.

    .. warning::
        If you pass ``overlap_with_ddp=True``, be wary of the following: Given
        the way that overlapping :class:`DistributedDataParallel` with
        :class:`ZeroRedundancyOptimizer` is currently implemented, the first
        two or three training iterations do not perform parameter updates in
        the optimizer step, depending on if ``static_graph=False`` or
        ``static_graph=True``, respectively. This is because it needs
        information about the gradient bucketing strategy used by
        :class:`DistributedDataParallel`, which is not finalized until the
        second forward pass if ``static_graph=False`` or until the third
        forward pass if ``static_graph=True``. To adjust for this, one option
        is to prepend dummy inputs.

    .. warning:: ZeroRedundancyOptimizer is experimental and subject to change.
    """
    initialized: bool
    _param_to_rank_cache: dict[torch.Tensor, int]
    _param_to_index_cache: dict[torch.Tensor, int]
    _partition_parameters_cache: list[list[dict]]
    _index_to_param_cache: list[torch.Tensor]
    _device_to_params_per_rank_cache: dict[torch.device, list[list[torch.Tensor]]]
    _bucket_assignments_per_rank_cache: list[dict[int, _DDPBucketAssignment]]
    _is_trainable_mask: Incomplete
    _default_device: Incomplete
    process_group: Incomplete
    world_size: int
    rank: int
    global_rank: int
    _overlap_with_ddp: bool
    _optim_defaults: Incomplete
    _optim_constructor: Incomplete
    _overlap_info: _OverlapInfo
    parameters_as_bucket_view: Incomplete
    _buckets: list[list[torch.Tensor]]
    _all_state_dicts: list[dict[str, Any]]
    def __init__(self, params, optimizer_class: type[Optimizer], process_group: Any | None = None, parameters_as_bucket_view: bool = False, overlap_with_ddp: bool = False, **defaults: Any) -> None:
        """Init."""
    def _clear_cache(self) -> None:
        """Clear the cached data structures giving partition information."""
    def add_param_group(self, param_group: dict[str, Any]) -> None:
        """
        Add a parameter group to the :class:`Optimizer` 's ``param_groups``.

        This can be useful when fine tuning a pre-trained network, as frozen
        layers can be made trainable and added to the :class:`Optimizer` as
        training progresses.

        Arguments:
            param_group (dict): specifies the parameters to be optimized and
                group-specific optimization options.

        .. warning:: This method handles updating the shards on all partitions
            but needs to be called on all ranks. Calling this on a subset of
            the ranks will cause the training to hang because communication
            primitives are called depending on the managed parameters and
            expect all the ranks to participate on the same set of parameters.
        """
    def consolidate_state_dict(self, to: int = 0) -> None:
        """
        Consolidate a list of ``state_dict`` s (one per rank) on the target rank.

        Arguments:
            to (int): the rank that receives the optimizer states (default: 0).

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.

        .. warning:: This needs to be called on all ranks.
        """
    def _verify_params_per_rank(self, params_per_rank: list[list[torch.Tensor]]) -> None:
        """
        Verify ``params_per_rank`` for :meth:`_partition_parameters`.

        The verification is done by checking that ``params_per_rank`` has length equal
        to the world size and that it does not contain any parameters not passed into the
        :class:`ZeroRedundancyOptimizer` constructor.

        The parameters in ``params_per_rank`` being a strict subset of those
        passed into the constructor is valid since some parameters may be
        frozen.

        Raises:
            ValueError: if ``params_per_rank`` does not have length equal to
                the world size or if it contains a parameter that was not
                passed into the :class:`ZeroRedundancyOptimizer` constructor.
        """
    def _partition_param_group(self, param_group: dict[str, Any], params_per_rank: list[list[torch.Tensor]]) -> None:
        """
        Partition the parameter group ``param_group`` according to ``params_per_rank``.

        The partition will modify the ``self._partition_parameters_cache``. This method should
        only be used as a subroutine for :meth:`_partition_parameters`.

        Arguments:
            param_group (dict[str, Any]): a parameter group as normally defined
                in an optimizer state.
            params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
                length world size containing :class:`list` s of parameters to
                assign to each rank.
        """
    def _partition_parameters(self, params_per_rank: list[list[torch.Tensor]] | None = None) -> list[list[dict]]:
        """
        Partitions parameters across distributed data parallel ranks.

        Arguments:
            params_per_rank (list[list[torch.Tensor]], optional): a
                :class:`list` of length world size containing :class:`list` s
                of parameters to assign to each rank; this provides a way to
                specify a partition manually.
                If ``None``, the parameters are partitioned according to an
                internal algorithm.
                (default: ``None``)

        Returns:
            A :class:`list` where each element of the list contains the
            ``param_groups`` for a rank (which itself is a :class:`list` of
            :class:`dict`); element 0 corresponds to rank 0, etc.; each rank
            stores the ``param_groups`` for all ranks for the collective
            communication in :meth:`step`.

        Raises:
            ValueError: see :meth:`_validate_params_per_rank`.
            RuntimeError: if ``params_per_rank`` is not ``None`` and this
                :class:`ZeroRedundancyOptimizer` instance is using more than
                one parameter group.
        """
    @property
    def _param_to_rank(self) -> dict[torch.Tensor, int]:
        """:class:`dict` mapping parameters to their assigned data parallel rank in the partition."""
    @property
    def _param_to_index(self) -> dict[torch.Tensor, int]:
        """
        :class:`dict` mapping parameters to their indices in the global optimizer state.

        NOTE: This assumes that the global optimizer state's indexing (in
        ``state_dict``) follows a linear ordering over the parameter groups.
        """
    @property
    def _index_to_param(self) -> list[torch.Tensor]:
        """List mapping parameter indices in the global optimizer scheme to the actual params."""
    def _broadcast_params_from_rank(self, rank: int):
        """
        Broadcast the shard of parameters from a given rank to all other ranks asynchronously.

        Arguments:
            rank (int): the source rank.

        Returns:
            A :class:`list` of async work handles for the ``broadcast()`` s
            performed to synchronize the parameters.
        """
    def _sync_params(self) -> None:
        """
        Sync all parameter shards across the ranks.

        This rank sends its shard of the parameters to all other ranks and
        receives a shard from each other rank. This is done using
        ``broadcast()``. Parameters are sent bucket-by-bucket if
        ``parameters_as_bucket_view=True``and sent parameter-by-parameter
        otherwise.
        """
    @property
    def _device_to_params_per_rank(self) -> dict[torch.device, list[list[torch.Tensor]]]:
        """
        Return device parameters assigned per rank.

        :class:`dict` mapping each device to a :class:`list` of the per-rank parameter
        lists filtered to only include the parameters stored on that device.
        Each per-rank parameter list gives the parameters assigned to that rank
        to update.

        This is used for constructing the parameter buckets if
        ``parameters_as_bucket_view=True``.

        Let ``dev_i`` denote the ``i``th device for this rank. Then:
        ``dev_0`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_0``,
            rank 1's assigned parameters stored on ``dev_0``,
            ...
        ``dev_1`` maps to a list containing:
            rank 0's assigned parameters stored on ``dev_1``,
            rank 1's assigned parameters stored on ``dev_1``,
            ...
        ...
        """
    def _get_min_index(self, values: list[int], disallowed_indices: set[int] | None = None) -> int:
        """
        Return ``values.index(min(values))``, except only uses one pass.

        It also excludes any indices in ``disallowed_indices`` if provided.

        Arguments:
            values: (List[int]): :class:`list` of values.
            disallowed_indices (Optional[set[int]]): indices that are
                disallowed from being the returned min index.
        """
    def _assign_bucket_subset_to_rank(self, bucket_index: int, bucket_params: list[torch.Tensor], bucket_offset: int, assigned_rank: int, assigned_ranks_per_bucket: list[set[int]]) -> None:
        """
        Assign ``bucket_params`` to the rank with the least size assigned so far and collects relevant information.

        The model parameters given by ``bucket_params`` represents a (possibly non-strict)
        subset of the parameters corresponding to a :class:`DistributedDataParallel` bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                gradient bucket.
            bucket_params (List[torch.Tensor]): subset of the parameters
                corresponding to the bucket to assign.
            bucket_offset (int): offset giving the index of the first element
                in ``bucket_params`` in the bucket's full parameter list.
            assigned_rank (int): group rank to assign to.
            assigned_ranks_per_bucket (list[set[int]]): :class:`set` of group ranks
                assigned to each bucket.
        """
    @property
    def _bucket_assignments_per_rank(self) -> list[dict[int, _DDPBucketAssignment]]:
        """
        Return DDP bucket parameters assigned per rank.

        :class:`list` of length world size consisting of :class:`dict` s
        mapping bucket indices to :class:`_DDPBucketAssignment` s for each
        rank.
        """
    def _local_step(self, gradients: list[torch.Tensor | None] | None = None, closure: Callable[[], float] | None = None, **kwargs: Any) -> float | None:
        """
        Perform a single optimizer step without syncing parameters across ranks.

        Arguments:
            gradients (list[Optional[torch.Tensor]], optional): a :class:`list`
                of length equal to the number of parameters assigned to this
                rank containing gradient tensors or ``None`` as its elements;
                a ``None`` in the :class:`list` indicates that the
                corresponding parameter should not be updated.
                If the argument itself is ``None``, then all parameters are
                updated, and the gradients are assumed to be already populated.
                (default: ``None``)
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers and should be
                ``None`` if ``gradients`` is not ``None``; (default: ``None``)
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. warning::
            The argument ``gradients`` should only be specified (i.e. not
            ``None``) if ``overlap_with_ddp=True``, in which case
            :class:`ZeroRedundancyOptimizer` wraps a functional optimizer.
        """
    def step(self, closure: Callable[[], float] | None = None, **kwargs: Any) -> float | None:
        """
        Perform a single optimizer step and syncs parameters across all ranks.

        Arguments:
            closure (Callable): a closure that re-evaluates the model and
                returns the loss; optional for most optimizers.
        Returns:
            Optional loss depending on the underlying local optimizer.

        .. note:: Any extra parameters are passed to the base optimizer as-is.
        """
    def join_hook(self, **kwargs):
        """
        Return the ZeRO join hook.

        It enables training on uneven inputs by
        shadowing the collective communications in the optimizer step.

        Gradients must be properly set before this hook is called.

        Arguments:
            kwargs (dict): a :class:`dict` containing any keyword arguments
                to modify the behavior of the join hook at run time; all
                :class:`Joinable` instances sharing the same join context
                manager are forwarded the same value for ``kwargs``.

        This hook does not support any keyword arguments; i.e. ``kwargs`` is
        unused.
        """
    @property
    def join_device(self) -> torch.device:
        """Return default device."""
    @property
    def join_process_group(self) -> Any:
        """Return process group."""
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state pertaining to the given rank from the input ``state_dict``, updating the local optimizer as needed.

        Arguments:
            state_dict (dict): optimizer state; should be an object returned
                from a call to :meth:`state_dict`.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt.
        """
    def state_dict(self) -> dict[str, Any]:
        """
        Return the last global optimizer state known to this rank.

        .. warning:
            If the state has not been consolidated to this rank, this raises a
            runtime error, and even if it has, the state may not be up-to-date,
            depending on when :meth:`consolidate_state_dict` was last called.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and this method is
                called before this :class:`ZeroRedundancyOptimizer` instance
                has been fully initialized, which happens once
                :class:`DistributedDataParallel` gradient buckets have been
                rebuilt; or if this method is called without a preceding call
                to :meth:`consolidate_state_dict`.
        """
    @staticmethod
    def _sync_param_groups(src_param_groups: list[dict[Any, Any]], dst_param_groups: list[dict[Any, Any]]) -> None:
        """
        Sync the attributes from the source parameter groups to the destination parameter groups.

        Example attributes include learning rate or scheduler attributes. The
        two parameter groups should have the same length (i.e. same number of
        parameter groups).

        Arguments:
            src_param_groups (list[dict]): parameter groups giving the
                attribute settings to copy.
            dst_param_groups (list[dict]): parameter groups giving the
                attribute settings to set.
        """
    def _build_param_buckets(self) -> None:
        """
        Build parameter buckets if ``parameters_as_bucket_view=True``.

        For each device that stores this rank's parameters, there is a
        bucket (represented as a tensor) containing all of the parameters on
        that device that are assigned to a given rank in the parameter update
        partition.

        This method is called in the constructor and any time parameter
        trainability is changed.

        .. warning::
            The current implementation assumes that all of the parameters in a
            bucket are of the same dense type when allocating the bucket's
            tensor.

        .. warning::
            If the model parameters are stored across more than one device,
            then the storage partitioning must be the same across all
            processes in order for parameter synchronization to work.
        """
    def _build_ddp_param_buckets(self) -> None:
        """
        Build the DDP bucket with parameters assigned to this rank.

        For each DDP bucket with parameters assigned to this rank, flattens the
        data of those parameters into a single tensor and saves the tensor to
        the ``tensor`` attribute in the corresponding
        :class:`_DDPBucketAssignment` instance stored in
        ``self._bucket_assignments_per_rank``.

        :class:`DistributedDataParallel` guarantees that the parameters
        corresponding to a gradient bucket have the same device and the same
        dtype.
        """
    _all_params: Incomplete
    def _verify_and_init_params(self, params: Any) -> list[torch.Tensor] | list[dict]:
        """
        Verify the type of ``params`` and initializes ``self._all_params`` as a :class:`list` of all parameters.

        The initializagtion will first make sure that provided ``params`` is valid.

        Arguments:
            params (Any): Candidate parameter list or parameter groups to verify.

        Raises:
            TypeError: ``params`` has an invalid type.
            ValueError: ``params`` is empty.

        Returns:
            The persistent form of ``params`` to be passed into the parent
            :class:`Optimizer` constructor -- i.e. returns ``params`` as a
            :class:`list` to ensure that it can be iterated over again.
        """
    def _verify_same_dense_param_type(self) -> None:
        """
        Verify that all parameters are of the same dense type.

        The method assumes that ``self._all_params`` has been initialized
        and is non-empty.

        Raises:
            ValueError: ``params`` contains sparse parameters or parameters
            of varying dense types.

        NOTE: This method can be removed once support for sparse parameters
        and varying parameter types is added.
        """
    def _get_is_trainable_mask(self) -> list[bool]:
        """Return a boolean mask indicating if each parameter is trainable (``requires_grad``) or not."""
    optim: Any
    def _init_local_optimizer(self) -> None:
        """
        Initialize this rank's local optimizer, responsible for its subset of the parameters.

        The local optimizer is saved in ``self.optim``.
        """
    def _init_zero_for_overlap(self) -> None:
        """Perform a delayed initialization of the local optimizer and the supporting data structures."""
    def _get_assigned_rank(self, bucket_index: int) -> int:
        """
        Return the single rank assigned to a :class:`DistributedDataParallel` gradient bucket.

        Arguments:
            bucket_index (int): index of the :class:`DistributedDataParallel`
                bucket for which to get the assigned rank.
        """
    def _check_overlap_initialized(self) -> None:
        """
        Check the delayed initialization depending on the value of ``overlap_with_ddp``.

        The delayed initialization has occurred (see
        :meth:`_init_zero_for_overlap`) if ``overlap_with_ddp=True``, and
        raises a ``RuntimeError`` if not. This should preface methods that
        should not be run before that delayed initialization.

        Raises:
            RuntimeError: if ``overlap_with_ddp=True`` and
                :meth:`_init_zero_for_overlap` has not been called.
        """
    def _get_optimizer_constructor(self, optimizer_class: Any) -> Any:
        """
        Return the optimizer constructor using validation and transformation depending on ``overlap_with_ddp``.

        Returns:
            - ``optimizer_class`` if ``overlap_with_ddp=False`` and
                ``optimizer_class`` is not a functional optimizer.
            - ``optimizer_class`` if ``overlap_with_ddp=True`` and
                ``optimizer_class`` is already a functional optimizer.
            - The functional equivalent of ``optimizer_class`` if
                ``overlap_with_ddp=True`` and ``optimizer_class`` is not
                already a functional optimizer (assuming the equivalent
                exists).

        Raises:
            ValueError:

                - if ``overlap_with_ddp=True`` but ``optimizer_class`` is
                    neither a functional optimizer nor translatable to a
                    functional optimizer.
                - if ``overlap_with_ddp=False`` and ``optimizer_class`` is a
                    functional optimizer.
        """
