import contextlib
import pickle
import torch
from .c10d_logger import _exception_logger, _time_logger
from .constants import default_pg_timeout as default_pg_timeout
from _typeshed import Incomplete
from datetime import timedelta
from torch._C._distributed_c10d import AllToAllOptions as AllToAllOptions, AllreduceCoalescedOptions as AllreduceCoalescedOptions, AllreduceOptions as AllreduceOptions, BarrierOptions as BarrierOptions, BroadcastOptions as BroadcastOptions, DebugLevel as DebugLevel, GatherOptions as GatherOptions, PrefixStore as PrefixStore, ProcessGroup as ProcessGroup, ProcessGroupGloo as ProcessGroupGloo, ReduceOp as ReduceOp, ReduceOptions as ReduceOptions, ReduceScatterOptions as ReduceScatterOptions, ScatterOptions as ScatterOptions, Store as Store, Work as Work, get_debug_level as get_debug_level
from typing import Any, Callable, NamedTuple

__all__ = ['Backend', 'BackendConfig', 'GroupMember', 'P2POp', 'all_gather', 'all_gather_coalesced', 'all_gather_object', 'all_reduce', 'all_reduce_coalesced', 'all_to_all', 'all_to_all_single', 'barrier', 'batch_isend_irecv', 'broadcast', 'send_object_list', 'recv_object_list', 'broadcast_object_list', 'destroy_process_group', 'gather', 'gather_object', 'get_backend_config', 'get_backend', 'get_default_backend_for_device', 'get_rank', 'get_world_size', 'get_pg_count', 'group', 'init_process_group', 'irecv', 'is_gloo_available', 'is_initialized', 'is_mpi_available', 'is_backend_available', 'is_nccl_available', 'is_torchelastic_launched', 'is_ucc_available', 'is_xccl_available', 'isend', 'monitored_barrier', 'new_group', 'new_subgroups', 'new_subgroups_by_enumeration', 'recv', 'reduce', 'reduce_scatter', 'scatter', 'scatter_object_list', 'send', 'supports_complex', 'AllreduceCoalescedOptions', 'AllreduceOptions', 'AllToAllOptions', 'BarrierOptions', 'BroadcastOptions', 'GatherOptions', 'PrefixStore', 'ProcessGroup', 'ReduceOp', 'ReduceOptions', 'ReduceScatterOptions', 'ScatterOptions', 'Store', 'DebugLevel', 'get_debug_level', 'Work', 'default_pg_timeout', 'get_group_rank', 'get_global_rank', 'get_process_group_ranks', 'reduce_op', 'all_gather_into_tensor', 'reduce_scatter_tensor', 'get_node_local_rank', 'split_group', 'ProcessGroupGloo']

_pickler = pickle.Pickler
_unpickler = pickle.Unpickler

def supports_complex(reduceOp: ReduceOp) -> bool:
    """Return true if reduce ops is supported. False otherwise."""

class Backend(str):
    '''
    An enum-like class for backends.

    Available backends: GLOO, NCCL, UCC, MPI, XCCL, and other registered backends.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    '''
    UNDEFINED: str
    GLOO: str
    NCCL: str
    UCC: str
    MPI: str
    XCCL: str

    class _BackendPlugin(NamedTuple):
        creator_fn: Incomplete
        extended_api: Incomplete
    _plugins: dict[str, _BackendPlugin]
    backend_list: Incomplete
    default_device_backend_map: dict[str, str]
    backend_capability: dict[str, list[str]]
    backend_type_map: dict[str, ProcessGroup.BackendType]
    def __new__(cls, name: str):
        """Create and return a new instance of the class."""
    @classmethod
    def register_backend(cls, name, func, extended_api: bool = False, devices: str | list[str] | None = None) -> None:
        '''
        Register a new backend with the given name and instantiating function.

        This class method is used by 3rd party ``ProcessGroup`` extension to
        register new backends.

        Args:
            name (str): Backend name of the ``ProcessGroup`` extension. It
                        should match the one in ``init_process_group()``.
            func (function): Function handler that instantiates the backend.
                             The function should be implemented in the backend
                             extension and takes four arguments, including
                             ``store``, ``rank``, ``world_size``, and ``timeout``.
            extended_api (bool, optional): Whether the backend supports extended argument structure.
                                           Default: ``False``. If set to ``True``, the backend
                                           will get an instance of ``c10d::DistributedBackendOptions``, and
                                           a process group options object as defined by the backend implementation.
            device (str or list of str, optional): device type this backend
                            supports, e.g. "cpu", "cuda", etc. If `None`,
                            assuming both "cpu" and "cuda"

        .. note:: This support of 3rd party backend is experimental and subject to change.

        '''

class BackendConfig:
    """Backend configuration class."""
    device_backend_map: dict[str, Backend]
    def __init__(self, backend: Backend) -> None:
        """Init."""
    def __repr__(self) -> str:
        """Return all the device:backend pairs separated by commas."""
    def get_device_backend_map(self) -> dict[str, Backend]:
        """Return backend map of the device."""

class _reduce_op:
    """
    Deprecated enum-like class.

    For reduction operations: ``SUM``, ``PRODUCT``, ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """
    __members__: Incomplete
    def __init__(self) -> None: ...
    def __getattribute__(self, key): ...

reduce_op: Incomplete

class P2POp:
    """
    A class to build point-to-point operations for ``batch_isend_irecv``.

    This class builds the type of P2P operation, communication buffer, peer rank,
    Process Group, and tag. Instances of this class will be passed to
    ``batch_isend_irecv`` for point-to-point communications.

    Args:
        op (Callable): A function to send data to or receive data from a peer process.
            The type of ``op`` is either ``torch.distributed.isend`` or
            ``torch.distributed.irecv``.
        tensor (Tensor): Tensor to send or receive.
        peer (int, optional): Destination or source rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with recv.
        group_peer (int, optional): Destination or source rank.
    """
    op: Incomplete
    tensor: Incomplete
    group: Incomplete
    peer: Incomplete
    tag: Incomplete
    group_peer: Incomplete
    def __init__(self, op: Callable, tensor: torch.Tensor, peer: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_peer: int | None = None) -> None:
        """Init."""
    def __new__(cls, op: Callable, tensor: torch.Tensor, peer: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_peer: int | None = None):
        """Create and return a new instance of the class."""
    def __repr__(self) -> str: ...

class _CollOp:
    """
    A class to capture collective operations.

    Args:
        op (Callable): A collective function, e.g. ``torch.distributed.all_reduce``.
        tensor (Tensor): Tensor to operate on.
        dst_tensor (Tensor, optional): Provided when source and destination tensors are not the same.
        redop (ReduceOp, optional): reduce operation.
        root (int, optional): root of broadcast or reduce.
    """
    op: Incomplete
    tensor: Incomplete
    dst_tensor: Incomplete
    redop: Incomplete
    root: Incomplete
    def __init__(self, op: Callable, tensor: torch.Tensor, dst_tensor: torch.Tensor | None = None, redop: ReduceOp | None = None, root: int | None = None) -> None: ...

class _World:
    """
    Container class for c10d process group state.

    This is used during registration and lookup of PG state.

    .. warning:: This is an experimental API intended to expose the inner workings
       of c10d and is subject to change..
    """
    _default_pg: Incomplete
    _pg_coalesce_state: dict[ProcessGroup, list[_CollOp]]
    def __init__(self) -> None: ...
    @property
    def default_pg(self) -> ProcessGroup | None:
        """
        Process group that includes all ranks of the cluster.

        This default ProcessGroup is used by c10d APIs when a ProcessGroup is needed
        but None is provided.
        """
    @default_pg.setter
    def default_pg(self, value) -> None: ...
    @property
    def pg_map(self) -> dict[ProcessGroup, tuple[str, Store]]:
        """
        Provide Mapping from ProcessGroup to backend name and store.

        For NCCL and GLOO pg, it is a map from ProcessGroup to (Backend, Store)
        For MPI pg, it is a map from ProcessGroup to (Backend, None)

        TODO don't expose the map, expose fine grained ops
        """
    @property
    def pg_names(self) -> dict[ProcessGroup, str]:
        """
        Process group's names, map from ProcessGroup to str.

        TODO don't expose the map, expose fine grained ops
        """
    @property
    def pg_group_ranks(self) -> dict[ProcessGroup, dict[int, int]]:
        """
        Process group's global rank to local rank mapping.

        TODO don't expose the map, expose fine grained ops
        """
    @property
    def pg_backend_config(self) -> dict[ProcessGroup, str]:
        """
        Process group's backend config.

        TODO don't expose the map, expose fine grained ops
        """
    @property
    def group_count(self) -> int:
        """
        Process group count for default naming.

        TODO don't expose group_count, use something else instead
        """
    @group_count.setter
    def group_count(self, value: int) -> None:
        """Use to compute the name of ProcessGroups when using global synchronization."""
    @property
    def tags_to_pg(self) -> dict[str, list[ProcessGroup]]: ...
    @property
    def pg_to_tag(self) -> dict[ProcessGroup, str]: ...
    @property
    def pg_coalesce_state(self) -> dict[ProcessGroup, list[_CollOp]]: ...
    @property
    def pg_config_info(self) -> list[dict[str, Any]]:
        """
        Return a list of dict with process groups and backends.

        Along with their unique IDs and configurations (types and ranks).
        """

class _WorldMeta(type):
    """
    Meta class of ``group`` and ``GroupMember``.

    Allows them to have the class property ``WORLD``.
    """
    @property
    def WORLD(cls) -> ProcessGroup | None: ...
    @WORLD.setter
    def WORLD(cls, pg: ProcessGroup | None): ...

class group(metaclass=_WorldMeta):
    """Group class. Placeholder."""

class GroupMember(metaclass=_WorldMeta):
    """Group member class."""
    NON_GROUP_MEMBER: int

def get_group_rank(group: ProcessGroup, global_rank: int) -> int:
    """
    Translate a global rank into a group rank.

    ``global_rank`` must be part of ``group`` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the relative rank.
        global_rank (int): Global rank to query.

    Returns:
        Group rank of ``global_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
def get_global_rank(group: ProcessGroup, group_rank: int) -> int:
    """
    Translate a group rank into a global rank.

    ``group_rank`` must be part of `group` otherwise this raises RuntimeError.

    Args:
        group (ProcessGroup): ProcessGroup to find the global rank from.
        group_rank (int): Group rank to query.

    Returns:
        Global rank of ``group_rank`` relative to ``group``

    N.B. calling this function on the default process group returns identity
    """
def get_process_group_ranks(group: ProcessGroup | None) -> list[int]:
    """
    Get all ranks associated with ``group``.

    Args:
        group (Optional[ProcessGroup]): ProcessGroup to get all ranks from.
            If None, the default process group will be used.

    Returns:
        List of global ranks ordered by group rank.
    """
def is_mpi_available() -> bool:
    """Check if the MPI backend is available."""
def is_nccl_available() -> bool:
    """Check if the NCCL backend is available."""
def is_gloo_available() -> bool:
    """Check if the Gloo backend is available."""
def is_ucc_available() -> bool:
    """Check if the UCC backend is available."""
def is_xccl_available() -> bool:
    """Check if the XCCL backend is available."""
def is_backend_available(backend: str) -> bool:
    """
    Check backend availability.

    Checks if the given backend is available and supports the built-in backends or
    third-party backends through function ``Backend.register_backend``.

    Args:
        backend (str): Backend name.
    Returns:
        bool: Returns true if the backend is available otherwise false.
    """
def is_initialized() -> bool:
    """Check if the default process group has been initialized."""
def is_torchelastic_launched() -> bool:
    """
    Check whether this process was launched with ``torch.distributed.elastic`` (aka torchelastic).

    The existence of ``TORCHELASTIC_RUN_ID`` environment
    variable is used as a proxy to determine whether the current process
    was launched with torchelastic. This is a reasonable proxy since
    ``TORCHELASTIC_RUN_ID`` maps to the rendezvous id which is always a
    non-null value indicating the job id for peer discovery purposes..
    """
def get_backend_config(group: ProcessGroup | None = None) -> str:
    """
    Return the backend configuration of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend configuration of the given process group as a lower case string.

    """
def get_backend(group: ProcessGroup | None = None) -> Backend:
    """
    Return the backend of the given process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
def get_default_backend_for_device(device: str | torch.device) -> str:
    """
    Return the default backend for the given device.

    Args:
        device (Union[str, torch.device]): The device to get the default backend for.

    Returns:
        The default backend for the given device as a lower case string.

    """
def get_pg_count() -> int:
    """
    Return the number of process groups.

    """
def get_node_local_rank(fallback_rank: int | None = None) -> int:
    """
    Return the local rank of the current process relative to the node.

    Semantically, this is a useful concept for mapping processes to devices.
    For example, on a node with 8 accelerator you could use the node local rank to decide
    which accelerator device to bind the process to.

    In practice, the actual assignment of node local ranks is handled by the process launcher outside of pytorch,
    and communicated via the `LOCAL_RANK` environment variable.

    Torchrun will automatically populate `LOCAL_RANK`, but other launchers may not.  If `LOCAL_RANK` is unspecified,
    this API will fall back to the provided kwarg 'fallback_rank' if specified, otherwise it will raise an error. The
    intent is to allow writing an application that runs either in single or multi device contexts without error.

    """
@_exception_logger
@_time_logger
def init_process_group(backend: str | None = None, init_method: str | None = None, timeout: timedelta | None = None, world_size: int = -1, rank: int = -1, store: Store | None = None, group_name: str = '', pg_options: Any | None = None, device_id: torch.device | int | None = None) -> None:
    '''
    Initialize the default distributed process group.

    This will also initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".


    Args:
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            ``nccl``, ``ucc``, or one that is registered by a third-party
            plugin.
            Since 2.6, if ``backend`` is not provided, c10d will use a backend
            registered for the device type indicated by the `device_id` kwarg
            (if provided). The known default registrations today are: ``nccl``
            for ``cuda``, ``gloo`` for ``cpu``.
            If neither ``backend`` nor ``device_id`` is provided, c10d will
            detect the accelerator on the run-time machine and use a backend
            registered for that detected accelerator (or ``cpu``).
            This field can be given as a lowercase string (e.g., ``"gloo"``),
            which can also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).
            If using multiple processes per machine with ``nccl`` backend, each
            process must have exclusive access to every GPU it uses, as sharing
            GPUs between processes can result in deadlock or NCCL invalid usage.
            ``ucc`` backend is experimental.
            Default backend for the device can be queried with
            :func:`get_default_backend_for_device`.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process (it should be a
                              number between 0 and ``world_size``-1).
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value is 10 minutes for NCCL and 30 minutes for other backends.
            This is the duration after which collectives will be aborted asynchronously and the process will crash.
            This is done since CUDA execution is async and it is no longer safe to continue executing user code since
            failed async NCCL operations might result in subsequent CUDA operations running on corrupted data.
            When TORCH_NCCL_BLOCKING_WAIT is set, the process will block and wait for this timeout.

        group_name (str, optional, deprecated): Group name. This argument is ignored
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. As of now, the only
            options we support is ``ProcessGroupNCCL.Options`` for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            the nccl backend can pick up high priority cuda streams when
            there\'re compute kernels waiting. For other available options to config nccl,
            See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-t
        device_id (torch.device | int, optional): a single, specific device
            this process will work on, allowing for backend-specific
            optimizations.  Currently this has two effects, only under
            NCCL: the communicator is immediately formed (calling
            ``ncclCommInit*`` immediately rather than the normal lazy
            call) and sub-groups will use ``ncclCommSplit`` when
            possible to avoid unnecessary overhead of group creation. If you
            want to know NCCL initialization error early, you can also use this
            field. If an `int` is provided, the API assumes that the accelerator
            type at compile time will be used.

    .. note:: To enable ``backend == Backend.MPI``, PyTorch needs to be built from source
        on a system that supports MPI.

    .. note:: Support for multiple backends is experimental. Currently when no backend is
        specified, both ``gloo`` and ``nccl`` backends will be created. The ``gloo`` backend
        will be used for collectives with CPU tensors and the ``nccl`` backend will be used
        for collectives with CUDA tensors. A custom backend can be specified by passing in
        a string with format "<device_type>:<backend_name>,<device_type>:<backend_name>", e.g.
        "cpu:gloo,cuda:custom_backend".

    '''
def destroy_process_group(group: ProcessGroup | None = None):
    """
    Destroy a given process group, and deinitialize the distributed package.

    Args:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
def get_rank(group: ProcessGroup | None = None) -> int:
    """
    Return the rank of the current process in the provided ``group``, default otherwise.

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
def get_world_size(group: ProcessGroup | None = None) -> int:
    """
    Return the number of processes in the current process group.

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
def isend(tensor: torch.Tensor, dst: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_dst: int | None = None) -> Work | None:
    """
    Send a tensor asynchronously.

    .. warning::
        Modifying ``tensor`` before the request completes causes undefined
        behavior.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike send, which is blocking, isend allows src == dst rank, i.e. send to self.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        A distributed request object.
        None, if not part of the group

    """
def irecv(tensor: torch.Tensor, src: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_src: int | None = None) -> Work | None:
    """
    Receives a tensor asynchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Unlike recv, which is blocking, irecv allows src == dst rank, i.e. recv from self.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.

    Returns:
        A distributed request object.
        None, if not part of the group

    """
@_exception_logger
def send(tensor: torch.Tensor, dst: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_dst: int | None = None) -> None:
    """
    Send a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank on global process group (regardless of ``group`` argument).
            Destination rank should not be the same as the rank of the current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match send with remote recv
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``.

    """
@_exception_logger
def recv(tensor: torch.Tensor, src: int | None = None, group: ProcessGroup | None = None, tag: int = 0, group_src: int | None = None) -> int:
    """
    Receives a tensor synchronously.

    .. warning::
        ``tag`` is not supported with the NCCL backend.

    Args:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank on global process group (regardless of ``group`` argument).
            Will receive from any process if unspecified.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        tag (int, optional): Tag to match recv with remote send
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.
    Returns:
        Sender rank
        -1, if not part of the group

    """

class _IllegalWork(Work):
    def __getattribute__(self, name) -> None: ...

class _CoalescingManager:
    works: list[Work]
    def __init__(self) -> None: ...
    def append(self, work: Work | None = None): ...
    def wait(self) -> None: ...

class _TimeEstimator:
    estimated_time: float | None
    def __init__(self) -> None: ...

def batch_isend_irecv(p2p_op_list: list[P2POp]) -> list[Work]:
    '''
    Send or Receive a batch of tensors asynchronously and return a list of requests.

    Process each of the operations in ``p2p_op_list`` and return the corresponding
    requests. NCCL, Gloo, and UCC backend are currently supported.

    Args:
        p2p_op_list: A list of point-to-point operations(type of each operator is
            ``torch.distributed.P2POp``). The order of the isend/irecv in the list
            matters and it needs to match with corresponding isend/irecv on the
            remote end.

    Returns:
        A list of distributed request objects returned by calling the corresponding
        op in the op_list.

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> send_tensor = torch.arange(2, dtype=torch.float32) + 2 * rank
        >>> recv_tensor = torch.randn(2, dtype=torch.float32)
        >>> send_op = dist.P2POp(dist.isend, send_tensor, (rank + 1) % world_size)
        >>> recv_op = dist.P2POp(
        ...     dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size
        ... )
        >>> reqs = batch_isend_irecv([send_op, recv_op])
        >>> for req in reqs:
        >>>     req.wait()
        >>> recv_tensor
        tensor([2, 3])     # Rank 0
        tensor([0, 1])     # Rank 1

    .. note:: Note that when this API is used with the NCCL PG backend, users must set
        the current GPU device with `torch.cuda.set_device`, otherwise it will
        lead to unexpected hang issues.

        In addition, if this API is the first collective call in the ``group``
        passed to ``dist.P2POp``, all ranks of the ``group`` must participate in
        this API call; otherwise, the behavior is undefined. If this API call is
        not the first collective call in the ``group``, batched P2P operations
        involving only a subset of ranks of the ``group`` are allowed.
    '''
@_exception_logger
def broadcast(tensor: torch.Tensor, src: int | None = None, group: ProcessGroup | None = None, async_op: bool = False, group_src: int | None = None):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Args:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank on global process group (regardless of ``group`` argument).
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_src (int): Source rank on ``group``.  Must specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
@_exception_logger
def all_reduce(tensor, op=..., group=None, async_op: bool = False):
    '''
    Reduces the tensor data across all machines in a way that all get the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("no rank")
        >>> # All tensors below are of torch.int64 type.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device=\'cuda:0\') # Rank 0
        tensor([3, 4], device=\'cuda:1\') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4, 6], device=\'cuda:0\') # Rank 0
        tensor([4, 6], device=\'cuda:1\') # Rank 1

        >>> # All tensors below are of torch.cfloat type.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device=\'cuda:0\') # Rank 0
        tensor([3.+3.j, 4.+4.j], device=\'cuda:1\') # Rank 1
        >>> dist.all_reduce(tensor, op=ReduceOp.SUM)
        >>> tensor
        tensor([4.+4.j, 6.+6.j], device=\'cuda:0\') # Rank 0
        tensor([4.+4.j, 6.+6.j], device=\'cuda:1\') # Rank 1

    '''
@_exception_logger
def all_reduce_coalesced(tensors, op=..., group=None, async_op: bool = False):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.

    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Complex tensors are supported.

    Args:
        tensors (Union[List[Tensor], Tensor]): Input and output of the collective.
            The function operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
@_exception_logger
def reduce(tensor: torch.Tensor, dst: int | None = None, op=..., group: ProcessGroup | None = None, async_op: bool = False, group_dst: int | None = None):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Args:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank on global process group (regardless of ``group`` argument)
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_dst (int): Destination rank on ``group``.  Must specify one of ``group_dst``
            and ``dst`` but not both.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
@_exception_logger
def all_gather_object(object_list, obj, group=None) -> None:
    '''
    Gathers picklable objects from the whole group into a list.

    Similar to :func:`all_gather`, but Python objects can be passed in.
    Note that the object must be picklable in order to be gathered.

    Args:
        object_list (list[Any]): Output list. It should be correctly sized as the
            size of the group for this collective and will contain the output.
        obj (Any): Pickable Python object to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.

    Returns:
        None. If the calling rank is part of this group, the output of the
        collective will be populated into the input ``object_list``. If the
        calling rank is not part of the group, the passed in ``object_list`` will
        be unmodified.

    .. note:: Note that this API differs slightly from the :func:`all_gather`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user\'s responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`all_gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`all_gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`all_gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.all_gather_object(output, gather_objects[dist.get_rank()])
        >>> output
        [\'foo\', 12, {1: 2}]
    '''
@_exception_logger
def gather_object(obj: Any, object_gather_list: list[Any] | None = None, dst: int | None = None, group: ProcessGroup | None = None, group_dst: int | None = None):
    '''
    Gathers picklable objects from the whole group in a single process.

    Similar to :func:`gather`, but Python objects can be passed in. Note that the
    object must be picklable in order to be gathered.

    Args:
        obj (Any): Input object. Must be picklable.
        object_gather_list (list[Any]): Output list. On the ``dst`` rank, it
            should be correctly sized as the size of the group for this
            collective and will contain the output. Must be ``None`` on non-dst
            ranks. (default is ``None``)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument).
            (If both ``dst`` and ``group_dst`` are None, default is global rank 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        None. On the ``dst`` rank, ``object_gather_list`` will contain the
        output of the collective.

    .. note:: Note that this API differs slightly from the gather collective
        since it does not provide an async_op handle and thus will be a blocking
        call.

    .. note:: For NCCL-based processed groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user\'s responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`gather_object` uses ``pickle`` module implicitly, which is
        known to be insecure. It is possible to construct malicious pickle data
        which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`gather_object` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`gather` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes world_size of 3.
        >>> gather_objects = ["foo", 12, {1: 2}] # any picklable object
        >>> output = [None for _ in gather_objects]
        >>> dist.gather_object(
        ...     gather_objects[dist.get_rank()],
        ...     output if dist.get_rank() == 0 else None,
        ...     dst=0
        ... )
        >>> # On rank 0
        >>> output
        [\'foo\', 12, {1: 2}]
    '''
@_exception_logger
def send_object_list(object_list: list[Any], dst: int | None = None, group: ProcessGroup | None = None, device: torch.device | None = None, group_dst: int | None = None):
    '''
    Sends picklable objects in ``object_list`` synchronously.

    Similar to :func:`send`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    sent.

    Args:
        object_list (List[Any]): List of input objects to sent.
            Each object must be picklable. Receiver must provide lists of equal sizes.
        dst (int): Destination rank to send ``object_list`` to.
            Destination rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before sending. Default is ``None``.
        group_dst (int, optional): Destination rank on ``group``.
            Must specify one of ``dst`` and ``group_dst`` but not both
    Returns:
        ``None``.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user\'s responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`send_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`send_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`send` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     dist.send_object_list(objects, dst=1, device=device)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     dist.recv_object_list(objects, src=0, device=device)
        >>> objects
        [\'foo\', 12, {1: 2}]
    '''
@_exception_logger
def recv_object_list(object_list: list[Any], src: int | None = None, group: ProcessGroup | None = None, device: torch.device | None = None, group_src: int | None = None):
    '''
    Receives picklable objects in ``object_list`` synchronously.

    Similar to :func:`recv`, but can receive Python objects.

    Args:
        object_list (List[Any]): List of objects to receive into.
            Must provide a list of sizes equal to the size of the list being sent.
        src (int, optional): Source rank from which to recv ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
            Will receive from any rank if set to None. Default is ``None``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, receives on this device.
            Default is ``None``.
        group_src (int, optional): Destination rank on ``group``.  Invalid to specify both ``src`` and ``group_src``.

    Returns:
        Sender rank. -1 if rank is not part of the group. If rank is part of the group,
        ``object_list`` will contain the sent objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user\'s responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`recv_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`recv_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`recv` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>>     dist.send_object_list(objects, dst=1, device=device)
        >>> else:
        >>>     objects = [None, None, None]
        >>>     dist.recv_object_list(objects, src=0, device=device)
        >>> objects
        [\'foo\', 12, {1: 2}]
    '''
@_exception_logger
def broadcast_object_list(object_list: list[Any], src: int | None = None, group: ProcessGroup | None = None, device: torch.device | None = None, group_src: int | None = None):
    '''
    Broadcasts picklable objects in ``object_list`` to the whole group.

    Similar to :func:`broadcast`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    broadcasted.

    Args:
        object_list (List[Any]): List of input objects to broadcast.
            Each object must be picklable. Only objects on the ``src`` rank will
            be broadcast, but each rank must provide lists of equal sizes.
        src (int): Source rank from which to broadcast ``object_list``.
            Source rank is based on global process group (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before broadcasting. Default is ``None``.
        group_src (int): Source rank on ``group``.  Must not specify one of ``group_src``
            and ``src`` but not both.

    Returns:
        ``None``. If rank is part of the group, ``object_list`` will contain the
        broadcasted objects from ``src`` rank.

    .. note:: For NCCL-based process groups, internal tensor representations
        of objects must be moved to the GPU device before communication takes
        place. In this case, the device used is given by
        ``torch.cuda.current_device()`` and it is the user\'s responsibility to
        ensure that this is set so that each rank has an individual GPU, via
        ``torch.cuda.set_device()``.

    .. note:: Note that this API differs slightly from the :func:`broadcast`
        collective since it does not provide an ``async_op`` handle and thus
        will be a blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`broadcast_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`broadcast_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`broadcast` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     objects = [None, None, None]
        >>> # Assumes backend is not NCCL
        >>> device = torch.device("cpu")
        >>> dist.broadcast_object_list(objects, src=0, device=device)
        >>> objects
        [\'foo\', 12, {1: 2}]
    '''
@_exception_logger
def scatter_object_list(scatter_object_output_list: list[Any], scatter_object_input_list: list[Any] | None = None, src: int | None = None, group: ProcessGroup | None = None, group_src: int | None = None):
    '''
    Scatters picklable objects in ``scatter_object_input_list`` to the whole group.

    Similar to :func:`scatter`, but Python objects can be passed in. On
    each rank, the scattered object will be stored as the first element of
    ``scatter_object_output_list``. Note that all objects in
    ``scatter_object_input_list`` must be picklable in order to be scattered.

    Args:
        scatter_object_output_list (List[Any]): Non-empty list whose first
            element will store the object scattered to this rank.
        scatter_object_input_list (List[Any], optional): List of input objects to scatter.
            Each object must be picklable. Only objects on the ``src`` rank will
            be scattered, and the argument can be ``None`` for non-src ranks.
        src (int): Source rank from which to scatter ``scatter_object_input_list``.
            Source rank is based on global process group (regardless of ``group`` argument).
            (If both ``src`` and ``group_src`` are None, default is global rank 0)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        group_src (int, optional): Source rank on ``group``.  Invalid to specify both ``src`` and ``group_src``

    Returns:
        ``None``. If rank is part of the group, ``scatter_object_output_list``
        will have its first element set to the scattered object for this rank.

    .. note:: Note that this API differs slightly from the scatter collective
        since it does not provide an ``async_op`` handle and thus will be a
        blocking call.

    .. warning::
        Object collectives have a number of serious performance and scalability
        limitations.  See :ref:`object_collectives` for details.

    .. warning::
        :func:`scatter_object_list` uses ``pickle`` module implicitly, which
        is known to be insecure. It is possible to construct malicious pickle
        data which will execute arbitrary code during unpickling. Only call this
        function with data you trust.

    .. warning::
        Calling :func:`scatter_object_list` with GPU tensors is not well supported
        and inefficient as it incurs GPU -> CPU transfer since tensors would be
        pickled. Please consider using :func:`scatter` instead.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 3.
        >>>     objects = ["foo", 12, {1: 2}] # any picklable object
        >>> else:
        >>>     # Can be any list on non-src ranks, elements are not used.
        >>>     objects = [None, None, None]
        >>> output_list = [None]
        >>> dist.scatter_object_list(output_list, objects, src=0)
        >>> # Rank i gets objects[i]. For example, on rank 2:
        >>> output_list
        [{1: 2}]
    '''
@_exception_logger
def all_gather(tensor_list, tensor, group=None, async_op: bool = False):
    '''
    Gathers tensors from the whole group in a list.

    Complex and uneven sized tensors are supported.

    Args:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
            Uneven sized tensors are supported.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.int64, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0, 0], device=\'cuda:0\'), tensor([0, 0], device=\'cuda:0\')] # Rank 0
        [tensor([0, 0], device=\'cuda:1\'), tensor([0, 0], device=\'cuda:1\')] # Rank 1
        >>> tensor = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor
        tensor([1, 2], device=\'cuda:0\') # Rank 0
        tensor([3, 4], device=\'cuda:1\') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1, 2], device=\'cuda:0\'), tensor([3, 4], device=\'cuda:0\')] # Rank 0
        [tensor([1, 2], device=\'cuda:1\'), tensor([3, 4], device=\'cuda:1\')] # Rank 1

        >>> # All tensors below are of torch.cfloat dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_list = [
        ...     torch.zeros(2, dtype=torch.cfloat, device=device) for _ in range(2)
        ... ]
        >>> tensor_list
        [tensor([0.+0.j, 0.+0.j], device=\'cuda:0\'), tensor([0.+0.j, 0.+0.j], device=\'cuda:0\')] # Rank 0
        [tensor([0.+0.j, 0.+0.j], device=\'cuda:1\'), tensor([0.+0.j, 0.+0.j], device=\'cuda:1\')] # Rank 1
        >>> tensor = torch.tensor(
        ...     [1 + 1j, 2 + 2j], dtype=torch.cfloat, device=device
        ... ) + 2 * rank * (1 + 1j)
        >>> tensor
        tensor([1.+1.j, 2.+2.j], device=\'cuda:0\') # Rank 0
        tensor([3.+3.j, 4.+4.j], device=\'cuda:1\') # Rank 1
        >>> dist.all_gather(tensor_list, tensor)
        >>> tensor_list
        [tensor([1.+1.j, 2.+2.j], device=\'cuda:0\'), tensor([3.+3.j, 4.+4.j], device=\'cuda:0\')] # Rank 0
        [tensor([1.+1.j, 2.+2.j], device=\'cuda:1\'), tensor([3.+3.j, 4.+4.j], device=\'cuda:1\')] # Rank 1

    '''
@_exception_logger
def all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op: bool = False):
    '''
    Gather tensors from all ranks and put them in a single output tensor.

    This function requires all tensors to be the same size on each process.

    Args:
        output_tensor (Tensor): Output tensor to accommodate tensor elements
            from all ranks. It must be correctly sized to have one of the
            following forms:
            (i) a concatenation of all the input tensors along the primary
            dimension; for definition of "concatenation", see ``torch.cat()``;
            (ii) a stack of all the input tensors along the primary dimension;
            for definition of "stack", see ``torch.stack()``.
            Examples below may better explain the supported output forms.
        input_tensor (Tensor): Tensor to be gathered from current rank.
            Different from the ``all_gather`` API, the input tensors in this
            API must have the same size across all ranks.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_in = torch.arange(2, dtype=torch.int64, device=device) + 1 + 2 * rank
        >>> tensor_in
        tensor([1, 2], device=\'cuda:0\') # Rank 0
        tensor([3, 4], device=\'cuda:1\') # Rank 1
        >>> # Output in concatenation form
        >>> tensor_out = torch.zeros(world_size * 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([1, 2, 3, 4], device=\'cuda:0\') # Rank 0
        tensor([1, 2, 3, 4], device=\'cuda:1\') # Rank 1
        >>> # Output in stack form
        >>> tensor_out2 = torch.zeros(world_size, 2, dtype=torch.int64, device=device)
        >>> dist.all_gather_into_tensor(tensor_out2, tensor_in)
        >>> tensor_out2
        tensor([[1, 2],
                [3, 4]], device=\'cuda:0\') # Rank 0
        tensor([[1, 2],
                [3, 4]], device=\'cuda:1\') # Rank 1
    '''
@_exception_logger
def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=None, async_op: bool = False):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Complex tensors are supported.

    Args:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        we have 2 process groups, 2 ranks.
        rank 0 passes:
            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        rank 1 passes:
            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        both rank 0 and 1 get:
            output_tensor_lists =
               [[[1, 1], [1, 1]], [2], [3, 3]],
                [[3, 3], [3, 3]], [5], [1, 1]]].

    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the
    all_gather_coalesced operation will proceed without complaint and return
    erroneous outputs. This lack of shape checking results in significant
    performance improvements but users of this function should take extra care
    to ensure that each node passes in tensors whose shapes match across nodes.
    """
@_exception_logger
def gather(tensor: torch.Tensor, gather_list: list[torch.Tensor] | None = None, dst: int | None = None, group: ProcessGroup | None = None, async_op: bool = False, group_dst: int | None = None):
    '''
    Gathers a list of tensors in a single process.

    This function requires all tensors to be the same size on each process.

    Args:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately,
            same-sized tensors to use for gathered data
            (default is None, must be specified on the destination rank)
        dst (int, optional): Destination rank on global process group (regardless of ``group`` argument).
            (If both ``dst`` and ``group_dst`` are None, default is global rank 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_dst (int, optional): Destination rank on ``group``.  Invalid to specify both ``dst`` and ``group_dst``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in gather_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("no rank")
        >>> # We have 2 process groups, 2 ranks.
        >>> tensor_size = 2
        >>> device = torch.device(f\'cuda:{rank}\')
        >>> tensor = torch.ones(tensor_size, device=device) + rank
        >>> if dist.get_rank() == 0:
        >>>     gather_list = [torch.zeros_like(tensor, device=device) for i in range(2)]
        >>> else:
        >>>     gather_list = None
        >>> dist.gather(tensor, gather_list, dst=0)
        >>> # Rank 0 gets gathered data.
        >>> gather_list
        [tensor([1., 1.], device=\'cuda:0\'), tensor([2., 2.], device=\'cuda:0\')] # Rank 0
        None                                                                   # Rank 1

    '''
@_exception_logger
def scatter(tensor: torch.Tensor, scatter_list: list[torch.Tensor] | None = None, src: int | None = None, group: ProcessGroup | None = None, async_op: bool = False, group_src: int | None = None):
    '''
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Complex tensors are supported.

    Args:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank on global process group (regardless of ``group`` argument).
            (If both ``src`` and ``group_src`` are None, default is global rank 0)
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        group_src (int, optional): Source rank on ``group``.  Invalid to specify both ``src`` and ``group_src``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: Note that all Tensors in scatter_list must have the same size.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> tensor_size = 2
        >>> device = torch.device(f\'cuda:{rank}\')
        >>> output_tensor = torch.zeros(tensor_size, device=device)
        >>> if dist.get_rank() == 0:
        >>>     # Assumes world_size of 2.
        >>>     # Only tensors, all of which must be the same size.
        >>>     t_ones = torch.ones(tensor_size, device=device)
        >>>     t_fives = torch.ones(tensor_size, device=device) * 5
        >>>     scatter_list = [t_ones, t_fives]
        >>> else:
        >>>     scatter_list = None
        >>> dist.scatter(output_tensor, scatter_list, src=0)
        >>> # Rank i gets scatter_list[i].
        >>> output_tensor
        tensor([1., 1.], device=\'cuda:0\') # Rank 0
        tensor([5., 5.], device=\'cuda:1\') # Rank 1

    '''
@_exception_logger
def reduce_scatter(output, input_list, op=..., group=None, async_op: bool = False):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Args:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
@_exception_logger
def reduce_scatter_tensor(output, input, op=..., group=None, async_op: bool = False):
    '''
    Reduces, then scatters a tensor to all ranks in a group.

    Args:
        output (Tensor): Output tensor. It should have the same size across all
            ranks.
        input (Tensor): Input tensor to be reduced and scattered. Its size
            should be output tensor size times the world size. The input tensor
            can have one of the following shapes:
            (i) a concatenation of the output tensors along the primary
            dimension, or
            (ii) a stack of the output tensors along the primary dimension.
            For definition of "concatenation", see ``torch.cat()``.
            For definition of "stack", see ``torch.stack()``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    Examples:
        >>> # xdoctest: +SKIP("need process group init")
        >>> # All tensors below are of torch.int64 dtype and on CUDA devices.
        >>> # We have two ranks.
        >>> device = torch.device(f"cuda:{rank}")
        >>> tensor_out = torch.zeros(2, dtype=torch.int64, device=device)
        >>> # Input in concatenation form
        >>> tensor_in = torch.arange(world_size * 2, dtype=torch.int64, device=device)
        >>> tensor_in
        tensor([0, 1, 2, 3], device=\'cuda:0\') # Rank 0
        tensor([0, 1, 2, 3], device=\'cuda:1\') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device=\'cuda:0\') # Rank 0
        tensor([4, 6], device=\'cuda:1\') # Rank 1
        >>> # Input in stack form
        >>> tensor_in = torch.reshape(tensor_in, (world_size, 2))
        >>> tensor_in
        tensor([[0, 1],
                [2, 3]], device=\'cuda:0\') # Rank 0
        tensor([[0, 1],
                [2, 3]], device=\'cuda:1\') # Rank 1
        >>> dist.reduce_scatter_tensor(tensor_out, tensor_in)
        >>> tensor_out
        tensor([0, 2], device=\'cuda:0\') # Rank 0
        tensor([4, 6], device=\'cuda:1\') # Rank 1

    '''
@_exception_logger
def all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op: bool = False):
    '''
    Split input tensor and then scatter the split list to all processes in a group.

    Later the received tensors are concatenated from all the processes in the group
    and returned as a single output tensor.

    Complex tensors are supported.

    Args:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all_single` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input
        tensor([0, 1, 2, 3])     # Rank 0
        tensor([4, 5, 6, 7])     # Rank 1
        tensor([8, 9, 10, 11])   # Rank 2
        tensor([12, 13, 14, 15]) # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([0, 4, 8, 12])    # Rank 0
        tensor([1, 5, 9, 13])    # Rank 1
        tensor([2, 6, 10, 14])   # Rank 2
        tensor([3, 7, 11, 15])   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = list(input.chunk(world_size))
        >>> gather_list = list(output.chunk(world_size))
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src = i)

        >>> # Another example with uneven split
        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> output = ...
        >>> dist.all_to_all_single(output, input, output_splits, input_splits)
        >>> output
        tensor([ 0,  1, 10, 11, 12, 20, 21, 30, 31])                     # Rank 0
        tensor([ 2,  3, 13, 14, 22, 32, 33])                             # Rank 1
        tensor([ 4, 15, 16, 23, 34, 35])                                 # Rank 2
        tensor([ 5, 17, 18, 24, 36])                                     # Rank 3


        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input
        tensor([1+1j, 2+2j, 3+3j, 4+4j])                                # Rank 0
        tensor([5+5j, 6+6j, 7+7j, 8+8j])                                # Rank 1
        tensor([9+9j, 10+10j, 11+11j, 12+12j])                          # Rank 2
        tensor([13+13j, 14+14j, 15+15j, 16+16j])                        # Rank 3
        >>> output = torch.empty([4], dtype=torch.int64)
        >>> dist.all_to_all_single(output, input)
        >>> output
        tensor([1+1j, 5+5j, 9+9j, 13+13j])                              # Rank 0
        tensor([2+2j, 6+6j, 10+10j, 14+14j])                            # Rank 1
        tensor([3+3j, 7+7j, 11+11j, 15+15j])                            # Rank 2
        tensor([4+4j, 8+8j, 12+12j, 16+16j])                            # Rank 3
    '''
@_exception_logger
def all_to_all(output_tensor_list, input_tensor_list, group=None, async_op: bool = False):
    '''
    Scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Complex tensors are supported.

    Args:
        output_tensor_list (list[Tensor]): List of tensors to be gathered one
            per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    .. warning::
        `all_to_all` is experimental and subject to change.

    Examples:
        >>> # xdoctest: +SKIP("Undefined rank")
        >>> input = torch.arange(4) + rank * 4
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([0]), tensor([1]), tensor([2]), tensor([3])]     # Rank 0
        [tensor([4]), tensor([5]), tensor([6]), tensor([7])]     # Rank 1
        [tensor([8]), tensor([9]), tensor([10]), tensor([11])]   # Rank 2
        [tensor([12]), tensor([13]), tensor([14]), tensor([15])] # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0]), tensor([4]), tensor([8]), tensor([12])]    # Rank 0
        [tensor([1]), tensor([5]), tensor([9]), tensor([13])]    # Rank 1
        [tensor([2]), tensor([6]), tensor([10]), tensor([14])]   # Rank 2
        [tensor([3]), tensor([7]), tensor([11]), tensor([15])]   # Rank 3

        >>> # Essentially, it is similar to following operation:
        >>> scatter_list = input
        >>> gather_list = output
        >>> for i in range(world_size):
        >>>     dist.scatter(gather_list[i], scatter_list if i == rank else [], src=i)

        >>> input
        tensor([0, 1, 2, 3, 4, 5])                                       # Rank 0
        tensor([10, 11, 12, 13, 14, 15, 16, 17, 18])                     # Rank 1
        tensor([20, 21, 22, 23, 24])                                     # Rank 2
        tensor([30, 31, 32, 33, 34, 35, 36])                             # Rank 3
        >>> input_splits
        [2, 2, 1, 1]                                                     # Rank 0
        [3, 2, 2, 2]                                                     # Rank 1
        [2, 1, 1, 1]                                                     # Rank 2
        [2, 2, 2, 1]                                                     # Rank 3
        >>> output_splits
        [2, 3, 2, 2]                                                     # Rank 0
        [2, 2, 1, 2]                                                     # Rank 1
        [1, 2, 1, 2]                                                     # Rank 2
        [1, 2, 1, 1]                                                     # Rank 3
        >>> input = list(input.split(input_splits))
        >>> input
        [tensor([0, 1]), tensor([2, 3]), tensor([4]), tensor([5])]                   # Rank 0
        [tensor([10, 11, 12]), tensor([13, 14]), tensor([15, 16]), tensor([17, 18])] # Rank 1
        [tensor([20, 21]), tensor([22]), tensor([23]), tensor([24])]                 # Rank 2
        [tensor([30, 31]), tensor([32, 33]), tensor([34, 35]), tensor([36])]         # Rank 3
        >>> output = ...
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([0, 1]), tensor([10, 11, 12]), tensor([20, 21]), tensor([30, 31])]   # Rank 0
        [tensor([2, 3]), tensor([13, 14]), tensor([22]), tensor([32, 33])]           # Rank 1
        [tensor([4]), tensor([15, 16]), tensor([23]), tensor([34, 35])]              # Rank 2
        [tensor([5]), tensor([17, 18]), tensor([24]), tensor([36])]                  # Rank 3

        >>> # Another example with tensors of torch.cfloat type.
        >>> input = torch.tensor(
        ...     [1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=torch.cfloat
        ... ) + 4 * rank * (1 + 1j)
        >>> input = list(input.chunk(4))
        >>> input
        [tensor([1+1j]), tensor([2+2j]), tensor([3+3j]), tensor([4+4j])]            # Rank 0
        [tensor([5+5j]), tensor([6+6j]), tensor([7+7j]), tensor([8+8j])]            # Rank 1
        [tensor([9+9j]), tensor([10+10j]), tensor([11+11j]), tensor([12+12j])]      # Rank 2
        [tensor([13+13j]), tensor([14+14j]), tensor([15+15j]), tensor([16+16j])]    # Rank 3
        >>> output = list(torch.empty([4], dtype=torch.int64).chunk(4))
        >>> dist.all_to_all(output, input)
        >>> output
        [tensor([1+1j]), tensor([5+5j]), tensor([9+9j]), tensor([13+13j])]          # Rank 0
        [tensor([2+2j]), tensor([6+6j]), tensor([10+10j]), tensor([14+14j])]        # Rank 1
        [tensor([3+3j]), tensor([7+7j]), tensor([11+11j]), tensor([15+15j])]        # Rank 2
        [tensor([4+4j]), tensor([8+8j]), tensor([12+12j]), tensor([16+16j])]        # Rank 3

    '''
@_exception_logger
def barrier(group: ProcessGroup | None = ..., async_op: bool = False, device_ids=None):
    """
    Synchronize all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        async_op (bool, optional): Whether this op should be an async op
        device_ids ([int], optional): List of device/GPU ids. Only one id is expected.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    .. note:: `ProcessGroupNCCL` now blocks the cpu thread till the completion of the barrier collective.
    """
def monitored_barrier(group: ProcessGroup | None = ..., timeout=None, wait_all_ranks: bool = False):
    '''
    Synchronize processes similar to ``torch.distributed.barrier``, but consider a configurable timeout.

    It is able to report ranks that did not pass this barrier within the provided timeout.
    Specifically, for non-zero ranks, will block until a send/recv is processed from rank 0.
    Rank 0 will block until all send /recv from other ranks are processed, and will report
    failures for ranks that failed to respond in time. Note that if one rank does not reach the
    monitored_barrier (for example due to a hang), all other ranks would fail in monitored_barrier.

    This collective will block all processes/ranks in the group, until the
    whole group exits the function successfully, making it useful for debugging
    and synchronizing. However, it can have a performance impact and should only
    be used for debugging or scenarios that require full synchronization points
    on the host-side. For debugging purposes, this barrier can be inserted
    before the application\'s collective calls to check if any ranks are
    desynchronized.

    .. note:: Note that this collective is only supported with the GLOO backend.

    Args:
        group (ProcessGroup, optional): The process group to work on. If
            ``None``, the default process group will be used.
        timeout (datetime.timedelta, optional): Timeout for monitored_barrier.
            If ``None``, the default process group timeout will be used.
        wait_all_ranks (bool, optional): Whether to collect all failed ranks or
            not. By default, this is ``False`` and ``monitored_barrier`` on rank 0
            will throw on the first failed rank it encounters in order to fail
            fast. By setting ``wait_all_ranks=True`` ``monitored_barrier`` will
            collect all failed ranks and throw an error containing information
            about all failed ranks.

    Returns:
        ``None``.

    Example::
        >>> # xdoctest: +SKIP("need process group init")
        >>> # Note: Process group initialization omitted on each rank.
        >>> import torch.distributed as dist
        >>> if dist.get_rank() != 1:
        >>>     dist.monitored_barrier() # Raises exception indicating that
        >>> # rank 1 did not call into monitored_barrier.
        >>> # Example with wait_all_ranks=True
        >>> if dist.get_rank() == 0:
        >>>     dist.monitored_barrier(wait_all_ranks=True) # Raises exception
        >>> # indicating that ranks 1, 2, ... world_size - 1 did not call into
        >>> # monitored_barrier.
    '''
@_time_logger
def split_group(parent_pg: ProcessGroup | None = None, split_ranks: list | None = None, timeout: timedelta | None = None, pg_options: Any | None = None, group_desc: str | None = None) -> ProcessGroup | None:
    """
    Create a new process group split from the given parent process group.

    warning:: This is an experimental API. Only the ``NCCL`` and custom plugin backends
    are supported. Other backends will raise an error.
    Users of this API must guarantee that all ranks in the parent group enter this API call,
    and the split of the sub groups is the same across all ranks in the parent group.

    Args:
        parent_pg (ProcessGroup, optional): The parent process group. If None,
            the default process group will be used. Users need to guarantee that
            the parent group is fully initialized (e.g, communicators are initialized)
        split_ranks (list[list[int]]): the split ranks, which is a list of list of ranks.
            Users need to make sure the validity of the split ranks such that one
            split (represented by one inner list of ints) does not overlap with any other split.
            Note that the ranks in each split is the group rank (instead of global rank)
            in the parent pg. For example, if the parent group has 4 ranks, and split_ranks can be
            [[0, 1], [2, 3]]. Note [[0,1]] is also a valid split, in which case ranks 2, 3 would
            return a non-group member.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        pg_options (ProcessGroupOptions, optional): Additional options need to be passed in during
            the construction of specific process groups. i.e.``is_high_priority_stream``
            can be specified so that process group can pick up high priority cuda streams.
        group_desc (str, optional): a string to describe the process group.

    Returns:
        ProcessGroup if the current rank is within one split/subgroup given by split_ranks,
        or None if the current rank is not part of any split_ranks`.

    """
@_time_logger
def new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization: bool = False, group_desc=None, device_id: torch.device | None = None):
    '''
    Create a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    .. warning::
        Safe concurrent usage:
        When using multiple process groups with the ``NCCL`` backend, the user
        must ensure a globally consistent execution order of collectives across
        ranks.

        If multiple threads within a process issue collectives, explicit
        synchronization is necessary to ensure consistent ordering.

        When using async variants of torch.distributed communication APIs,
        a work object is returned and the communication kernel is
        enqueued on a separate CUDA stream, allowing overlap of communication
        and computation. Once one or more async ops have been issued on one process
        group, they must be synchronized with other cuda streams by calling `work.wait()`
        before using another process group.

        See `Using multiple NCCL communicators concurrently
        <https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently>`
        for more details.

    Args:
        ranks (list[int]): List of ranks of group members. If ``None``, will be
            set to all ranks. Default is ``None``.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``). If ``None`` is passed in, the backend
            corresponding to the default process group will be used. Default is
            ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams. For other available options to config nccl,
            See https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#ncclconfig-tuse_local_synchronization
            (bool, optional): perform a group-local barrier at the end of the process group creation.
            This is different in that non-member ranks don\'t need to call into API and don\'t
            join the barrier.
        group_desc (str, optional): a string to describe the process group.
        device_id (torch.device, optional): a single, specific device
            to "bind" this process to,  The `new_group` call will try to initialize
            a communication backend immediately for the device if this field is given.

    Returns:
        A handle of distributed group that can be given to collective calls or
        GroupMember.NON_GROUP_MEMBER if the rank is not part of ``ranks``.

    N.B. use_local_synchronization doesn\'t work with MPI.

    N.B. While use_local_synchronization=True can be significantly faster with larger
    clusters and small process groups, care must be taken since it changes cluster behavior
    as non-member ranks don\'t join the group barrier().

    N.B. use_local_synchronization=True can lead to deadlocks when each rank creates
    multiple overlapping process groups. To avoid that, make sure all ranks follow the
    same global creation order.
    '''
def new_subgroups(group_size=None, group=None, timeout=None, backend=None, pg_options=None, group_desc=None):
    '''
    Create subgroups of equal size.

    By default, it creates intra-machine subgroups,
    where each of which contains all the ranks of a machine, based on the assumption
    that each machine has the same number of devices.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        If ``group_size`` is passed in, the world size must be divisible by ``group_size``.
        If no ``group_size`` is passed in, it believe that you are creating a group based
        on CUDA and determining the group size by number of CUDA devices, and if not all
        the machines have the same number of devices, the subgroup division will be
        different across nodes and can cause unexpected behaviors. Therefore, if you are
        creating a subgroup that does not depend on CUDA (such as Gloo on CPU), please
        pass in ``group_size`` correctly.

    .. warning::
        See warning `Safe concurrent usage` for `new_group` API for important details about
        using multiple process groups concurrently in a safe manner.

    Args:
        group_size (int, optional): The size of each subgroup. If ``None``,
            the default subgroup size is equal to the number of devices on each machine,
            based on the assumption that each machine has exactly the same
            number of devices. Default is ``None``.
        group (ProcessGroup, optional): The process group to work on. If
            ``None``, the default process group will be used. Default is ``None``.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``). If ``None`` is passed in, the backend
            corresponding to the default process group will be used. Default is
            ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams.
        group_desc (str, optional): A string describing the group. Each subgroup will
            inherit its group_desc

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.

    Examples:
        >>> # Create intra-machine subgroups.
        >>> # xdoctest: +SKIP("need process group init")
        >>> cur_subgroup, subgroups = dist.new_subgroups()
        >>> # Allreduce within the machine.
        >>> rank = dist.get_rank()
        >>> tensor = torch.ones(1, device=rank) * rank
        >>> dist.all_reduce(tensor, group=cur_subgroup)
        >>> tensor
        tensor([28])  # Assume 8 CUDA devices per machine.  28 is sum(range(8)).
        >>> # Cleanup.
        >>> for subgroup in subgroups:
        >>>     dist.destroy_process_group(subgroup)
    '''
def new_subgroups_by_enumeration(ranks_per_subgroup_list, timeout=None, backend=None, pg_options=None, group_desc=None):
    '''
    Create subgroups by dividing the global world.

    The division is specified by a nested list of ranks. The subgroups cannot have
    overlap, and some ranks may not have to be in any subgroup.

    This is a convenience API that calls ``new_group`` to generate multiple subgroups.
    It requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group.

    .. warning::
        See warning `Safe concurrent usage` for `new_group` API for important details about
        using multiple process groups concurrently in a safe manner.

    Args:
        ranks_per_subgroup_list (list[list[int]]): A nested list of ranks of
            group members.
        timeout (timedelta, optional): see `init_process_group` for details and default value.
        backend (str or Backend, optional): The backend to use. Depending on
             build-time configurations, valid values are ``gloo`` and ``nccl``.
             By default uses the same backend as the global group. This field
             should be given as a lowercase string (e.g., ``"gloo"``), which can
             also be accessed via :class:`Backend` attributes (e.g.,
             ``Backend.GLOO``). If ``None`` is passed in, the backend
             corresponding to the default process group will be used. Default is
             ``None``.
        pg_options (ProcessGroupOptions, optional): process group options
            specifying what additional options need to be passed in during
            the construction of specific process groups. i.e. for the ``nccl``
            backend, ``is_high_priority_stream`` can be specified so that
            process group can pick up high priority cuda streams.
        group_desc (str, optional): A string describing the group. Each subgroup will
            inherit its group_desc.

    Returns:
        The subgroup containing the current rank, and all the subgroups used for cleanup.

    Examples:
        >>> # Create two subgroups, where each has 2 processes.
        >>> # xdoctest: +SKIP("need process group init")
        >>> cur_subgroup, subgroups = dist.new_subgroups(ranks=[[0, 2], [1, 3]])
        >>> rank = dist.get_rank()
        >>> tensor = torch.ones(1, device=rank) * rank
        >>> dist.all_reduce(tensor, group=cur_subgroup)
        >>> tensor
        tensor([2])     # Subgroup 0: ranks 0 and 2
        tensor([4])     # Subgroup 1: ranks 1 and 3
    '''
