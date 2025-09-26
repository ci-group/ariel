from .distributed_c10d import *
import pdb
import torch
import typing
from .device_mesh import DeviceMesh as DeviceMesh, init_device_mesh as init_device_mesh
from .distributed_c10d import _CoalescingManager as _CoalescingManager, _all_gather_base as _all_gather_base, _coalescing_manager as _coalescing_manager, _create_process_group_wrapper as _create_process_group_wrapper, _get_process_group_name as _get_process_group_name, _rank_not_in_group as _rank_not_in_group, _reduce_scatter_base as _reduce_scatter_base, _time_estimator as _time_estimator, get_node_local_rank as get_node_local_rank
from .remote_device import _remote_device as _remote_device
from .rendezvous import _create_store_from_options as _create_store_from_options, register_rendezvous_handler as register_rendezvous_handler, rendezvous as rendezvous
from _typeshed import Incomplete
from torch._C._distributed_c10d import BuiltinCommHookType as BuiltinCommHookType, DebugLevel as DebugLevel, FileStore as FileStore, GradBucket as GradBucket, HashStore as HashStore, Logger as Logger, PrefixStore as PrefixStore, Reducer as Reducer, Store as Store, TCPStore as TCPStore, _ControlCollectives as _ControlCollectives, _DEFAULT_FIRST_BUCKET_BYTES as _DEFAULT_FIRST_BUCKET_BYTES, _StoreCollectives as _StoreCollectives, _broadcast_coalesced as _broadcast_coalesced, _compute_bucket_assignment_by_size as _compute_bucket_assignment_by_size, _make_nccl_premul_sum as _make_nccl_premul_sum, _register_builtin_comm_hook as _register_builtin_comm_hook, _register_comm_hook as _register_comm_hook, _test_python_store as _test_python_store, _verify_params_across_processes as _verify_params_across_processes, get_debug_level as get_debug_level, set_debug_level as set_debug_level, set_debug_level_from_env as set_debug_level_from_env

log: Incomplete

def is_available() -> bool:
    """
    Return ``True`` if the distributed package is available.

    Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
DistError = torch._C._DistError
DistBackendError = torch._C._DistBackendError
DistNetworkError = torch._C._DistNetworkError
DistStoreError = torch._C._DistStoreError
QueueEmptyError = torch._C._DistQueueEmptyError

class _DistributedPdb(pdb.Pdb):
    """
        Supports using PDB from inside a multiprocessing child process.

        Usage:
        _DistributedPdb().set_trace()
        """
    def interaction(self, *args, **kwargs) -> None: ...

_breakpoint_cache: dict[int, typing.Any]

def breakpoint(rank: int = 0, skip: int = 0):
    """
        Set a breakpoint, but only on a single rank.  All other ranks will wait for you to be
        done with the breakpoint before continuing.

        Args:
            rank (int): Which rank to break on.  Default: ``0``
            skip (int): Skip the first ``skip`` calls to this breakpoint. Default: ``0``.
        """

class _ProcessGroupStub: ...
