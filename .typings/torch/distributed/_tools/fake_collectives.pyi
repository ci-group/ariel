import torch
from _typeshed import Incomplete
from torch._C._distributed_c10d import FakeWork as FakeWork, ProcessGroup as ProcessGroup, Work as Work, _resolve_process_group as _resolve_process_group
from torch.utils._pytree import tree_map_only as tree_map_only
from typing import Any

c10d: Incomplete
_c10d_functional: Incomplete
_c10d_functional_autograd: Incomplete
_dtensor: Incomplete
used_ids: set[int]

def generate_unique_id() -> int: ...
def create_fakework(args, return_first_arg: bool = True): ...

_META_FUNCTIONS: Incomplete
lib_impl: Incomplete
non_functional_collectives: set[torch._ops.OpOverload]
functional_collectives: set[torch._ops.OpOverload]
sync_ops: set[torch._ops.OpOverload]
collective_ops: Incomplete

class CollectiveOp:
    PG_ARG_1: Incomplete
    PG_ARG_2: Incomplete
    PG_ARG_3: Incomplete
    PG_ARG_4: Incomplete
    WK_ARG_1: Incomplete
    WK: Incomplete
    COMM_TENSOR_ARG_0: Incomplete
    COMM_TENSOR_ARG_1: Incomplete
    COMM_TENSOR_ARG_RES: Incomplete
    COMM_TENSOR_SINGLE_UNTYPED_STORAGE: Incomplete
    COMM_TENSOR_ARG_0_AND_RES: Incomplete
    COMM_TENSOR_RES_SUM: Incomplete
    @staticmethod
    def sum_tensors(arg: Any) -> int:
        """Calculate total memory consumed by the tensors in the argument."""
    @staticmethod
    def get_process_group(func, args) -> ProcessGroup:
        """Retrieve the process group for collective operations, except `wait_tensor`."""
    @staticmethod
    def get_comm_tensor_size(func, res, args, kwargs) -> int:
        """Compute the communication tensor size, except for `wait_tensor`, `barrier`, and `monitored_barrier`."""
    @staticmethod
    def get_work(func, res) -> Work: ...
