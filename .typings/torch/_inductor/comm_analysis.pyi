import functools
from . import ir as ir
from .utils import get_dtype_size as get_dtype_size, sympy_product as sympy_product
from .virtualized import V as V
from _typeshed import Incomplete
from enum import IntEnum

class NCCL_COLL(IntEnum):
    ALL_REDUCE = 0
    ALL_GATHER = 1
    REDUCE_SCATTER = 2

class NVIDIA_GPU_TYPE(IntEnum):
    VOLTA = 0
    AMPERE = 1
    HOPPER = 2

@functools.lru_cache
def get_gpu_type() -> NVIDIA_GPU_TYPE: ...
def get_collective_type(node: ir.IRNode) -> NCCL_COLL: ...
def get_collective_input_size_bytes(node: ir.IRNode) -> int: ...
def get_collective_group_size(node: ir.IRNode) -> int: ...

class NCCL_HW(IntEnum):
    NVLINK = 0
    PCI = 1
    NET = 2

class NCCL_ALGO(IntEnum):
    TREE = 0
    RING = 1

class NCCL_PROTO(IntEnum):
    LL = 0

baseLat: Incomplete
hwLat: Incomplete
llMaxBws: Incomplete

def estimate_nccl_collective_runtime(node: ir.IRNode) -> float:
    '''
    Returns estimated NCCL collective runtime in nanoseconds (ns).

    The following heuristics are copied from https://github.com/NVIDIA/nccl/blob/master/src/graph/tuning.cc.
    We aim to estimate the runtime as accurately as possible.

    Assumptions:
    - only ring algorithm (NCCL_ALGO_RING) is used
    - only Low-Latency protocol (NCCL_PROTO_LL) is used, i.e. Simple or LL128 is not used
    - 8 gpus per node  # TODO: Need to find a way to get accurate "gpus per node" and "# nodes" info.
    - collective is one of: allreduce, reducescatter, allgather
    '''
