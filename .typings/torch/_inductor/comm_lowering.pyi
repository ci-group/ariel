from . import config as config, ir as ir
from .virtualized import V as V
from _typeshed import Incomplete
from torch._inductor.utils import is_symbolic as is_symbolic
from torch.utils._ordered_set import OrderedSet as OrderedSet

log: Incomplete

def can_realize_as_comm_buffer(x: ir.TensorBox, comm_buffer_type: ir.CommBufferType) -> bool:
    """
    Check if an input can be realized as a comm buffer of the specified
    `comm_buffer_type`.
    """
def realize_as_comm_buffer(x: ir.TensorBox, comm_buffer_type: ir.CommBufferType, group_name: str) -> None:
    """
    Realize an input as a comm buffer of the specified `comm_buffer_type`.

    Specifically, this realizes the underlying buffer if it's still unrealized
    and changes the layout of the buffer to `ir.CommBufferLayout`.
    """
def _get_data(x: ir.TensorBox) -> ir.IRNode: ...

_bufs_to_skip_wait: Incomplete

def mark_as_skip_wait(x: ir.IRNode) -> None:
    """
    If a non-blocking collective is lowered as a blocking collective, the wait
    node in the original graph becomes useless and we can skip the lowering it.
    """
def should_skip_wait(x: ir.IRNode) -> bool: ...
def _should_lower_as_one_shot_all_reduce(inp: ir.TensorBox, reduce_op: str, group_name: str): ...
def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name): ...
def register_comm_lowerings(): ...
