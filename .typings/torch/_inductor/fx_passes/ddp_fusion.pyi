import logging
import torch
import torch.fx as fx
from ..fx_utils import get_fake_args_kwargs as get_fake_args_kwargs
from ..virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Generator
from dataclasses import dataclass
from torch._dynamo.utils import counters as counters
from torch.fx.passes.graph_transform_observer import GraphTransformObserver as GraphTransformObserver
from torch.fx.passes.shape_prop import TensorMetadata as TensorMetadata, _extract_tensor_metadata as _extract_tensor_metadata
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._pytree import tree_flatten as tree_flatten, tree_map as tree_map, tree_unflatten as tree_unflatten
from typing import Any, Callable

aten: Incomplete
logger: logging.Logger

def move_block_after(block: list[fx.Node], target_node: fx.Node) -> None: ...
def move_block_before(block: list[fx.Node], target_node: fx.Node) -> None: ...
def call_function(graph: fx.Graph, target: str | Callable[..., Any], args: tuple[fx.node.Argument, ...] | None = None, kwargs: dict[str, fx.node.Argument] | None = None) -> fx.Node: ...

@dataclass(unsafe_hash=True)
class CommBlock:
    shape: torch.Size | list[torch.Size]
    node_list: list[fx.Node]
    inputs: list[fx.Node]
    wait_nodes: list[fx.Node]
    comm_node: fx.Node
    outputs: OrderedSet[fx.Node]

def get_comm_block(comm_node: fx.Node) -> CommBlock | None:
    """
    Given a collective node (e.g., allreduce), find out all the nodes belong to
    this communication.

    Args:
        comm_node(fx.Node): The target communication/collective node.
    Returns:
        The CommBlock that encapsulates the related nodes (e.g., wait_node) of
        the given comm_node.
    """
def get_all_comm_blocks(graph: fx.Graph, comm_ops: tuple[torch._ops.OpOverload, ...], comm_filter: Callable[..., bool] | None = None) -> list[CommBlock]: ...
def _fuse_allreduce_by_concat(graph: fx.Graph, last_input_node: fx.Node, all_input_nodes: list[fx.Node], last_comm_block: CommBlock) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce using concat."""
def _fuse_with_coalesced_op(graph: fx.Graph, last_input_node: fx.Node, all_input_nodes: list[fx.Node], last_comm_block: CommBlock) -> CommBlock:
    """Given a list of inputs in order, create a fused allreduce by coalesced."""
def _scatter_fused_allreduce_waits(graph: fx.Graph, fused_comm_block: CommBlock, orig_comm_blocks: list[CommBlock], node_indices: dict[fx.Node, int], split_and_reshape: bool = True) -> None:
    """
    Scatters the result of the fused communication node to the original users.
    If the fused method is concat splitting the output and reshape will be inserted,
    before inserting getitem. Otherwise getitem will be used as the users of the
    wait node.
    """
def _fuse_allreduce(graph: fx.Graph, comm_blocks: list[CommBlock], node_indices: dict[fx.Node, int], use_concat: bool) -> CommBlock:
    """Given a list of allreduce CommBlock, fuse the CommBlocks into one CommBlock."""
def _bucket_size_fusion(graph: fx.Graph, comm_blocks: list[CommBlock], bucket_size_mb: int) -> Generator[list[CommBlock], None, None]: ...
def _fuse_ddp_communication(graph: fx.Graph, algorithm_fn: Callable[..., Any], fusion_fn: Callable[..., Any]) -> None: ...
def fuse_ddp_with_coalesced_op(graph: fx.Graph, bucket_size_mb: int) -> None: ...
def fuse_ddp_with_concat_op(graph: fx.Graph, bucket_size_mb: int) -> None: ...
def schedule_comm_wait(graph: fx.Graph) -> None:
    """
    Delay the execution of wait tensors of allreduce until its first user.

    This algorithm considers the intermediate users, like split, getitem,
    of the wait node and schedule those intermediate users as well.
    This will result in a better overlapping result.
    """
def fuse_ddp_communication(graph: fx.Graph, passes: list[Callable[..., None] | str], bucket_size_mb: int) -> None: ...
