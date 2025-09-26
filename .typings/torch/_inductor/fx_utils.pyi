import torch
import torch.fx
from .virtualized import V as V
from _typeshed import Incomplete
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.fx.experimental.symbolic_shapes import compute_unbacked_bindings as compute_unbacked_bindings, rebind_unbacked as rebind_unbacked, statically_known_true as statically_known_true, sym_eq as sym_eq
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils._pytree import tree_map as tree_map
from torch.utils.flop_counter import flop_registry as flop_registry
from typing import Any, Callable

def matches_module_function_pattern(pattern: tuple[type[torch.nn.modules.Module], Callable[..., Any]], node: torch.fx.node.Node, modules: dict[str, torch.nn.modules.Module]) -> bool: ...

class FakeTensorUpdater:
    """
    The main idea here is that it's difficult to maintain accurate fake
    tensors (our primary form of metadata) for each node in our graph as we
    transform it.

    The most reliable way to obtain this information is by rerunning
    faketensor propagation. However, in general, faketensor propagation is
    fairly expensive. So, instead we'd like to only rerun faketensor
    propagation on nodes that have changed.

    In order to detect which nodes have changed, we first hash its node,
    target, and argument lists (which are immutable in FX).

    Then, whenever we call incremental_update, we check which FX nodes have a
    new hash, and recompute the faketensor metadata for that node. Then, we
    continue to recursively compute the faketensors for all users until the
    fake tensors stop changing.
    """
    processed_hashes: Incomplete
    graph: Incomplete
    def __init__(self, graph: torch.fx.Graph) -> None: ...
    def hash_node(self, node: torch.fx.Node): ...
    def incremental_update(self): ...

def get_storage(t: torch.Tensor) -> int: ...
def get_node_storage(node: torch.fx.Node) -> int | None: ...
def get_fake(x): ...
def get_fake_args_kwargs(x: torch.fx.Node) -> tuple[bool, tuple[Any], dict[str, Any]]:
    """
    First value returns a boolean if any of the input nodes don't have a faketensor.
    """
def is_node_realized(node: torch.fx.Node) -> bool:
    """Returns true if a node is always realized when lowered to inductor IR.

    NOTE: This may return some false negatives. e.g. it doesn't
    handle buffers realized heuristically during lowering, or
    buffers realized indirectly through view ops.
    """
def count_flops_fx(node: torch.fx.Node) -> int | None: ...
def countable_fx(node: torch.fx.Node) -> bool:
    """
    Whether or not we can count the flops of an FX node.
    """
