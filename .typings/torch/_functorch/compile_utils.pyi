import torch
import torch.fx as fx
from _typeshed import Incomplete
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols as free_unbacked_symbols
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._pytree import tree_flatten as tree_flatten
from typing import Callable

aten: Incomplete

def get_aten_target(node: fx.Node) -> Callable: ...

rand_ops: Incomplete

def fx_graph_cse(fx_g: torch.fx.graph.Graph): ...
def raise_getitems(gm: fx.GraphModule) -> fx.GraphModule: ...
def strip_overloads(gm) -> None:
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
def get_placeholders(graph): ...
def get_outputs(graph): ...
