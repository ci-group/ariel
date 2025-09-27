import torch
import torch.fx
from _typeshed import Incomplete
from typing import Callable

__all__ = ['FoldedGraphModule', 'get_unique_attr_name_in_module', 'split_const_subgraphs']

class FoldedGraphModule(torch.fx.GraphModule):
    """
    FoldedGraphModule is a GraphModule which also contains another
    `const_subgraph_module` representing a subgraph which has all const attr
    inputs and which can be run once before running the main standard
    `graph`. The `const_output_names` are the ordered list names of attrs which
    represent what each respective output from the const_subgraph should be set
    on which attrs.
    """
    const_subgraph_module: Incomplete
    has_folding_been_run: bool
    fx_const_folded_attrs_name: Incomplete
    device_for_folded_attrs: Incomplete
    def __init__(self, root: torch.nn.Module, graph: torch.fx.Graph, const_subgraph: torch.fx.Graph | None = None, fx_const_folded_attrs_name: str | None = None, device_for_folded_attrs: str = 'cuda') -> None: ...
    def __call__(self, *args, **kwargs): ...
    def run_folding(self): ...

def get_unique_attr_name_in_module(mod_traced: torch.fx.GraphModule, name: str) -> str:
    """
    Make sure the name is unique (in a module) and can represents an attr.
    """
def split_const_subgraphs(module: torch.nn.Module | torch.fx.GraphModule, skip_folding_node_fn: Callable[[torch.fx.Node], bool] | None = None, device_for_folded_attrs: str = 'cpu') -> FoldedGraphModule:
    """
    Looks through `module` for any nodes that have all constant attribute inputs
    and separates them out into their own constant subgraph, and returns a
    FoldedGraphModule which runs that constant subgraph on the first run to set
    attributes on the module prior to running the non-constant portion of the
    graph.
    """
