import torch
from _typeshed import Incomplete
from torch.fx import GraphModule
from torch.fx.graph import Graph
from typing import Any

__all__ = ['FusedGraphModule', 'ObservedGraphModule', 'ObservedStandaloneGraphModule', 'QuantizedGraphModule']

class FusedGraphModule(GraphModule):
    preserved_attr_names: Incomplete
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: Graph, preserved_attr_names: set[str]) -> None: ...
    def __deepcopy__(self, memo): ...

class ObservedGraphModule(GraphModule):
    preserved_attr_names: Incomplete
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: Graph, preserved_attr_names: set[str]) -> None: ...
    def __deepcopy__(self, memo): ...

class ObservedStandaloneGraphModule(ObservedGraphModule):
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: Graph, preserved_attr_names: set[str]) -> None: ...
    def __deepcopy__(self, memo): ...

class QuantizedGraphModule(GraphModule):
    """This class is created to make sure PackedParams
    (e.g. LinearPackedParams, Conv2dPackedParams) to appear in state_dict
    so that we can serialize and deserialize quantized graph module with
    torch.save(m.state_dict()) and m.load_state_dict(state_dict)
    """
    preserved_attr_names: Incomplete
    def __init__(self, root: torch.nn.Module | dict[str, Any], graph: Graph, preserved_attr_names: set[str]) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    def __deepcopy__(self, memo): ...
