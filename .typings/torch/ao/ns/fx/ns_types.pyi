import enum
from torch.fx.graph import Node as Node
from typing import Any, Callable, NamedTuple

class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = 'weight'
    NODE_OUTPUT = 'node_output'
    NODE_INPUT = 'node_input'

class NSSubgraph(NamedTuple):
    start_node: Node
    end_node: Node
    base_op_node: Node
NSSingleResultType = dict[str, Any]
NSResultsType = dict[str, dict[str, dict[str, list[NSSingleResultType]]]]
NSNodeTargetType = Callable | str
