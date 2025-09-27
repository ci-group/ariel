import torch.fx
from _typeshed import Incomplete
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import Node
from torch.utils._ordered_set import OrderedSet

__all__ = ['FakeTensorProp']

class FakeTensorProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """
    _mode: Incomplete
    seen_subgraphs: OrderedSet[str]
    def __init__(self, module: torch.fx.GraphModule, mode: FakeTensorMode | None = None) -> None: ...
    def run_node(self, n: Node): ...
    def propagate(self, *args): ...
    def propagate_dont_convert_inputs(self, *args): ...
