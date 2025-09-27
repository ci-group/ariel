from _typeshed import Incomplete
from torch.fx import Graph as Graph, GraphModule as GraphModule, Node as Node
from torch.fx.passes.infra.pass_base import PassBase as PassBase, PassResult as PassResult
from torch.utils._pytree import tree_flatten as tree_flatten

aten: Incomplete
rand_ops: Incomplete
inplace_ops: Incomplete

def get_CSE_banned_ops(): ...

class CSEPass(PassBase):
    banned_ops: Incomplete
    def __init__(self, banned_ops=None) -> None:
        """
        This version of CSE Pass aims to be dialect agnostic, and it's implemented purely based on the connectivity between fx.Node.

        For functional dialects, user would only need to specify the random ops in ban list.

        Warning: CSE Pass cannot be safely applied on a FX graph in non-functional dialects.
        If your dialect contains stateful operators, please customized the banned_ops.

        """
    def call(self, graph_module: GraphModule) -> PassResult:
        """
        Return a new copy of torch.fx.GraphModule with CSE applied to the input graph

        Example usage:

        from torch.fx.experimental.proxy_tensor import make_fx
        def f(a):
            b = a * a
            c = a * a
            return b+c

        p = CSEPass()
        traced_graph = make_fx(f)(torch.tensor(1))
        print(traced_graph)
        result = p(traced_graph)
        print(result.graph_module)
        """
