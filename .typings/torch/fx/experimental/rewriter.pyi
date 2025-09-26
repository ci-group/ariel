import ast
import torch
from torch._sources import normalize_source_lines as normalize_source_lines
from torch.fx._symbolic_trace import Tracer as Tracer
from torch.fx.graph import Graph as Graph
from types import FunctionType
from typing import Any, Callable

class AST_Rewriter(ast.NodeTransformer):
    """
    Take a FunctionType object representing a `forward` method, then
    perform an AST rewrite to swap out nodes that are not symbolically
    traceable with a callsite to the FX alternative.

    To support swapping out an AST node, define a new `visit` method on
    that node. For more details, see:
    https://docs.python.org/3/library/ast.html#ast.NodeTransformer
    """
    @torch._dynamo.disable
    def rewrite(self, fn: FunctionType): ...
    def visit_Assert(self, node):
        """
        Swap out the Assert node (Python's `assert`) with a callsite to the
        symbolically-traceable torch._assert function
        """
    def visit_AnnAssign(self, node):
        """
        Swap out Python's AnnAssign with an Assign node where the annotation function is called.
        Example:
             Original:
             y: Tensor_Type(1,2,3, Dyn) = f2(x)
            Output:
             y = annotate(f2(x),Tensor_Type((1,2,3,Dyn)))
        """

class RewritingTracer(Tracer):
    def trace(self, root: torch.nn.Module | Callable, concrete_args: dict[str, Any] | None = None) -> Graph: ...

def _rewrite(fn: torch.nn.Module | Callable) -> torch.nn.Module | Callable: ...
