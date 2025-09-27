import torch
from . import ir as ir
from .exc import SubgraphLoweringException as SubgraphLoweringException
from .ops_handler import SimpleCSEHandler as SimpleCSEHandler
from .virtualized import V as V, WrapperHandler as WrapperHandler, ops as ops
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

T = TypeVar('T')
_P = ParamSpec('_P')
OpOverload = torch._ops.OpOverload
LoweringDict = dict[OpOverload | str, Callable[..., Any]]
TargetType = Callable[..., Any] | str

class PointwiseSubgraphLowering(torch.fx.Interpreter):
    """
    Lowers a pointwise subgraph to a single set of buffers with a separate
    lowering object. Errors if buffers are created unexpectedly
    """
    graph_outputs: list[ir.IRNode] | None
    root_graph: torch._inductor.graph.GraphLowering
    _current_op: TargetType | None
    allowed_mutations: OrderedSet[OpOverload] | None
    additional_lowerings: LoweringDict | None
    buffers: list[ir.Buffer]
    mutated_buffers: OrderedSet[str]
    def __init__(self, gm: torch.fx.GraphModule, root_graph_lowering: torch._inductor.graph.GraphLowering, allowed_mutations: OrderedSet[OpOverload] | None = None, additional_lowerings: LoweringDict | None = None) -> None: ...
    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        """Set which op is being processed in call function to know if we can mutate buffers"""
    def _approved_mutator(self) -> bool: ...
    def mark_buffer_mutated(self, name: str) -> None: ...
    def register_buffer(self, buffer: ir.Buffer, *, set_name: bool = False) -> str: ...
    def __getattr__(self, name: str) -> Any: ...
    def call_function(self, target: TargetType, args: Any, kwargs: dict[str, Any]) -> Any: ...
    def output(self, target: str, args: tuple[Any], kwargs: dict[str, Any]) -> None: ...

@dataclass
class InputDescriptor:
    dtype: torch.dtype
    device: torch.device

class TracingOpsHandler(WrapperHandler):
    tracer: Incomplete
    placeholders: Incomplete
    def __init__(self, tracer: torch.fx.Tracer, num_inputs: int) -> None: ...
    def placeholder(self, idx: int) -> torch.fx.Proxy: ...
    def output(self, *args: tuple[object]) -> None: ...

def lower_pointwise_subgraph(subgraph: ir.Subgraph, inputs: list[InputDescriptor]) -> Callable[_P, Any]: ...
