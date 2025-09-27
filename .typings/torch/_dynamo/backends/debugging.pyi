import dataclasses
import torch
from .common import aot_autograd as aot_autograd
from .registry import register_debug_backend as register_backend
from _typeshed import Incomplete
from torch import _guards as _guards
from torch._functorch.compilers import ts_compile as ts_compile
from typing import Any

log: Incomplete

@register_backend
def eager(gm, fake_tensor_inputs, **kwargs): ...
def make_eager_backend_with_torch_function_mode(mode): ...
def make_eager_backend_with_torch_function_modes(modes):
    """Used to trace HOPs (cond and while) for eager execution, the metadata
    TF mode mutates vars outside of the scope of the HOP, and we can't have graph breaks
    in the HOP, so we need to externally run this mode and not trace it."""
@register_backend
def eager_noexcept(gm, fake_tensor_inputs, **kwargs): ...
@register_backend
def pre_dispatch_eager(gm, fake_tensor_inputs, **kwargs): ...
@register_backend
def eager_debug(gm, fake_tensor_inputs, **kwargs): ...
def torchscript(gm, fake_tensor_inputs): ...
def boxed_nop(fx_g, example_inputs): ...
def boxed_nop_with_mode(fx_g, example_inputs, *, mode): ...
def fake_crossref_boxed_nop(fx_g, example_inputs, ignore_op_fn=None): ...
def ignore_builtins(op: torch._ops.OpOverload) -> bool: ...
def get_nop_func(): ...
def aot_eager(gm, fake_tensor_inputs, fw_compiler=None, bw_compiler=None, **kwargs): ...

aot_eager_default_partitioner: Incomplete

def aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs): ...
def aot_eager_decomp_partition_with_mode(gm, fake_tensor_inputs, mode, **kwarg): ...
def aot_eager_decomp_partition_crossref(gm, fake_tensor_inputs, **kwargs): ...

aot_ts: Incomplete

class ReluCompileError(Exception): ...
class TestingOnlyCompileError(Exception): ...

@register_backend
def relu_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs): ...
@register_backend
def relu_runtime_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs): ...
@register_backend
def relu_accuracy_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs): ...
@register_backend
def non_leaf_compile_error_TESTING_ONLY(gm: torch.fx.GraphModule, example_inputs): ...

@dataclasses.dataclass
class ExplainOutput:
    """
    This is the output of :func:`torch._dynamo.explain()`
    There is no reason to create this class directly.
    """
    graphs: list[torch.fx.GraphModule]
    graph_count: int
    graph_break_count: int
    break_reasons: list[Any]
    op_count: int
    ops_per_graph: list[torch.fx.Node] | None = ...
    out_guards: list[_guards.Guard] | None = ...
    compile_times: str | None = ...
    def __str__(self) -> str: ...

def _explain_graph_detail(gm: torch.fx.GraphModule, graphs, op_count, ops_per_graph, break_reasons):
    """
    This function is a utility which processes a torch.fx.GraphModule and
    accumulates information about its ops, graph breaks, and other details. It
    is intended to be used by the ExplainWithBackend class and
    `torch._dynamo.explain()` to provide details from Dynamo's graph capture.

    Parameters:
        gm (torch.fx.GraphModule): The GraphModule to be processed.
        graphs (list): A list that accumulates all the GraphModules processed.
        op_count (int): The total count of operations in all GraphModules processed so far.
        ops_per_graph (list): A list that accumulates the operations of each GraphModule.
        break_reasons (list): A list that accumulates the reasons for breaks in each GraphModule.

    Returns:
        tuple: A tuple containing the processed GraphModule, the updated lists of graphs,
               operations per graph, and break reasons, and the updated operation count.
    """

class ExplainWithBackend:
    '''
    This class is intended to be used as a backend for `torch.compile`. It is
    composable with other backends. When used in this way, it accumulates
    information about graph breaks, ops, and other info and provides a string
    representation summarizing this information.

    Attributes:
        backend (str): The name of the backend to use for optimization.
        graphs (list): A list of the graphs captured by TorchDynamo.
        op_count (int): The total number of operations in all optimized graphs.
        break_reasons (list): A list of graph break reasons with stack traces.

    Example Usage:
        def fn(x):
            x = torch.sigmoid(x)
            return x

        torch._dynamo.reset()
        eb = ExplainWithBackend("inductor")
        optimized_fn = torch.compile(fn, backend=eb)
        result = optimized_fn(torch.randn(5))
        print(eb.output())
    '''
    backend: Incomplete
    graphs: Incomplete
    op_count: int
    break_reasons: Incomplete
    def __init__(self, backend) -> None: ...
    def __call__(self, gm: torch.fx.GraphModule, example_inputs): ...
    def output(self) -> ExplainOutput: ...
