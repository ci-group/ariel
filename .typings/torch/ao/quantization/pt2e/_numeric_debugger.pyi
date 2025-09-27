import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from dataclasses import dataclass
from torch.ao.ns.fx.utils import compute_sqnr as compute_sqnr
from torch.ao.quantization.pt2e.graph_utils import bfs_trace_with_node_process as bfs_trace_with_node_process
from torch.export import ExportedProgram as ExportedProgram
from torch.fx import GraphModule as GraphModule, Node as Node
from typing import Callable

NUMERIC_DEBUG_HANDLE_KEY: str
CUSTOM_KEY: str
log: Incomplete

def generate_numeric_debug_handle(ep: ExportedProgram) -> None:
    """
    Attach numeric_debug_handle_id for all nodes in the graph module of the given
    ExportedProgram, like conv2d, squeeze, conv1d, etc, except for placeholder.
    Notice that nodes like getattr are out of scope since they are not in the graph.

    The graph nodes of input exported program are modified inplace.

    Here's an example of using debug handle quantize flow::

        ep = export_for_training(eager_model, example_inputs)
        generate_numeric_debug_handle(ep)

        m = ep.module()
        quantizer = XNNPACKQuantizer()
        m = prepare_pt2e(m, quantizer)
        m = convert_pt2e(m)
    """
def _detach(x: object) -> object: ...
def _tensor_shape_equals(x: object, y: object) -> bool: ...
def _loss_fn(loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x: object, y: object) -> object:
    """The returned loss will have the same structure as `x` and `y`, e.g.
    if both are Tensor, we'll return a Tensor
    if both are list, we'll return a list of Tensors
    if both are dict, we'll return a dict with the same key, and value being the loss between the
    two Tensors
    """

class OutputLogger(torch.nn.Module):
    """
    Base class for capturing output values for nodes in a GraphModule, it only captures
    Tensor output currently, but we can extend it to work for other types of inputs later if needed
    """
    _is_impure: bool
    node_name: Incomplete
    nn_module_stack: Incomplete
    debug_handle: Incomplete
    stats: list[object]
    def __init__(self, debug_handle: int, node_name: str | None = None, nn_module_stack: object | None = None) -> None: ...
    def forward(self, x: object) -> object: ...
    def __extra_repr__(self) -> str: ...

def _insert_logger(model: GraphModule, node: Node, debug_handle: int) -> Node:
    """For a given node, adds an OutputLogger that observes the output of that node,
    and all its users use the OutputLogger output instead.
    The OutputLogger will contain the debug_handle which can be used to compare
    graphs after transforms"""
def prepare_for_propagation_comparison(model: GraphModule) -> GraphModule:
    """Add output loggers to node that has numeric_debug_handle

    Args:
        model (GraphModule): original model
    Returns:
        a model with output loggers for all nodes that has numeric_debug_handle_id
    """

@dataclass(frozen=True)
class QuantizationComparisonResult:
    actual: torch.Tensor
    ref: torch.Tensor
    @property
    def mse_loss(self) -> object: ...
    @property
    def sqnr(self) -> object: ...
    def loss(self, loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> object: ...
    def __repr__(self) -> str: ...
    def __post_init__(self) -> None: ...

@dataclass(frozen=True)
class NodeAccuracySummary:
    handle: int
    actual_node_name: str
    actual_module_stack: str
    ref_node_name: str
    ref_module_stack: str
    results: Sequence[QuantizationComparisonResult]

def _module_stack_to_str(module_stack: object) -> str:
    '''Simplifies the stack from ("mod", "mod.foo", "mod.foo.0", "mod.foo.0.linear")
    to "mod.foo.0.linear"
    '''
def extract_results_from_loggers(model: GraphModule) -> dict[int, tuple[str | None, object, list[object]]]:
    """For a given model, extract the tensors stats and related information for each debug handle.
    The reason we have a list of object, instead of Tensor is because the output of node may not be
    a Tensor, it could be (nested) list, tuple or dict as well.

    Returns:
        A dict is keyed by the debug_handle id and the values are a list of object recorded
        in loggers

    """
def compare_results(ref_results: dict[int, tuple[str | None, object, list[torch.Tensor]]], actual_results: dict[int, tuple[str | None, object, list[torch.Tensor]]]) -> dict[int, NodeAccuracySummary]:
    """Given two dict mapping from `debug_handle_id` (int) to list of tensors
    return a map from `debug_handle_id` to `NodeAccuracySummary` that contains
    comparison information like SQNR, MSE etc.

    Args:
        ref_results (Dict[int, Tuple[str, object, List[torch.Tensor]]]): reference results for each debug_handle_id
        actual_results (Dict[int, Tuple[str, object, List[torch.Tensor]]]): actual results for each debug_handle_id

    Returns:
        Dict[int, NodeAccuracySummary]
    """
