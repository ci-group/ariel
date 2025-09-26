import torch.fx
from collections.abc import Sequence
from onnxscript.function_libs.torch_lib import graph_building as onnxscript_graph_building
from torch.onnx._internal.fx import _pass as _pass, onnxfunction_dispatcher as onnxfunction_dispatcher, type_utils as fx_type_utils
from torch.utils import _pytree as _pytree
from typing import Callable

def _fx_node_to_onnx_message_formatter(fn: Callable, self, node: torch.fx.Node, *args, **kwargs) -> str: ...
def _fx_graph_to_onnx_message_formatter(fn: Callable, self, fx_graph_module: torch.fx.GraphModule, *args, **kwargs) -> str: ...
def _retrieve_or_adapt_input_to_graph_set(fx_node_arg: fx_type_utils.Argument, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator):
    """Map FX value to TorchScript value.

    When creating TorchScript graph from FX graph, we need a mapping from FX variable
    to TorchScript variable. This function maps FX variable, fx_node_arg, to torch.jit.Value.
    """
def filter_incompatible_and_dtype_convert_kwargs(kwargs):
    """Filter out kwargs that are not supported by onnxscript."""
def _fill_tensor_shape_type(onnxscript_values: onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...], name: str, expected_values: fx_type_utils.META_VALUE_TYPE | list[fx_type_utils.META_VALUE_TYPE] | tuple[fx_type_utils.META_VALUE_TYPE | None, ...]):
    """Fill the meta information of onnxscript_values with that from the fx FakeTensor."""
def _fill_in_default_kwargs(node: torch.fx.Node) -> tuple[list[fx_type_utils.Argument], dict[str, fx_type_utils.Argument]]:
    """Find and Fill in the not provided kwargs with default values."""
def _wrap_fx_args_as_onnxscript_args(complete_args: list[fx_type_utils.Argument], complete_kwargs: dict[str, fx_type_utils.Argument], fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator) -> tuple[Sequence[onnxscript_graph_building.TorchScriptTensor | str | int | float | bool | list | complex | None], dict[str, fx_type_utils.Argument]]:
    """Map all FX arguments of a node to arguments in TorchScript graph."""

class FxOnnxInterpreter:
    """Stateless class to process FX graph Nodes and translate them into their ONNX counterparts.

    All FX nodes described by [FX Graph](https://pytorch.org/docs/stable/fx.html#torch.fx.Graph) are supported.
    Similarly to [FX Interpreter pattern](https://pytorch.org/docs/stable/fx.html#torch.fx.Interpreter), each FX node
    must be implemented on its own method in this class.

    Each operator's implementation returns either an `onnxscript.OnnxFunction` or
    `onnxscript.TracedOnnxFunction` instance based on the dispatch algorithm. They can
    also raise RuntimeError: If there are no overloaded functions available for the given FX node.
    """
    def run_node(self, node, fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]]):
        """Execute a single FX node to produce its ONNX counterpart.

        Args:
            node: The FX node to be translated.
            fx_graph_module: The FX graph module containing the node.
            onnxfunction_dispatcher: The dispatcher to find the best matched ONNX op.
            onnxscript_graph: The ONNX graph to be populated.
            onnxscript_tracer: The tracer to trace the ONNX graph.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNX Script value.

        Raises:
            RuntimeError: When a node.op is not supported.
        """
    def run(self, fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, parent_onnxscript_graph: onnxscript_graph_building.TorchScriptGraph | None = None) -> onnxscript_graph_building.TorchScriptGraph:
        """Analyze all FX nodes and trigger their ONNX translation.

        Args:
            fx_graph_module: FX graph module to be translated.
            onnxfunction_dispatcher: ONNX function dispatcher.
            parent_onnxscript_graph: The parent TorchScript graph. Must be provided if
                `fx_graph_module` is a submodule. If not provided,
                `fx_graph_module` is assumed to be the root module.
        """
    def placeholder(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]]): ...
    def call_function(self, node: torch.fx.Node, onnxscript_tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]], onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher, fx_graph_module: torch.fx.GraphModule): ...
    def output(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]]): ...
    def call_method(self, node: torch.fx.Node): ...
    def call_module(self, node: torch.fx.Node, parent_onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]], tracer: onnxscript_graph_building.TorchScriptTracingEvaluator, root_fx_graph_module: torch.fx.GraphModule, onnxfunction_dispatcher: onnxfunction_dispatcher.OnnxFunctionDispatcher) -> None:
        """Export a fx.GraphModule submodule to ONNXScript graph.

        The export process specifically targets `call_module` nodes that are created by
        the exporter's `Modularize` pass. Each `call_module` node has an associated fx.GraphModule
        by `node.target` underneath the root fx.GraphModule. These `call_module` nodes are exported as ONNX
        function nodes. The related `sub_module` is then exported as an ONNX model local function,
        which is represented by another `TorchScriptGraph`. This `TorchScriptGraph` sets the current
        `onnxscript_graph` as its parent.

        Args:
            node: The call_module node in the FX graph that represents the submodule call.
            parent_onnxscript_graph: The parent ONNXScript graph to which the ONNX function and
                function node belong.
            fx_name_to_onnxscript_value: The mapping from FX node name to ONNXScript value.
            tracer: The tracer used to trace the ONNXScript graph.
            root_fx_graph_module: The root FX module.
            onnxfunction_dispatcher: The dispatcher.
        """
    def get_attr(self, node: torch.fx.Node, onnxscript_graph: onnxscript_graph_building.TorchScriptGraph, fx_name_to_onnxscript_value: dict[str, onnxscript_graph_building.TorchScriptTensor | tuple[onnxscript_graph_building.TorchScriptTensor, ...]], fx_graph_module: torch.fx.GraphModule): ...
