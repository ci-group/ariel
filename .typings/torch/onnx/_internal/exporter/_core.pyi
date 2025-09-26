import numpy.typing as npt
import onnxscript
import os
import torch
import torch.fx
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from onnxscript import ir
from torch.export import graph_signature as graph_signature
from torch.onnx._internal._lazy_import import onnxscript_apis as onnxscript_apis
from torch.onnx._internal.exporter import _analysis as _analysis, _building as _building, _capture_strategies as _capture_strategies, _constants as _constants, _dispatching as _dispatching, _errors as _errors, _flags as _flags, _fx_passes as _fx_passes, _ir_passes as _ir_passes, _onnx_program as _onnx_program, _registration as _registration, _reporting as _reporting, _tensors as _tensors, _type_casting as _type_casting, _verification as _verification
from typing import Any, Callable, Literal

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, ir.DataType]
_BLUE: str
_END: str
_STEP_ONE_ERROR_MESSAGE: Incomplete
_STEP_TWO_ERROR_MESSAGE: Incomplete
_STEP_THREE_ERROR_MESSAGE: Incomplete
logger: Incomplete
current_tracer: _building.OpRecorder | None

def torch_dtype_to_onnx_dtype(dtype: torch.dtype) -> ir.DataType: ...

class TorchTensor(ir.Tensor):
    def __init__(self, tensor: torch.Tensor, name: str | None = None) -> None: ...
    raw: torch.Tensor
    def numpy(self) -> npt.NDArray: ...
    def __array__(self, dtype: Any = None, copy: bool | None = None) -> npt.NDArray: ...
    def tobytes(self) -> bytes: ...

def _set_shape_types(values: Sequence[ir.Value], meta_vals: Sequence[torch.Tensor], complex_to_float: bool = True) -> None: ...
def _set_shape_type(value: ir.Value, meta_val: torch.Tensor | torch.SymBool | torch.SymInt | torch.SymFloat | tuple[torch.Tensor], complex_to_float: bool) -> None: ...
def _get_qualified_module_name(cls) -> str: ...
def _get_node_namespace(node: torch.fx.Node) -> tuple[str, list[str], list[str]]:
    '''Get the namespace and scope of the node.

    Example::

        {
            \'L__self__\': (\'\', <class \'torchvision.models.resnet.ResNet\'>),
            \'L__self___avgpool\': (\'avgpool\', <class \'torch.nn.modules.pooling.AdaptiveAvgPool2d\'>)
        }

    Will yield

    namespace: ": torchvision.models.resnet.ResNet/avgpool: torch.nn.modules.pooling.AdaptiveAvgPool2d/node_name: node_target"
    class_hierarchy: ["torchvision.models.resnet.ResNet", "torch.nn.modules.pooling.AdaptiveAvgPool2d", <node_target>]
    name_scopes: ["", "avgpool", <node_name>]

    Args:
        node: The node to get the namespace and scope of.

    Returns:
        (namespace, class_hierarchy, name_scope)
    '''
def _set_node_metadata(fx_node: torch.fx.Node, ir_node: ir.Node) -> None:
    """Adds namespace and other node metadata to the ONNX node."""
def _handle_getitem_node(node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]]) -> ir.Value:
    """Handle a getitem node.

    Add the input value it is getting to the mapping, then return the value.

    There are two cases for this node:
    1. The output is a Sequence (traced), we can simply get the value from the sequence
    2. The output is produced by a SplitToSequence node, we need to get the value from the sequence value
    This function only handles the first case
    """
def _handle_call_function_node(graph_like: ir.Graph | ir.Function, node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]]) -> None:
    """Handle a call_function node.

    Args:
        graph: The ONNX graph at construction.
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ir.Value.
    """
def _convert_fx_arg_to_onnx_arg(arg, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]], node_name_to_local_functions: dict[str, ir.Function]) -> Any:
    """Convert an FX argument to an ONNX compatible argument.

    This function
    - Converts a torch dtype to an integer
    - Converts a torch device/memory_format/layout to a string
    - Converts a torch.fx.Node to an ir.Value
    - Converts a sequence of torch.fx.Node to a sequence of ir.Value
    - Converts a get_attr node to an ir.Function
    """
def _get_onnxscript_opset(opset_version: int) -> onnxscript.values.Opset: ...
def _is_onnx_op(op: Any) -> bool:
    """Whether the op overload is an ONNX custom op implemented with PyTorch."""
def _parse_onnx_op(op: torch._ops.OpOverload) -> tuple[str, int]:
    """Parse the ONNX custom op overload name to get the op type and opset version."""
def _handle_call_function_node_with_lowering(model: ir.Model, node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]], *, graph_like: ir.Graph | ir.Function, constant_farm: dict[Any, ir.Value], registry: _registration.ONNXRegistry, opset: onnxscript.values.Opset, node_name_to_local_functions: dict[str, ir.Function]) -> None:
    """Translate a call_function node to an ONNX node.

    Args:
        model: The ONNX model at construction.
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ONNX ``Value``.
        graph_like: The current ONNX graph at construction.
            Must add nodes to this graph because it can be a subgraph that is currently being constructed.
        constant_farm: A mapping of constant values to existing ONNX ``Value``s.
        registry: The registry of all aten to ONNX decomposition functions.
        opset: The ONNX Script opset object for constructing ONNX nodes.
        node_name_to_local_functions: A mapping of subgraph names to the corresponding ONNX functions.
    """
def _handle_placeholder_node(node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]], *, graph_like: ir.Graph | ir.Function, lower: str, opset: onnxscript.values.Opset) -> None: ...
def _handle_get_attr_node(node: torch.fx.Node, *, owned_graphs: Mapping[str, ir.Function], node_name_to_local_functions: dict[str, ir.Function]) -> None:
    '''Handle a get_attr node by assigning the corresponding ONNX function to the node name.

    An example ExportedProgram that has uses get_attr nodes is:

        ExportedProgram:
        class GraphModule(torch.nn.Module):
            def forward(self, arg0_1: "f32[5]"):
                true_graph_0 = self.true_graph_0  # get_attr
                false_graph_0 = self.false_graph_0  # get_attr
                conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [arg0_1]);  true_graph_0 = false_graph_0 = arg0_1 = None
                getitem: "f32[5]" = conditional[0];  conditional = None
                return (getitem,)

            class <lambda>(torch.nn.Module):
                def forward(self, arg0_1: "f32[5]"):
                    cos: "f32[5]" = torch.ops.aten.cos.default(arg0_1);  arg0_1 = None
                    return (cos,)

            class <lambda>(torch.nn.Module):
                def forward(self, arg0_1: "f32[5]"):
                    sin: "f32[5]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
                    return (sin,)

    Args:
        node: The FX node to translate.
        owned_graphs: A mapping of subgraph names to the corresponding ONNX functions.
        node_name_to_local_functions: A mapping of local function names to their corresponding ONNX functions.
    '''
def _handle_output_node(node: torch.fx.Node, node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]], graph_like: ir.Graph | ir.Function) -> None:
    """Handle an output node by adding the output to the graph's outputs.

    Args:
        node: The FX node to translate.
        node_name_to_values: A mapping of FX node names to their produced ONNX ``Value``.
        graph_like: The ONNX graph at construction.
    """
def _translate_fx_graph(fx_graph: torch.fx.Graph, model: ir.Model, *, graph_like: ir.Graph | ir.Function, owned_graphs: Mapping[str, ir.Function], lower: Literal['at_conversion', 'none'], registry: _registration.ONNXRegistry) -> dict[str, ir.Value | Sequence[ir.Value]]:
    '''Translate a submodule to an ONNX function.

    Any functions used by the traced functions will be added to the model.

    Args:
        fx_graph: The FX graph module to translate.
        model: The ONNX model at construction.
        current_scope: The current name scope of the submodule, excluding the current module name.
            E.g. "true_graph_0.false_graph_0".
        graph_name: The name of the submodule. E.g. "true_graph_0".
        graph: The ONNX graph at construction.
        owned_graphs: The subgraphs owned by the current graph.
        lower: The lowering strategy to use.
        registry: The registry of all aten to ONNX decomposition functions.

    Returns:
        A mapping of FX node names to their produced ONNX ``Value``.
    '''
def _get_inputs_and_attributes(node: torch.fx.Node) -> tuple[list[torch.fx.Node | None], dict[str, Any], list[str], list[str]]:
    """Find and Fill in the not provided kwargs with default values.

    Returns:
        (inputs, attributes, input_names, output_names)
    """
def _maybe_start_profiler(should_profile: bool) -> Any: ...
def _maybe_stop_profiler_and_get_result(profiler) -> str | None: ...
def _format_exception(e: Exception) -> str:
    """Format the full traceback as Python would show it."""
def _summarize_exception_stack(e: BaseException) -> str:
    """Format the exception stack by showing the text of each exception."""
def _format_exceptions_for_all_strategies(results: list[_capture_strategies.Result]) -> str:
    """Format all the exceptions from the capture strategies."""
def exported_program_to_ir(exported_program: torch.export.ExportedProgram, *, registry: _registration.ONNXRegistry | None = None, lower: Literal['at_conversion', 'none'] = 'at_conversion') -> ir.Model:
    """Convert an exported program to an ONNX IR model.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.ir_spec.html

    Args:
        exported_program: The exported program to convert.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower whe translating the FX graph to ONNX IR.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
def _prepare_exported_program_for_export(exported_program: torch.export.ExportedProgram, *, registry: _registration.ONNXRegistry) -> torch.export.ExportedProgram:
    """Decompose and apply pre-export transformations to the exported program."""
def _get_scope_name(scoped_name: str) -> tuple[str, str]:
    """Get the scope and name of a node.

    Examples::
        >>> _get_scope_name('')
        ('', '')
        >>> _get_scope_name('true_graph')
        ('', 'true_graph')
        >>> _get_scope_name('true_graph.false_graph')
        ('true_graph', 'false_graph')
        >>> _get_scope_name('true_graph.false_graph.some_graph')
        ('true_graph.false_graph', 'some_graph')

    Args:
        scoped_name: The scoped name of the node.

    Returns:
        (scope, name)
    """
def _exported_program_to_onnx_program(exported_program: torch.export.ExportedProgram, *, registry: _registration.ONNXRegistry, lower: Literal['at_conversion', 'none'] = 'at_conversion') -> _onnx_program.ONNXProgram:
    """Convert an exported program to an ONNX Program.

    The exported_program field in the returned ONNXProgram is one that is after
    decompositions have been applied.

    Reference:
        - ExportedProgram spec: https://pytorch.org/docs/stable/export.ir_spec.html

    Args:
        exported_program: The exported program to convert. The exported program
            should be the one that is after decompositions have been applied.
        lower: Whether to lower the graph to core ONNX operators.
            at_conversion: Lower whe translating the FX graph to ONNX IR.
            none: Do not lower the graph.
        registry: The registry of all ONNX Script decomposition.
    """
def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
@_flags.set_onnx_exporting_flag
def export(model: torch.nn.Module | torch.export.ExportedProgram | torch.fx.GraphModule | torch.jit.ScriptModule | torch.jit.ScriptFunction, args: tuple[Any, ...] = (), kwargs: dict[str, Any] | None = None, *, registry: _registration.ONNXRegistry | None = None, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None, input_names: Sequence[str] | None = None, output_names: Sequence[str] | None = None, report: bool = False, verify: bool = False, profile: bool = False, dump_exported_program: bool = False, artifacts_dir: str | os.PathLike = '.', verbose: bool | None = None) -> _onnx_program.ONNXProgram:
    """Export a PyTorch model to ONNXProgram.

    Args:
        model: The model to export. This can be a PyTorch nn.Module or an ExportedProgram.
        args: The arguments to pass to the model.
        kwargs: The keyword arguments to pass to the model.
        registry: The registry of all ONNX decompositions.
        dynamic_shapes: Dynamic shapes in the graph.
        input_names: If provided, rename the inputs.
        output_names: If provided, rename the outputs.
        report: Whether to generate an error report if the export fails.
        verify: Whether to verify the ONNX model after exporting.
        profile: Whether to profile the export process. When report is True,
            the profile result will be saved in the report. Otherwise, the profile
            result will be printed.
        dump_exported_program: Whether to save the exported program to a file.
        artifacts_dir: The directory to save the exported program and error reports.
        verbose: Whether to print verbose messages. If None (default), some messages will be printed.

    Returns:
        The ONNXProgram with the exported IR graph.

    Raises:
        TorchExportError: If the export process fails with torch.export.
        ConversionError: If the ExportedProgram to ONNX translation fails.
    """
