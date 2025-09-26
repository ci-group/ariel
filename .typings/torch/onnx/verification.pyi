import dataclasses
import enum
import numpy as np
import torch
import torch._C._onnx as _C_onnx
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from torch import _C
from torch.onnx import _experimental
from torch.onnx._internal.exporter._verification import VerificationInfo as VerificationInfo, verify_onnx_program as verify_onnx_program
from torch.types import Number
from typing import Any, Callable

__all__ = ['OnnxBackend', 'VerificationOptions', 'verify', 'check_export_model_diff', 'VerificationInfo', 'verify_onnx_program', 'GraphInfo', 'GraphInfoPrettyPrinter', 'OnnxTestCaseRepro', 'find_mismatch', 'verify_aten_graph']

_NumericType = Number | torch.Tensor | np.ndarray
_ModelType = torch.nn.Module | torch.jit.ScriptModule
_InputArgsType = torch.Tensor | tuple[Any, ...]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Sequence[_NumericType] | Sequence

class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.
    """
    REFERENCE = 'ONNXReferenceEvaluator'
    ONNX_RUNTIME_CPU = 'CPUExecutionProvider'
    ONNX_RUNTIME_CUDA = 'CUDAExecutionProvider'

@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """
    flatten: bool = ...
    ignore_none: bool = ...
    check_shape: bool = ...
    check_dtype: bool = ...
    backend: OnnxBackend = ...
    rtol: float = ...
    atol: float = ...
    remained_onnx_input_idx: Sequence[int] | None = ...
    acceptable_error_percentage: float | None = ...

class _GraphDiff:
    """A class to represent the difference between two graphs."""
    graph_a: Incomplete
    graph_b: Incomplete
    def __init__(self, graph_a: _C.Graph, graph_b: _C.Graph) -> None:
        """Construct a _GraphDiff object.

        Args:
            graph_a (_C.Graph): First graph to compare.
            graph_b (_C.Graph): Second graph to compare.
        """
    def __str__(self) -> str:
        """See function :func:`diff_report`."""
    def _indent(self, lines: str) -> str: ...
    def diff_report(self) -> str:
        """Return a string representation of the graph difference.

        The report shows the first pair of nodes that diverges. It also shows the source
        location of the pair of nodes.

        Returns:
            graph_diff_report (str): A string representation of the graph difference.
        """

def check_export_model_diff(model: torch.nn.Module | torch.jit.ScriptModule, test_input_groups: Sequence[tuple[tuple[Any, ...], Mapping[str, Any]]], export_options: _experimental.ExportOptions | None = None) -> str:
    """Verify exported model discrepancy between different groups of inputs.

    A graph is exported for each group of inputs. The exported graphs are then compared
    to each other, and discrepancies of first pair of nodes are reported. This function
    first checks the jit graph. If no discrepancies were found, it then checks the onnx
    graph.

    Unless otherwise specified, the jit/ONNX graph is expected to be the same, regardless
    of the inputs used for exporting. A discrepancy implies the graph exported is
    not accurate when run on other groups of inputs, which will typically results in
    runtime errors or mismatching output.

    Args:
        model (torch.nn.Module or torch.jit.ScriptModule): The model to be exported.
        test_input_groups (Sequence[Tuple[Tuple[Any, ...], Mapping[str, Any]]]): A sequence
            of input groups to be used to export the model. Each input group is a pair of
            (args, kwargs).
        export_options (_experimental.ExportOptions, optional): An _experimental.ExportOptions
            object that controls the export behavior.

    Returns:
        str: A string containing the diff of the exported models.
    """
def verify(model: _ModelType, input_args: _InputArgsType, input_kwargs: _InputKwargsType | None = None, do_constant_folding: bool = True, dynamic_axes: Mapping[str, Mapping[int, str] | Mapping[str, Sequence[int]]] | None = None, input_names: Sequence[str] | None = None, output_names: Sequence[str] | None = None, training: _C_onnx.TrainingMode = ..., opset_version: int | None = None, keep_initializers_as_inputs: bool = True, verbose: bool = False, fixed_batch_size: bool = False, use_external_data: bool = False, additional_test_inputs: Sequence[_InputArgsType] | None = None, options: VerificationOptions | None = None):
    """Verify model export to ONNX against original PyTorch model.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Args:
        model: See :func:`torch.onnx.export`.
        input_args: See :func:`torch.onnx.export`.
        input_kwargs: See :func:`torch.onnx.export`.
        do_constant_folding: See :func:`torch.onnx.export`.
        dynamic_axes: See :func:`torch.onnx.export`.
        input_names: See :func:`torch.onnx.export`.
        output_names: See :func:`torch.onnx.export`.
        training: See :func:`torch.onnx.export`.
        opset_version: See :func:`torch.onnx.export`.
        keep_initializers_as_inputs: See :func:`torch.onnx.export`.
        verbose: See :func:`torch.onnx.export`.
        fixed_batch_size: Legacy argument, used only by rnn test cases.
        use_external_data: Explicitly specify whether to export the model with external data.
        additional_test_inputs: List of tuples. Each tuple is a group of
            input arguments to test. Currently only ``*args`` are supported.
        options: A VerificationOptions object that controls the verification behavior.

    Raises:
        AssertionError: if outputs from ONNX model and PyTorch model are not
            equal up to specified precision.
        ValueError: if arguments provided are invalid.
    """
def verify_aten_graph(graph: torch.Graph, input_args: tuple[Any, ...], export_options: _experimental.ExportOptions, params_dict: dict[str, Any] | None = None, verification_options: VerificationOptions | None = None) -> tuple[AssertionError | None, torch.Graph, _OutputsType, _OutputsType]:
    """Verify aten graph export to ONNX against original PyTorch model.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.
    """

class GraphInfoPrettyPrinter:
    graph_info: GraphInfo | None
    upper_printer: GraphInfoPrettyPrinter | None
    lower_printer: GraphInfoPrettyPrinter | None
    graph_str_lambdas: Mapping[int, str]
    connector_str_lambdas: Mapping[int, str]
    children_str_lambdas: Mapping[int, str]
    def __init__(self, graph_info: GraphInfo | None) -> None: ...
    def _total_rows(self) -> int: ...
    def _node_count_segment_str(self) -> str: ...
    def _graph_id_segment_str(self) -> str: ...
    def _max_segment_columns(self) -> int: ...
    def _graph_segment_str_at_line(self, line: int) -> str:
        """Get the string representation of the graph segment at the given line."""
    def _connector_segment_str_at_line(self, line: int) -> str:
        """Get the connector segment string at the given line."""
    def _children_str_at_line(self, line: int) -> str:
        """Get the string representation of the children at the given line.

        Recursively calls `_str_at_line` on children nodes.
        """
    def _str_at_line(self, line: int) -> str:
        """Get the string representation of the graph at the given line."""
    def pretty_print(self) -> None: ...

class OnnxTestCaseRepro:
    repro_dir: Incomplete
    def __init__(self, repro_dir) -> None: ...
    @classmethod
    def create_test_case_repro(cls, proto: bytes, inputs, outputs, dir: str, name: str | None = None):
        '''Create a repro under "{dir}/test_{name}" for an ONNX test case.

        The test case contains the model and the inputs/outputs data. The directory
        structure is as follows:

        dir
        ├── test_<name>
        │   ├── model.onnx
        │   └── test_data_set_0
        │       ├── input_0.pb
        │       ├── input_1.pb
        │       ├── output_0.pb
        │       └── output_1.pb

        Args:
            proto: ONNX model proto.
            inputs: Inputs to the model.
            outputs: Outputs of the model.
            dir: Directory to save the repro.
            name: Name of the test case. If not specified, a name based on current time
                will be generated.
        Returns:
            Path to the repro.
        '''
    def validate(self, options: VerificationOptions):
        """Run the ONNX test case with options.backend, and compare with the expected outputs.

        Args:
            options: Options for validation.

        Raise:
            AssertionError: if outputs from options.backend and expected outputs are not
                equal up to specified precision.
        """

@dataclasses.dataclass
class GraphInfo:
    """GraphInfo contains validation information of a TorchScript graph and its converted ONNX graph.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.
    """
    graph: torch.Graph
    input_args: tuple[Any, ...]
    params_dict: dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(default_factory=_experimental.ExportOptions)
    mismatch_error: AssertionError | None = dataclasses.field(default=None, init=False)
    pt_outs: Sequence[_NumericType] | None = dataclasses.field(default=None, init=False)
    upper_graph_info: GraphInfo | None = dataclasses.field(default=None, init=False)
    lower_graph_info: GraphInfo | None = dataclasses.field(default=None, init=False)
    id: str = dataclasses.field(default='')
    _onnx_graph: torch.Graph | None = dataclasses.field(init=False, default=None)
    _EXCLUDED_NODE_KINDS: frozenset[str] = ...
    def clear(self) -> None:
        """Clear states and results of previous verification."""
    def pretty_print_tree(self) -> None:
        """Pretty print `GraphInfo` tree.

        Each node represents a subgraph, showing the number of nodes in the subgraph and
        a check mark if the subgraph has output mismatch between torch and ONNX.

        The id of the subgraph is shown under the node. The `GraphInfo` object for any
        subgraph can be retrieved by calling `graph_info.find_partition(id)`.

        Example::

            ==================================== Tree: =====================================
            5 X   __2 X    __1 ✓
            id:  |  id: 0 |  id: 00
                 |        |
                 |        |__1 X (aten::relu)
                 |           id: 01
                 |
                 |__3 X    __1 ✓
                    id: 1 |  id: 10
                          |
                          |__2 X     __1 X (aten::relu)
                             id: 11 |  id: 110
                                    |
                                    |__1 ✓
                                       id: 111
            =========================== Mismatch leaf subgraphs: ===========================
            ['01', '110']
            ============================= Mismatch node kinds: =============================
            {'aten::relu': 2}

        """
    def pretty_print_mismatch(self, graph: bool = False):
        """Pretty print details of the mismatch between torch and ONNX.

        Args:
            graph: If True, print the ATen JIT graph and ONNX graph.
        """
    def has_mismatch(self) -> bool:
        """Return True if the subgraph has output mismatch between torch and ONNX."""
    def essential_node_count(self) -> int:
        """Return the number of nodes in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
    def essential_node_kinds(self) -> set[str]:
        """Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
    def all_mismatch_leaf_graph_info(self) -> list[GraphInfo]:
        """Return a list of all leaf `GraphInfo` objects that have mismatch."""
    def find_partition(self, id: str) -> GraphInfo | None:
        """Find the `GraphInfo` object with the given id."""
    def export_repro(self, repro_dir: str | None = None, name: str | None = None) -> str:
        '''Export the subgraph to ONNX along with the input/output data for repro.

        The repro directory will contain the following files::

            dir
            ├── test_<name>
            │   ├── model.onnx
            │   └── test_data_set_0
            │       ├── input_0.pb
            │       ├── input_1.pb
            │       ├── output_0.pb
            │       └── output_1.pb

        Args:
            repro_dir: The directory to export the repro files to. Defaults to current
                working directory if None.
            name: An optional name for the test case folder: "test_{name}".

        Returns:
            The path to the exported repro directory.
        '''
    def _graph_partition_pivot(self) -> int:
        """Find the pivot index to partition the graph.

        The pivot is the node that splits the graph into two parts. Each part should
        have the similar amount of nodes, excluding non essential ops, defined in
        `_EXCLUDED_NODE_KINDS`, such as `prim::Constant`.
        If the graph has an odd number of nodes, the upper part will have one more node.
        If the graph does not have any node that can be partitioned, return -1.

        Returns:
            The index of the pivot node.
        """
    def _partition_upper_graph(self) -> torch.Graph: ...
    def _partition_lower_graph(self) -> torch.Graph: ...
    def _partition_node(self, node: torch.Node, complete_upper_nodes_set: set[torch.Node], complete_lower_nodes_set: set[torch.Node], original_graph_outputs: set[torch.Value], covered_bridge_values: set[torch.Value], process_bridge_value: Callable[[torch.Value], torch.Value]): ...
    def _partition_nodes(self, graph: torch.Graph, pivot: int, process_bridge_value: Callable[[torch.Value], torch.Value]) -> tuple[list[torch.Node], list[torch.Node], set[torch.Node], set[torch.Node]]: ...
    def _bridge_kwargs(self): ...
    def _args_and_params_for_partition_graph(self, graph: torch.Graph, bridge_kwargs: Mapping[str, _NumericType | Sequence[_NumericType]], full_kwargs: Mapping[str, torch.Tensor], full_params: Mapping[str, torch.Tensor]): ...
    def verify_export(self, options: VerificationOptions) -> tuple[AssertionError | None, torch.Graph, _OutputsType, _OutputsType]:
        """
        Verify the export from TorchScript IR graph to ONNX.

        Export the TorchScript IR graph to ONNX, with the inputs, parameters and export
        options recorded in this object. Then verify the exported ONNX graph against
        the original TorchScript IR graph under the provided verification options.

        Args:
            options: The verification options.

        Returns:
            error: The AssertionError raised during the verification. Returns None if no
            error is raised.
            onnx_graph: The exported ONNX graph in TorchScript IR format.
            onnx_outs: The outputs from running exported ONNX model under the onnx
            backend in `options`.
            pt_outs: The outputs from running the TorchScript IR graph.
        """
    def find_mismatch(self, options: VerificationOptions | None = None):
        """
        Find all mismatches between the TorchScript IR graph and the exported onnx model.

        Binary searches the model graph to find the minimal subgraph that exhibits the
        mismatch. A `GraphInfo` object is created for each subgraph, recording the test
        inputs and export options, as well as the validation results.

        Args:
            options: The verification options.
        """

def find_mismatch(model: torch.nn.Module | torch.jit.ScriptModule, input_args: tuple[Any, ...], do_constant_folding: bool = True, training: _C_onnx.TrainingMode = ..., opset_version: int | None = None, keep_initializers_as_inputs: bool = True, verbose: bool = False, options: VerificationOptions | None = None) -> GraphInfo:
    '''Find all mismatches between the original model and the exported model.

    .. deprecated:: 2.7
        Consider using ``torch.onnx.export(..., dynamo=True)`` and use the returned
        ``ONNXProgram`` to test the ONNX model.

    Experimental. The API is subject to change.

    This tool helps debug the mismatch between the original PyTorch model and exported
    ONNX model. It binary searches the model graph to find the minimal subgraph that
    exhibits the mismatch.

    Args:
        model: The model to be exported.
        input_args: The input arguments to the model.
        do_constant_folding: Same as `do_constant_folding` in :func:`torch.onnx.export`.
        training: Same as `training` in :func:`torch.onnx.export`.
        opset_version: Same as `opset_version` in :func:`torch.onnx.export`.
        keep_initializers_as_inputs: Same as `keep_initializers_as_inputs` in :func:`torch.onnx.export`.
        verbose: Same as `verbose` in :func:`torch.onnx.export`.
        options: The options for the mismatch verification.

    Returns:
        A GraphInfo object that contains the mismatch information.

    Example::

        >>> import torch
        >>> import torch.onnx.verification
        >>> torch.manual_seed(0)
        >>> opset_version = 15
        >>> # Define a custom symbolic function for aten::relu.
        >>> # The custom symbolic function is incorrect, which will result in mismatches.
        >>> def incorrect_relu_symbolic_function(g, self):
        ...     return self
        >>> torch.onnx.register_custom_op_symbolic(
        ...     "aten::relu",
        ...     incorrect_relu_symbolic_function,
        ...     opset_version=opset_version,
        ... )
        >>> class Model(torch.nn.Module):
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.layers = torch.nn.Sequential(
        ...             torch.nn.Linear(3, 4),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(4, 5),
        ...             torch.nn.ReLU(),
        ...             torch.nn.Linear(5, 6),
        ...         )
        ...     def forward(self, x):
        ...         return self.layers(x)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> graph_info = torch.onnx.verification.find_mismatch(
        ...     Model(),
        ...     (torch.randn(2, 3),),
        ...     opset_version=opset_version,
        ... )
        ===================== Mismatch info for graph partition : ======================
        ================================ Mismatch error ================================
        Tensor-likes are not close!
        Mismatched elements: 12 / 12 (100.0%)
        Greatest absolute difference: 0.2328854203224182 at index (1, 2) (up to 1e-07 allowed)
        Greatest relative difference: 0.699536174352349 at index (1, 3) (up to 0.001 allowed)
        ==================================== Tree: =====================================
        5 X   __2 X    __1 \\u2713
        id:  |  id: 0 |  id: 00
             |        |
             |        |__1 X (aten::relu)
             |           id: 01
             |
             |__3 X    __1 \\u2713
                id: 1 |  id: 10
                      |
                      |__2 X     __1 X (aten::relu)
                         id: 11 |  id: 110
                                |
                                |__1 \\u2713
                                   id: 111
        =========================== Mismatch leaf subgraphs: ===========================
        [\'01\', \'110\']
        ============================= Mismatch node kinds: =============================
        {\'aten::relu\': 2}

    '''
