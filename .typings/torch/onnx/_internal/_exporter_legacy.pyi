import abc
import contextlib
import dataclasses
import io
import onnxruntime
import onnxscript
import torch
from _typeshed import Incomplete
from collections.abc import Generator, Mapping, Sequence
from torch._subclasses import fake_tensor
from torch.onnx._internal import io_adapter
from torch.onnx._internal.fx import registration
from typing import Any, Callable

__all__ = ['ExportOptions', 'ONNXRuntimeOptions', 'OnnxRegistry', 'enable_fake_mode']

@dataclasses.dataclass
class ONNXFakeContext:
    """A dataclass used to store context for model export using FakeTensor.

    This dataclass stores the FakeTensorMode instance used to convert
    real tensors and model parameters into fake tensors. This :attr:`ONNXFakeContext.fake_mode` is
    reused internally during tracing of a :class:`torch.nn.Module` into a FX :class:`GraphModule`.
    """
    fake_mode: fake_tensor.FakeTensorMode
    state_dict_paths: tuple[str | io.BytesIO | dict[str, Any]] | None = ...

class OnnxRegistry:
    """Registry for ONNX functions.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    The registry maintains a mapping from qualified names to symbolic functions under a
    fixed opset version. It supports registering custom onnx-script functions and for
    dispatcher to dispatch calls to the appropriate function.

    """
    _registry: dict[registration.OpName, list[registration.ONNXFunction]]
    _opset_version: Incomplete
    def __init__(self) -> None:
        """Initializes the registry"""
    @property
    def opset_version(self) -> int:
        """The ONNX opset version the exporter should target."""
    def _initiate_registry_from_torchlib(self) -> None:
        """Populates the registry with ATen functions from torchlib.

        Args:
            torchlib_registry: The torchlib registry to use for populating the registry.
        """
    def _register(self, internal_qualified_name: registration.OpName, symbolic_function: registration.ONNXFunction) -> None:
        """Registers a ONNXFunction to an operator.

        Args:
            internal_qualified_name: The qualified name of the operator to register: OpName.
            symbolic_function: The ONNXFunction to register.
        """
    def register_op(self, function: onnxscript.OnnxFunction | onnxscript.TracedOnnxFunction, namespace: str, op_name: str, overload: str | None = None, is_complex: bool = False) -> None:
        """Registers a custom operator: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            function: The onnx-sctip function to register.
            namespace: The namespace of the operator to register.
            op_name: The name of the operator to register.
            overload: The overload of the operator to register. If it's default overload,
                leave it to None.
            is_complex: Whether the function is a function that handles complex valued inputs.

        Raises:
            ValueError: If the name is not in the form of 'namespace::op'.
        """
    def get_op_functions(self, namespace: str, op_name: str, overload: str | None = None) -> list[registration.ONNXFunction] | None:
        """Returns a list of ONNXFunctions for the given op: torch.ops.<namespace>.<op_name>.<overload>.

        The list is ordered by the time of registration. The custom operators should be
        in the second half of the list.

        Args:
            namespace: The namespace of the operator to get.
            op_name: The name of the operator to get.
            overload: The overload of the operator to get. If it's default overload,
                leave it to None.
        Returns:
            A list of ONNXFunctions corresponding to the given name, or None if
            the name is not in the registry.
        """
    def is_registered_op(self, namespace: str, op_name: str, overload: str | None = None) -> bool:
        """Returns whether the given op is registered: torch.ops.<namespace>.<op_name>.<overload>.

        Args:
            namespace: The namespace of the operator to check.
            op_name: The name of the operator to check.
            overload: The overload of the operator to check. If it's default overload,
                leave it to None.

        Returns:
            True if the given op is registered, otherwise False.
        """
    def _all_registered_ops(self) -> set[str]:
        """Returns the set of all registered function names."""

class ExportOptions:
    """Options to influence the TorchDynamo ONNX exporter.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    Attributes:
        dynamic_shapes: Shape information hint for input/output tensors.
            When ``None``, the exporter determines the most compatible setting.
            When ``True``, all input shapes are considered dynamic.
            When ``False``, all input shapes are considered static.
        fake_context: The fake context used for symbolic tracing.
        onnx_registry: The ONNX registry used to register ATen operators to ONNX functions.
    """
    dynamic_shapes: Incomplete
    fake_context: Incomplete
    onnx_registry: Incomplete
    def __init__(self, *, dynamic_shapes: bool | None = True, fake_context: ONNXFakeContext | None = None, onnx_registry: OnnxRegistry | None = None) -> None: ...

class ResolvedExportOptions(ExportOptions):
    """Consolidates :class:`ExportOptions` with default values.
    All unspecified options from :class:`ExportOptions` are assigned a default value.
    This is an internal class and its API may be changed at any time without notice.
    """
    dynamic_shapes: bool
    fx_tracer: dynamo_graph_extractor.DynamoExport
    fake_context: Incomplete
    onnx_registry: OnnxRegistry
    decomposition_table: Incomplete
    onnxfunction_dispatcher: Incomplete
    def __init__(self) -> None: ...

@contextlib.contextmanager
def enable_fake_mode() -> Generator[Incomplete]:
    '''Enable fake mode for the duration of the context.

    Internally it instantiates a :class:`torch._subclasses.fake_tensor.FakeTensorMode` context manager
    that converts user input and model parameters into :class:`torch._subclasses.fake_tensor.FakeTensor`.

    A :class:`torch._subclasses.fake_tensor.FakeTensor`
    is a :class:`torch.Tensor` with the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a ``meta`` device. Because
    there is no actual data being allocated on the device, this API allows for
    initializing and exporting large models without the actual memory footprint needed for executing it.

    It is highly recommended to initialize the model in fake mode when exporting models that
    are too large to fit into memory.

    .. note::
        This function does not support torch.onnx.export(..., dynamo=True, optimize=True).
        Please call ONNXProgram.optimize() outside of the function after the model is exported.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> class MyModel(torch.nn.Module):  # Model with a parameter
        ...     def __init__(self) -> None:
        ...         super().__init__()
        ...         self.weight = torch.nn.Parameter(torch.tensor(42.0))
        ...     def forward(self, x):
        ...         return self.weight + x
        >>> with torch.onnx.enable_fake_mode():
        ...     # When initialized in fake mode, the model\'s parameters are fake tensors
        ...     # They do not take up memory so we can initialize large models
        ...     my_nn_module = MyModel()
        ...     arg1 = torch.randn(2, 2, 2)
        >>> onnx_program = torch.onnx.export(my_nn_module, (arg1,), dynamo=True, optimize=False)
        >>> # Saving model WITHOUT initializers (only the architecture)
        >>> onnx_program.save(
        ...     "my_model_without_initializers.onnx",
        ...     include_initializers=False,
        ...     keep_initializers_as_inputs=True,
        ... )
        >>> # Saving model WITH initializers after applying concrete weights
        >>> onnx_program.apply_weights({"weight": torch.tensor(42.0)})
        >>> onnx_program.save("my_model_with_initializers.onnx")

    .. warning::
        This API is experimental and is *NOT* backward-compatible.

    '''

class ONNXRuntimeOptions:
    """Options to influence the execution of the ONNX model through ONNX Runtime.

    .. deprecated:: 2.7
        Please use ``torch.onnx.export(..., dynamo=True)`` instead.

    Attributes:
        session_options: ONNX Runtime session options.
        execution_providers: ONNX Runtime execution providers to use during model execution.
        execution_provider_options: ONNX Runtime execution provider options.
    """
    session_options: Sequence[onnxruntime.SessionOptions] | None
    execution_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None
    execution_provider_options: Sequence[dict[Any, Any]] | None
    def __init__(self, *, session_options: Sequence[onnxruntime.SessionOptions] | None = None, execution_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None, execution_provider_options: Sequence[dict[Any, Any]] | None = None) -> None: ...

class FXGraphExtractor(abc.ABC, metaclass=abc.ABCMeta):
    """Abstract interface for FX graph extractor engines.
    This class isolates FX extraction logic from the rest of the export logic.
    That allows a single ONNX exporter that can leverage different FX graphs."""
    input_adapter: io_adapter.InputAdapter
    output_adapter: io_adapter.OutputAdapter
    def __init__(self) -> None: ...
    @abc.abstractmethod
    def generate_fx(self, options: ResolvedExportOptions, model: torch.nn.Module | Callable, model_args: Sequence[Any], model_kwargs: Mapping[str, Any]) -> torch.fx.GraphModule:
        """Analyzes user ``model`` and generates a FX graph.
        Args:
            options: The export options.
            model: The user model.
            model_args: The model's positional input arguments.
            model_kwargs: The model's keyword input arguments.
        Returns:
            The generated FX Graph.
        """
    @abc.abstractmethod
    def pre_export_passes(self, options: ResolvedExportOptions, original_model: torch.nn.Module | Callable, fx_module: torch.fx.GraphModule, fx_module_args: Sequence[Any]):
        """Applies pre-export passes to the FX graph.

        Pre-export passes are FX-to-FX graph transformations that make the graph
        more palatable for the FX-to-ONNX conversion.
        For example, it can be used to flatten model input/output, add explicit
        casts to the graph, replace/decompose operators, functionalize the graph, etc.
        """
