import dataclasses
import onnx
import onnxruntime
import torch
import torch.fx
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from onnxruntime.capi import _pybind_state as ORTC
from torch.fx.passes.operator_support import OperatorSupport
from typing import Any, Callable, Final
from typing_extensions import TypeAlias

__all__ = ['is_onnxrt_backend_supported', 'torch_compile_backend', 'OrtExecutionProvider', 'OrtBackendOptions', 'OrtBackend']

def is_onnxrt_backend_supported() -> bool:
    '''Returns ``True`` if ONNX Runtime dependencies are installed and usable
    to support TorchDynamo backend integration; ``False`` otherwise.

    Example::

        # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> import torch
        >>> if torch.onnx.is_onnxrt_backend_supported():
        ...     @torch.compile(backend="onnxrt")
        ...     def f(x):
        ...             return x * x
        ...     print(f(torch.randn(10)))
        ... else:
        ...     print("pip install onnx onnxscript onnxruntime")
        ...
    '''

class OrtOperatorSupport(OperatorSupport):
    """Operator support for ONNXRuntime backend.

    It has two-level of support decision. One is via support_dict and the other one
    is via extra_support_dict. The logic of using support_dict is implemented in
    OrtOperatorSupport and extra_support_dict is used by OperatorSupport.is_node_supported.
    """
    _onnx_support_dict: Incomplete
    def __init__(self, support_dict: set[Any], extra_support_dict: dict[str, Any]) -> None: ...
    def is_node_supported(self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool: ...

class OrtExecutionInfoPerSession:
    """Information required to execute torch.fx.GraphModule using onnxruntime.InferenceSession"""
    session: onnxruntime.InferenceSession
    input_names: tuple[str, ...]
    input_value_infos: tuple[onnx.ValueInfoProto, ...]
    output_names: tuple[str, ...]
    output_value_infos: tuple[onnx.ValueInfoProto, ...]
    input_devices: tuple[ORTC.OrtDevice, ...]
    output_devices: tuple[ORTC.OrtDevice, ...]
    example_outputs: tuple[torch.Tensor, ...] | torch.Tensor
    def __init__(self, session: onnxruntime.InferenceSession, input_names: tuple[str, ...], input_value_infos: tuple['onnx.ValueInfoProto', ...], output_names: tuple[str, ...], output_value_infos: tuple['onnx.ValueInfoProto', ...], input_devices: tuple['ORTC.OrtDevice', ...], output_devices: tuple['ORTC.OrtDevice', ...], example_outputs: tuple[torch.Tensor, ...] | torch.Tensor) -> None: ...
    def is_supported(self, *args): ...

@dataclasses.dataclass
class OrtExecutionInfoForAllGraphModules:
    execution_info_per_graph_module: dict[torch.fx.GraphModule, list[OrtExecutionInfoPerSession]] = ...
    def __init__(self) -> None: ...
    def search_reusable_session_execution_info(self, graph_module: torch.fx.GraphModule, *args): ...
    def cache_session_execution_info(self, graph_module: torch.fx.GraphModule, info: OrtExecutionInfoPerSession): ...
OrtExecutionProvider: TypeAlias = str | tuple[str, Mapping[str, Any]]

@dataclasses.dataclass(frozen=True)
class OrtBackendOptions:
    '''Options for constructing an ``OrtBackend``, the ONNX Runtime
    backend (``"onnxrt"``) for ``torch.compile``.

    Example::

        >>> @torch.compile(
        ...     backend="onnxrt",
        ...     options=torch.onnx._OrtBackendOptions(...),
        ... )
        ... def ort_function(x):
        ...     return x ** x
    '''
    preferred_execution_providers: Sequence[OrtExecutionProvider] | None = ...
    infer_execution_providers: bool = ...
    default_execution_providers: Sequence[OrtExecutionProvider] | None = ...
    preallocate_output: bool = ...
    use_aot_autograd: bool = ...
    ort_session_options: onnxruntime.SessionOptions | None = ...
    pre_ort_model_transforms: Sequence[Callable[[onnx.ModelProto], None]] | None = ...

class OrtBackend:
    """A backend compiles (sub-)graphs in torch.fx.GraphModule to onnxruntime.InferenceSession calls.

    The compiler entry point is OrtBackend.compile, which
        1. partitions the original graph into supported sub-graphs (type: torch.fx.GraphModule) and unsupported
           sub-graphs.
        2. For each supported sub-graph, it replaces its _wrapped_call function with _ort_accelerated_call.
        3. Inside _ort_accelerated_call, it creates onnxruntime.InferenceSession and calls it to execute the sub-graph.
    """
    _options: Final[Incomplete]
    _resolved_onnx_exporter_options: Incomplete
    _supported_ops: Incomplete
    _partitioner_cache: dict[torch.fx.GraphModule, torch.fx.GraphModule]
    _all_ort_execution_info: Incomplete
    _assert_allclose_to_baseline: bool
    execution_count: int
    run: Incomplete
    def __init__(self, options: OrtBackendOptions | None = None) -> None: ...
    def _select_eps(self, graph_module: torch.fx.GraphModule, *args) -> Sequence[tuple[str, Mapping[str, Any]]]: ...
    preallocate_output: bool
    def _ort_acclerated_call(self, graph_module: torch.fx.GraphModule, *args, **kwargs):
        """This function replaces GraphModule._wrapped_call in compiled model.

        The _wrapped_call is the underlying implementation of forward method. Replacing
        it means we delegate the computation to _ort_acclerated_call and therefore
        onnxruntime.InferenceSession.
        """
    def compile(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule: ...
    def __call__(self, graph_module: torch.fx.GraphModule, args) -> torch.fx.GraphModule:
        """If ``OrtBackendOptions.use_aot_autograd`` is ``True``, the `auto_autograd` compiler
        will be invoked, wrapping this ``OrtBackend`` instance's ``compile`` method. Otherwise,
        the ``compile`` method is invoked directly."""
    __instance_cache_max_count: Final[int]
    __instance_cache: Final[list['OrtBackend']]
    @staticmethod
    def get_cached_instance_for_options(options: OrtBackendOptions | Mapping[str, Any] | None = None) -> OrtBackend:
        """Returns a possibly cached instance of an ``OrtBackend``. If an existing
        backend was created previously through this function with the same options,
        it will be returned. Otherwise a new backend will be created, cached, and
        returned.

        Note: if ``options`` sets ``ort_session_options``, a new ``OrtBackend``
        will always be returned, since ``onnxruntime.SessionOptions`` cannot
        participate in caching."""
    @staticmethod
    def clear_cached_instances() -> None: ...
    @staticmethod
    def get_cached_instances(): ...

def torch_compile_backend(graph_module: torch.fx.GraphModule, args, *, options: OrtBackendOptions | Mapping[str, Any] | None = None): ...
