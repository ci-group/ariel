import sympy
import torch
import torch.fx as fx
import types
from _typeshed import Incomplete
from collections.abc import Generator, Mapping, MutableMapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from torch import Tensor
from torch._library.fake_class_registry import FakeScriptObject
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx import GraphModule, Proxy, Tracer
from torch.fx.node import Argument, Target
from torch.nn import Module
from torch.overrides import TorchFunctionMode
from torch.types import PySymType, py_sym_types as py_sym_types
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._stats import count
from torch.utils._thunk import Thunk
from typing import Any, Callable, Protocol, TypeVar, overload
from typing_extensions import ParamSpec, Self, TypeVarTuple
from weakref import WeakKeyDictionary

__all__ = ['PythonKeyTracer', 'dispatch_trace', 'make_fx', 'DecompositionInterpreter', 'py_sym_types', 'get_innermost_proxy_mode', 'get_proxy_mode', 'handle_sym_dispatch', 'maybe_enable_thunkify', 'maybe_disable_thunkify']

_AnyScriptObjectType = torch.ScriptObject | FakeScriptObject
T = TypeVar('T')
U = TypeVar('U')
_P = ParamSpec('_P')
R = TypeVar('R')
_Ts = TypeVarTuple('_Ts')

class _NoDefault: ...

class _HasMeta(Protocol):
    meta: dict[str, PySymType]
_PySymProxyType = Thunk[Proxy]

@contextmanager
def maybe_disable_thunkify() -> Generator[None, None, None]:
    """Within a context, disable thunkification.  See :func:`maybe_enable_thunkify`
    for more details.  This is helpful if you have a wrapper function which
    you want to enable thunkification on, but in some segment on the inside (say,
    the original user function), you want to disable thunkification as you know
    it is not needed there.
    """
@contextmanager
def maybe_enable_thunkify() -> Generator[None, None, None]:
    '''Within this context manager, if you are doing make_fx tracing, we will thunkify
    all SymNode compute and avoid tracing it into the graph unless it is actually needed.
    You should prefer to avoid using this as much as possible, as lazy evaluation of
    SymNode tracing can lead to long chains of thunks which will stack overflow
    if you evaluate them.  However, this is currently sometimes necessary as there
    are buggy parts of PT2 which will fail with "s0 is not tracked with proxy" error
    due to insufficient tracing of SymNode computation.
    '''

@dataclass
class _ProxyTensor:
    proxy: Proxy
    constant: Tensor | None

class _SymNodeDict:
    """
    Wrapper around a dictionary that will hash SymInts with their nodes
    """
    sym_node_dict: dict[PySymType, _PySymProxyType]
    def __init__(self) -> None: ...
    def __setitem__(self, key: PySymType, value: _PySymProxyType) -> None: ...
    def __getitem__(self, key: PySymType) -> _PySymProxyType: ...
    def __contains__(self, key: PySymType) -> bool: ...
    def get(self, key: PySymType, default: _PySymProxyType | None = None) -> _PySymProxyType: ...
    def __iter__(self) -> Any: ...
    def __len__(self) -> int: ...

class PythonKeyTracer(Tracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: _SymNodeDict
    sympy_expr_tracker: dict[sympy.Symbol, object]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool
    stack_trace: bool
    torch_fn_metadata: Incomplete
    def __init__(self) -> None: ...
    def call_module(self, m: Module, forward: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...
    def getattr(self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]) -> object: ...
    def create_arg(self, a: object) -> fx.node.Node: ...
    @overload
    def unwrap_proxy(self, e: Tensor) -> Proxy | Tensor: ...
    @overload
    def unwrap_proxy(self, e: PySymType) -> Proxy | PySymType: ...
    @overload
    def unwrap_proxy(self, e: _AnyScriptObjectType) -> Proxy | _AnyScriptObjectType: ...
    def create_node(self, kind: str, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Argument], name: str | None = None, type_expr: Any | None = None) -> torch.fx.Node: ...

@torch._disable_dynamo
def dispatch_trace(root: Module | Callable, tracer: Tracer, concrete_args: tuple[Any, ...] | None = None) -> GraphModule: ...

class TorchFunctionMetadataMode(TorchFunctionMode):
    tracer: Incomplete
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(self, func: OpOverload, types: tuple[torch._C._TensorMeta, ...], args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None) -> object: ...

class PreDispatchTorchFunctionMode(TorchFunctionMode):
    tracer: Incomplete
    enter_autocast_nodes: list[torch.fx.Node]
    def __init__(self, tracer: _ProxyTracer) -> None: ...
    def __torch_function__(self, func: OpOverload | Callable, types: tuple[torch._C._TensorMeta, ...], args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None) -> object: ...

class ProxyTorchDispatchMode(TorchDispatchMode):
    @property
    def enable_tracing(self) -> bool: ...
    tracer: Incomplete
    tracing_mode: Incomplete
    pre_dispatch: Incomplete
    _allow_fake_constant: Incomplete
    _error_on_data_dependent_ops: Incomplete
    _mode_key: Incomplete
    enter_stack: list[ProxyTorchDispatchMode | None]
    decomp_layers: int
    emulate_precision_casts: bool
    def __init__(self, tracer: _ProxyTracer, tracing_mode: str, pre_dispatch: bool = False, _allow_fake_constant: bool = False, _error_on_data_dependent_ops: bool = True) -> None: ...
    @count
    def __torch_dispatch__(self, func: OpOverload, types: tuple[torch._C._TensorMeta, ...], args: tuple[object, ...] = (), kwargs: dict[str, object] | None = None) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> bool | None: ...
    @classmethod
    def is_infra_mode(cls) -> bool: ...
    def _compute_proxy(self, func: OpOverload, args: tuple[object, ...], out: PySymType) -> Proxy: ...
    def __sym_dispatch__(self, func: OpOverload, types: tuple[torch._C._TensorMeta, ...], args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...

class _GraphAppendingTracerEx(fx.proxy.GraphAppendingTracer):
    script_object_tracker: MutableMapping[_AnyScriptObjectType, Proxy]
    symnode_tracker: MutableMapping[PySymType, _PySymProxyType]
    tensor_tracker: MutableMapping[Tensor, _ProxyTensor]
    sympy_expr_tracker: dict[sympy.Symbol, object]
    torch_fn_metadata: OpOverload | None
    torch_fn_counts: dict[OpOverload, int]
    enable_thunkify: bool
    def __init__(self, graph: fx.graph.Graph) -> None: ...

class DecompositionInterpreter(fx.Interpreter):
    new_graph: Incomplete
    tracer: Incomplete
    decomposition_table: Incomplete
    mode: Incomplete
    def __init__(self, module: fx.GraphModule, new_graph: fx.Graph, decomposition_table: Mapping[OpOverload, Callable] | None = None, **kwargs: object) -> None: ...
    def placeholder(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def get_attr(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def output(self, target: str, args: tuple[object, ...], kwargs: dict[str, object]) -> object: ...
    def run(self, *args: object, **kwargs: object) -> object: ...

class _ModuleNotInstalledAsSubmoduleError(NameError): ...

class _AttrProxy:
    def reset_proxy_mapping(self, base: Module, path: str) -> None: ...

class _ModuleStackTracer(PythonKeyTracer):
    '''Customized version of PythonKeyTracer that retains module stack
    information in node.meta["nn_module_stack"].

    FX symbolic trace actually does this already, but it relies on `self.root`
    being the actual module being traced. Since make_fx traces a lambda of our
    creation, things don\'t work properly.

    So for this version we hold onto a reference to the original module
    (scope_root) and use that to match the path. Also when we see,
            A
           / \\\n          B   C
           \\ /
            D
    we want to record the path as A.B.D by recording only one path.
    See Note [Preserving the nn module stack metadata during export non-strict mode]  # noqa: W605
    '''
    stack_trace: bool
    scope_root: Incomplete
    enable_attr_proxy: bool
    submodule_paths: Incomplete
    proxy_paths: WeakKeyDictionary[_AttrProxy, str]
    attr_proxy_map: WeakKeyDictionary[Module, _AttrProxy]
    proxy_modules: WeakKeyDictionary[_AttrProxy, Module]
    counter: int
    module_id_cache: Incomplete
    __class__: Incomplete
    __dict__: Incomplete
    proxy_type: Incomplete
    def __init__(self, scope_root: GraphModule) -> None: ...
    def path_of_module(self, mod: Module) -> str:
        """
        Use tracked access path during tracing instead of the default BFS behavior.
        Still use all the possible module paths to verify the result.
        """
    def getattr(self, attr: str, attr_val: object, parameter_proxy_cache: dict[str, Proxy]) -> object: ...
    def trace(self, root: Module | Callable, concrete_args: dict[str, object] | None) -> fx.Graph: ...
    def call_module(self, m: Module, forward: Callable, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        """PythonKeyTracer overrides call_module to avoid the scope handling,
        but we actually want it.
        """
    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool: ...
    def create_node(self, *args: object, **kwargs: object) -> fx.node.Node:
        """
        Create node and add on metadata.
        Add nn_module_stack here instead of TracerBase,
        since calls to make_fx() might not want to record module stack metadata.
        Add torch_fn by looking at torch_fn_metadata and torch_fn_counts.
        Add stack_trace by filtering out forward() stack frames.
        """

class _MakefxTracer:
    decomposition_table: dict[OpOverload, Callable]
    tracing_mode: str
    _allow_non_fake_inputs: bool
    pre_dispatch: bool
    record_module_stack: bool
    _allow_fake_constant: bool
    _error_on_data_dependent_ops: bool
    fake_tensor_mode: FakeTensorMode | None
    proxy_mode: nullcontext | ProxyTorchDispatchMode
    proxy_function_mode: nullcontext | PreDispatchTorchFunctionMode
    fx_tracer: PythonKeyTracer | None
    python_dispatcher_mode: nullcontext | Any
    torch_fn_metadata_mode: nullcontext | TorchFunctionMetadataMode
    stack_trace: Incomplete
    def __init__(self, decomposition_table: Mapping[OpOverload, Callable] | None, tracing_mode: str, _allow_non_fake_inputs: bool, pre_dispatch: bool, record_module_stack: bool, _allow_fake_constant: bool, _error_on_data_dependent_ops: bool, stack_trace: bool = False) -> None: ...
    def _checkpoint_modes(self) -> list[Any]: ...
    def _restore_modes(self, prev_fake_tensor_mode: FakeTensorMode | None, prev_proxy_mode: nullcontext | ProxyTorchDispatchMode, prev_proxy_function_mode: nullcontext | PreDispatchTorchFunctionMode, prev_fx_tracer: PythonKeyTracer | None, prev_python_dispatcher_mode: nullcontext | Any, prev_torch_fn_metadata_mode: nullcontext | TorchFunctionMetadataMode) -> None: ...
    @contextmanager
    def _init_modes_from_inputs(self, f: Callable, args: tuple[object, ...]) -> Generator[None, None, None]: ...
    def _construct_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None: ...
    @contextmanager
    def _init_modes_from_parent(self, parent_tracer: _MakefxTracer) -> Generator[None, None, None]: ...
    def _trace_inner(self, f: Callable, *args: object) -> GraphModule: ...
    def trace(self, f: Callable, *args: object) -> fx.GraphModule: ...
    def trace_subgraph(self, f: Callable, *args: object) -> GraphModule: ...

def make_fx(f: Callable, decomposition_table: Mapping[OpOverload, Callable] | None = None, tracing_mode: str = 'real', _allow_non_fake_inputs: bool = False, *, pre_dispatch: bool = False, record_module_stack: bool = False, _allow_fake_constant: bool = False, _error_on_data_dependent_ops: bool = True, stack_trace: bool = False) -> Callable[..., GraphModule]:
    '''
    Given a function f, return a new function which when executed with valid
    arguments to f, returns an FX GraphModule representing the set of operations that
    were executed during the course of execution.

    If stack_trace is True, the stack_trace will be preserved on node.meta["stack_trace"]
    '''
def get_innermost_proxy_mode() -> ProxyTorchDispatchMode | None: ...
def get_proxy_mode() -> ProxyTorchDispatchMode | None:
    """
    Current the currently active proxy tracing mode, or None if
    we are not currently tracing.  This includes pre-dispatch proxy
    tracing.
    """
def handle_sym_dispatch(func: Callable[_P, R], args: _P.args, kwargs: _P.kwargs) -> R:
    """
    Call into the currently active proxy tracing mode to do a
    SymInt/SymFloat/SymBool dispatch trace on a function that operates on
    these arguments.
    """
