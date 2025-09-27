import torch
from .ops_handler import DefaultHandler as DefaultHandler, KernelFormatterHandler as KernelFormatterHandler, MockHandler as MockHandler, OpsHandler as OpsHandler, ReductionType as ReductionType, StoreMode as StoreMode, WrapperHandler as WrapperHandler
from _typeshed import Incomplete
from contextlib import AbstractContextManager
from torch._inductor.choices import InductorChoices as InductorChoices
from torch._inductor.codegen.cpp_utils import LocalBufferContext as LocalBufferContext
from torch._inductor.debug import DebugContext as DebugContext
from torch._inductor.graph import GraphLowering as GraphLowering
from torch._inductor.loop_body import InterpreterShim as InterpreterShim
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable, Generic, TypeVar

threadlocal: Incomplete
T = TypeVar('T')

class NullHandler:
    """
    Sentinel indicating that a global variable is unset ala None.  Typically,
    attempting to access the global variable before it's set is an error, but with
    NullHandler it won't fail until you try to access an attribute on it.
    """

_PoisonedVirtual: Incomplete

class Virtualized(Generic[T]):
    '''
    Implements a global variable that redirects via thread local variable
    (NB: construct this class to create the global variable; this is not
    a singleton class!)

    This allows us to swap in different op implementations in codegen.

    NB: Despite the fact that we typically call these "handlers" (e.g., NullHandler is
    the default value of the variable), we sometimes use these variables to
    store other things, like booleans.
    '''
    _vname: Incomplete
    _key: str
    _default: Incomplete
    def __init__(self, vname: str, default: Callable[[], T] | type[NullHandler]) -> None: ...
    def _set_handler(self, value: T) -> AbstractContextManager[None]: ...
    def _get_handler(self, check_poisoned: bool = True) -> T: ...
    def __getattr__(self, name: str) -> Any: ...

class NullKernelHandler(NullHandler):
    """
    We need access `V.kernel.removed_buffers` in DeferredLine class when there
    is no kernel in the context. This happens when codegening the wrapper.
    Initialize `removed_buffers` and `inplaced_to_remove` explicitly so we don't
    need call 'getattr' with default value which is error prone to typo in
    attribute name.
    """
    removed_buffers: Incomplete
    inplaced_to_remove: Incomplete
    index_dtype: str
    def __init__(self) -> None: ...
    def get_index_dtype_as_torch_dtype(self): ...

_ops: Virtualized[OpsHandler[Any]]
_graph: Virtualized[GraphLowering]
_real_inputs: Virtualized[list[torch.Tensor]]
_fake_mode: Virtualized[FakeTensorMode]
_kernel: Virtualized[NullKernelHandler]
_debug: Virtualized[DebugContext]
_interpreter: Virtualized[InterpreterShim]
_aot_compilation: Virtualized[bool]
_current_node: Virtualized[torch.fx.Node]
_local_buffer_context: Virtualized[LocalBufferContext]

def _choices_default():
    """
    Lazy init the global choices handler

    We virtualize InductorChoices to allow changing inductor heuristics from out of tree.
    """

_choices: Virtualized[InductorChoices]

class OpsValue:
    """The return type of most ops calls.

    This exists so we can overload magic methods, and write mathematical
    expressions much more fluently. So instead of

        ops.add(ops.mul(ops.mul(ops.sub(ops.mul(_Ap2, x), _Ap3), x), x), _1)

    we can write

        (_Ap2 * x - _Ap3) * x * x + _1

    """
    value: Any
    def __init__(self, value) -> None: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __add__(self, other): ...
    def __mul__(self, other): ...
    def __sub__(self, other): ...
    def __neg__(self): ...
    def __truediv__(self, other): ...
    def __floordiv__(self, other): ...
    def __mod__(self, other): ...
    def __pow__(self, other): ...
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __gt__(self, other): ...
    def __ge__(self, other): ...
    def __and__(self, other): ...
    def __or__(self, other): ...
    def __xor__(self, other): ...
    def __invert__(self): ...
    def __rshfit__(self, n): ...
    def __lshift__(self, n): ...

class OpsWrapper(DefaultHandler):
    """This wraps any returned IR values into an `OpsValue` instance, so that we
    can overload the magic methods for writing mathematical expressions fluently.
    """
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any: ...
    @staticmethod
    def _unwrap(x): ...
    @staticmethod
    def _wrap(x): ...
    @staticmethod
    def indirect_indexing(index, size, check: bool = True, wrap_neg: bool = True): ...

ops: OpsHandler[Any]

class _V:
    MockHandler = MockHandler
    KernelFormatterHandler = KernelFormatterHandler
    WrapperHandler = WrapperHandler
    set_ops_handler: Callable[[OpsHandler[Any]], AbstractContextManager[None]]
    get_ops_handler: Callable[[], OpsHandler[Any]]
    set_graph_handler: Callable[[GraphLowering], Any]
    set_real_inputs: Callable[[Any], Any]
    get_real_inputs: Callable[[], Any]
    set_fake_mode: Callable[[Any], Any]
    get_fake_mode: Callable[[], Any]
    set_kernel_handler: Callable[[Any], Any]
    set_debug_handler: Callable[[Any], Any]
    set_interpreter_handler: Callable[[Any], Any]
    set_aot_compilation: Callable[[bool], Any]
    get_aot_compilation: Callable[[], Any]
    set_current_node: Callable[[Any], Any]
    get_current_node: Callable[[], Any]
    set_local_buffer_context: Callable[[Any], Any]
    get_local_buffer_context: Callable[[], Any]
    set_choices_handler: Callable[[Any], Any]
    @property
    def ops(self) -> OpsHandler[Any]:
        """The operator handler specific to the current codegen task"""
    @property
    def graph(self) -> GraphLowering:
        """The graph currently being generated"""
    @property
    def real_inputs(self):
        """non-fake example inputs"""
    @property
    def fake_mode(self):
        """The graph currently being generated"""
    @property
    def kernel(self):
        """The kernel currently being generated"""
    @property
    def debug(self): ...
    @property
    def interpreter(self): ...
    @property
    def aot_compilation(self): ...
    @property
    def current_node(self): ...
    @property
    def local_buffer_context(self): ...
    @property
    def choices(self) -> InductorChoices: ...

V: Incomplete
