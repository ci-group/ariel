import contextlib
import torch
import types
from _typeshed import Incomplete
from collections import deque
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from torch._C import DispatchKey as DispatchKey, _get_dispatch_stack_at as _get_dispatch_stack_at, _len_torch_dispatch_stack as _len_torch_dispatch_stack, _pop_torch_dispatch_stack as _pop_torch_dispatch_stack, _push_on_torch_dispatch_stack as _push_on_torch_dispatch_stack
from typing import Any, Protocol, overload
from typing_extensions import TypeIs

_is_in_torch_dispatch_mode: bool
_is_in_non_infra_torch_dispatch_mode: bool

def is_in_torch_dispatch_mode(include_infra_modes: bool = True) -> bool: ...

class TorchDispatchMode:
    """
    A ``TorchDispatchMode`` allows you to override the meaning of all
    ``__torch_dispatch__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchDispatchMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_dispatch__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_dispatch__`` implementation, either explicitly
    invoke ``self.__torch_dispatch__(...)``, or use the context manager
    ``__torch_dispatch__(self)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """
    old_dispatch_mode_flags: deque[bool]
    old_non_infra_dispatch_mode_flags: deque[bool]
    def __init__(self, _dispatch_key=None) -> None: ...
    def _lazy_init_old_dispatch_mode_flags(self) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    @classmethod
    def push(cls, *args, **kwargs): ...
    @classmethod
    def is_infra_mode(cls): ...

def _get_current_dispatch_mode(): ...
def _detect_infra_mode(key): ...
def _unset_infra_mode(key): ...
def _disable_infra_mode(key) -> Generator[Incomplete]: ...
def _get_current_dispatch_mode_stack(): ...
def _push_mode(mode: TorchDispatchMode): ...
def _pop_mode(k: DispatchKey | torch._C._TorchDispatchModeKey | None = None): ...
@contextlib.contextmanager
def _pop_mode_temporarily(k: DispatchKey | None = None): ...
@contextlib.contextmanager
def _disable_current_modes() -> Generator[Incomplete]: ...

class BaseTorchDispatchMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...

class TensorWithFlatten(Protocol):
    def __tensor_flatten__(self) -> tuple[Sequence[str], object]: ...
    @staticmethod
    def __tensor_unflatten__(inner_tensors: int, flatten_spec: int, outer_size: int, outer_stride: int) -> torch.Tensor: ...
    shape: torch._C.Size
    @overload
    def stride(self, dim: None = None) -> tuple[int, ...]: ...
    @overload
    def stride(self, dim: int) -> int: ...
    @overload
    def size(self, dim: None = None) -> tuple[int, ...]: ...
    @overload
    def size(self, dim: int) -> int: ...
    def storage_offset(self) -> int: ...
    def dim(self) -> int: ...
    @overload
    def to(self, dtype: torch.types._dtype, non_blocking: bool = False, copy: bool = False, *, memory_format: torch.memory_format | None = None) -> torch.Tensor: ...
    @overload
    def to(self, device: torch._prims_common.DeviceLikeType | None = None, dtype: torch.types._dtype | None = None, non_blocking: bool = False, copy: bool = False, *, memory_format: torch.memory_format | None = None) -> torch.Tensor: ...
    @overload
    def to(self, other: torch.Tensor, non_blocking: bool = False, copy: bool = False, *, memory_format: torch.memory_format | None = None) -> torch.Tensor: ...

def is_traceable_wrapper_subclass(t: object) -> TypeIs[TensorWithFlatten]:
    """
    Returns whether or not a tensor subclass that implements __torch_dispatch__
    is 'traceable' with torch.compile.
    In order for a tensor subclass to support TorchDispatchMode-style tracing in PT2,
    It must implement two magic methods: __tensor_flatten__ and __tensor_unflatten__.
    It is also expected to obey some restrictions around traceability and aliasing:
        * The subclass's __torch_dispatch__() implementation should desugar into pytorch
            dispatcher operations that can be traced into a graph.
        * The subclass should use return_and_correct_aliasing(). This is needed today to make
            sure that torch.compile does the right thing in a few cases around input mutation
            and output aliasing.

    Expected magic method signatures:
        attrs, ctx = t.__tensor_flatten__()
            attrs: list of attribute name strings for inner tensors
            ctx: dict containing any other subclass-specific metadata needed for unflattening

        t = MySubClass.__tensor_unflatten__(inner_tensors, ctx, outer_size, outer_stride)
            inner_tensors: dict mapping attribute name -> tensor for each inner tensor
            ctx: dict with subclass metadata in the form that __tensor_flatten__() produces
            outer_size: expected (possibly symbolic) size that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the shape info to be constructed. In most cases, this arg can be
                safely ignored.
            outer_stride: expected (possibly symbolic) stride that the returned subclass
                instance should have. Note that this arg is useful for certain subclasses
                that require the stride info to be constructed. In most cases, this arg can be
                safely ignored.
    """
def is_traceable_wrapper_subclass_type(t: type) -> TypeIs[type[TensorWithFlatten]]:
    """Same as above, but takes a type argument instead of an instance."""
def transform_subclass(t, callback, outer_size=None, outer_stride=None):
    """
    Given a traceable, wrapper tensor subclass ``t`` that implements
    ``__torch_dispatch__`` and holds some inner tensors,
    and a callback of type ``Callable[[str, torch.Tensor], torch.Tensor]``,
    `transform_subclass` will construct a fresh instance of the wrapper tensor subclass.
    It will do so by grabbing each inner tensor attribute from the wrapper,
    passing them into ``callback`` to get a transformed tensor,
    and putting each transformed tensor into the fresh tensor subclass instance.

    Note: this function will not handle ensuring that the fresh subclass
    gets the same (autograd, and aliasing) metadata as the original tensor.
    This is generally handled in other subsystems like AOTAutograd.
    """
def _correct_storage_aliasing(func, schema_info, args, outs):
    """
    Given: an OpOverload, a SchemaInfo (cached information from torchgen about schema),
    and the inputs/outputs to the OpOverload,
    this function checks to see if func is a view operator
    (by checking if any of the outputs in the op's schema
     are immutable aliases of inputs).
    If so, this function manually aliases the storage of the output tensor
    with its corresponding input tensor alias.
    It does this by unsafely overwriting the storage field of the output tensor
    to be the same storage as the input.
    """

@dataclass
class AliasInfo:
    alias_set: set[str]
    is_write: bool
    name: str | None

@dataclass
class SchemaInfo:
    args: list[AliasInfo]
    outs: list[AliasInfo]

parsed_schema_map: dict[Any, SchemaInfo]

def get_alias_info(func) -> SchemaInfo: ...
def return_and_correct_aliasing(func, args, kwargs, out):
    """
    This function should be used by wrapper tensor ``__torch_dispatch__`` subclasses
    that would like to work with torch.compile. It ensures that the subclass
    properly implements the aliasing behavior of every op,
    which is needed for correctness in AOTAutograd.
    This function will handle:

        * When we see a view op, we will alias the storages of any
          input and output tensor subclasses

        * When we see an inplace or out= op, we will directly
          return the corresponding input tensor, instead of returning
          a (potentially) fresh output tensor.
    """
