import atexit
import contextlib
import functools
import threading
import torch
import weakref
from ._fake_tensor_utils import _CacheKeyState as _CacheKeyState, _PySymInputStub as _PySymInputStub, _SymIntOutputStub as _SymIntOutputStub
from _typeshed import Incomplete
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from torch import SymBool as SymBool, SymFloat as SymFloat, SymInt as SymInt, Tensor as Tensor
from torch._C._functorch import is_functorch_wrapped_tensor as is_functorch_wrapped_tensor, is_legacy_batchedtensor as is_legacy_batchedtensor
from torch._guards import Source as Source
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._library.fake_profile import MissingOpProfile as MissingOpProfile
from torch._logging import dtrace_structured as dtrace_structured
from torch._ops import OpOverload as OpOverload
from torch._prims_common import suggest_memory_format as suggest_memory_format
from torch._subclasses.fake_impls import _device_not_kwarg_ops as _device_not_kwarg_ops, _is_tensor_constructor as _is_tensor_constructor, _like_tensor_constructors as _like_tensor_constructors, contains_tensor_types as contains_tensor_types, get_fast_op_impls as get_fast_op_impls, has_meta as has_meta, op_implementations_checks as op_implementations_checks, stride_incorrect_op as stride_incorrect_op
from torch._subclasses.meta_utils import MetaConverter as MetaConverter, assert_eq as assert_eq, assert_metadata_eq as assert_metadata_eq, is_sparse_any as is_sparse_any, is_sparse_compressed as is_sparse_compressed
from torch._utils import render_call as render_call
from torch.fx.experimental.symbolic_shapes import ShapeEnv as ShapeEnv, SymbolicContext as SymbolicContext
from torch.fx.immutable_collections import immutable_dict as immutable_dict
from torch.fx.operator_schemas import normalize_function as normalize_function
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.overrides import TorchFunctionMode as TorchFunctionMode
from torch.types import IntLikeType as IntLikeType, py_sym_types as py_sym_types
from torch.utils._backport_slots import dataclass_slots as dataclass_slots
from torch.utils._mode_utils import no_dispatch as no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode as TorchDispatchMode, is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from torch.utils._pytree import KeyPath as KeyPath, PyTree as PyTree, TreeSpec as TreeSpec, keystr as keystr, tree_map as tree_map, tree_map_ as tree_map_
from torch.utils._stats import count as count
from torch.utils._traceback import CapturedTraceback as CapturedTraceback
from types import TracebackType
from typing import Any, Literal, TypeVar
from typing_extensions import Self, TypeGuard
from weakref import ReferenceType

log: Incomplete
hc_log: Incomplete
not_implemented_log: Incomplete
DimList = list
pytree = torch.utils._pytree
T = TypeVar('T')
aten: Incomplete
CONSTANT_NUMEL_LIMIT: int
RECURSION_COUNT: int

class IncrementRecursionCount:
    def __init__(self) -> None: ...
    def __del__(self) -> None: ...

@dataclass
class UnsupportedFakeTensorException(RuntimeError):
    reason: str

@dataclass
class DynamicOutputShapeException(RuntimeError):
    func: OpOverload

@dataclass
class DataDependentOutputException(RuntimeError):
    func: OpOverload

@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload

@dataclass
class UnsupportedMutationAliasingException(RuntimeError):
    reason: str

@dataclass
class MetadataMismatchError(RuntimeError):
    reason: str

class FakeTensorTLS(threading.local):
    allow_non_fake_inputs_override: bool | None
    def __init__(self) -> None: ...

fake_tensor_tls: Incomplete

def ordered_set(*items: T) -> dict[T, Literal[True]]: ...
@contextlib.contextmanager
def unset_fake_temporarily() -> Generator[TorchDispatchMode | None, None, None]: ...
@contextlib.contextmanager
def disable_fake_tensor_cache(fake_mode: FakeTensorMode) -> Generator[None, None, None]: ...
def get_plain_tensors(subclass: Tensor, *, out: list[Tensor | int | SymInt]) -> list[Tensor | int | SymInt]: ...
def is_fake(x: object) -> TypeGuard[Tensor]: ...
def maybe_get_fake_mode(t: object) -> FakeTensorMode | None: ...
@functools.cache
def get_schema_info(func: OpOverload) -> torch._C._SchemaInfo: ...
@functools.cache
def torch_decomp_decompositions(func: OpOverload) -> bool: ...
def tree_flatten_only(ty: type[T], tree: PyTree) -> list[T]: ...
def _is_plain_tensor(t: object) -> bool: ...

class FakeTensorConverter:
    @property
    def tensor_memo(self) -> weakref.WeakValueDictionary: ...
    meta_converter: MetaConverter
    constant_storage_mapping: dict[StorageWeakRef, list[ReferenceType]]
    export: bool
    def __init__(self, *, copy_data: bool = False, export: bool = False) -> None: ...
    def add_constant_storage_mapping(self, fake_tensor: FakeTensor) -> None: ...
    def invalidate_constant_aliases(self, tensor: Tensor) -> None: ...
    def _get_memo(self, t: Tensor) -> FakeTensor | None: ...
    def set_tensor_memo(self, t: Tensor, v: FakeTensor) -> None: ...
    def from_real_tensor(self, fake_mode: FakeTensorMode, t: Tensor, make_constant: bool = False, shape_env: ShapeEnv | None = None, *, source: Source | None = None, symbolic_context: SymbolicContext | None = None, trace: bool = True) -> FakeTensor: ...
    def from_meta_and_device(self, fake_mode: FakeTensorMode, t: Tensor, device: torch.device, pytype: type[torch.Tensor] | None = None, dispatch_keys: torch.DispatchKeySet | None = None) -> FakeTensor: ...

@functools.cache
def init_gpu_context(device: torch.device) -> None: ...
@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode: FakeTensorMode) -> Generator[None, None, None]: ...
def should_allow_numbers_as_tensors(func: OpOverload) -> bool: ...

class FakeTensorConfig:
    debug: Incomplete

class SymNumberMemoDescriptor:
    _name: str
    _is_nested_int: bool
    def __init__(self, *, is_nested_int: bool = False) -> None: ...
    def __set_name__(self, owner: str, name: str) -> None: ...
    def _memo(self, obj: FakeTensor) -> str: ...
    def _memo_vc(self, obj: FakeTensor) -> str: ...
    def _memo_epoch(self, obj: FakeTensor) -> str: ...
    def __get__(self, obj: FakeTensor, objtype: type[FakeTensor] | None = None) -> torch.SymInt | torch.SymFloat | None: ...
    def __set__(self, obj: FakeTensor, value: torch.SymInt | torch.SymFloat | None) -> None: ...

class FakeTensor(Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """
    fake_device: torch.device
    fake_mode: FakeTensorMode
    constant: Tensor | None
    real_tensor: Tensor | None
    nonzero_memo: Incomplete
    item_memo: Incomplete
    unique_memo: Incomplete
    unique_consecutive_memo: Incomplete
    nested_int_memo: Incomplete
    pytype: type[Tensor] | None
    dispatch_keys: torch.DispatchKeySet | None
    _mode_key: Incomplete
    @property
    def device(self) -> torch.device: ...
    @device.setter
    def device(self, _: torch.device) -> None: ...
    @property
    def names(self) -> list[str]: ...
    @names.setter
    def names(self, _: list[str]) -> None: ...
    _debug_trace: Incomplete
    @staticmethod
    def __new__(cls, fake_mode: FakeTensorMode, elem: Tensor, device: torch.device, constant: Tensor | None = None, real_tensor: Tensor | None = None, pytype: type[Tensor] | None = None, dispatch_keys: torch.DispatchKeySet | None = None) -> Self: ...
    def __init__(self, *args: object, **kwargs: object) -> None: ...
    @staticmethod
    def from_tensor(t: Tensor, fake_mode: FakeTensorMode) -> FakeTensor: ...
    @classmethod
    @count
    def __torch_dispatch__(cls, func: OpOverload, types: Sequence[type], args: Sequence[object] = (), kwargs: Mapping[str, object] = ...) -> object: ...
    @staticmethod
    def _find_common_device(func: OpOverload, flat_args: Sequence[object]) -> tuple[torch.device, bool]: ...
    def get_nested_int(self, *, coeff: int | torch.SymInt = 1) -> torch.SymInt: ...
    def tolist(self) -> Any: ...

_MetadataIntLike: Incomplete

@dataclass
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """
    dtype: torch.dtype
    shape: tuple[_MetadataIntLike, ...]
    stride: tuple[_MetadataIntLike, ...]
    device: torch.device
    layout: torch.layout
    memory_format: torch.memory_format | None
    storage_offset: _MetadataIntLike
    storage_bytes: _MetadataIntLike | None
    requires_grad: bool
    is_quantized: bool
    is_conj: bool
    is_neg: bool
    is_inference: bool
    is_sparse: bool
    is_coalesced: bool | None
    dense_dim: int | None
    sparse_dim: int | None
    def _flatten_into(self, result: list[object], mode: FakeTensorMode, state: _CacheKeyState) -> None: ...

def extract_tensor_metadata(t: Tensor) -> TensorMetadata:
    """
    Extract the TensorMetadata of a tensor.
    """

@dataclass
class _DispatchCacheKey:
    """
    Key for the FakeTensor dispatch cache.
    """
    key: tuple[object, ...]
    hashvalue: int
    def __init__(self, tup: tuple[object, ...]) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def strip_shape_env(self) -> None: ...

class SingletonConstant: ...

@dataclass(frozen=True)
class _DispatchCacheEntryOutputInfo:
    """
    Entry type for the FakeTensor dispatch cache for an output. Accounts for three
    possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a
       given index.
    2) We need to synthesize a new FakeTensor given tensor metadata. For view
       ops, we further capture the index of the arg to alias.
    3) if the tensor related fields are None, then it is a constant value (e.g.
    None or integer)
    """
    inplace_idx: int | None
    metadata: TensorMetadata | None
    view_idx: int | None
    constant_value: Any | None = ...

@dataclass(frozen=True)
class _DispatchCacheValidEntry:
    """
    Entry type for the FakeTensor dispatch cache. It supports two types of outputs
    1) tensor
    2) tuple of tensors

    is_output_tuple flag helps in differentiating the return type
    """
    output_infos: tuple[_DispatchCacheEntryOutputInfo]
    is_output_tuple: bool = ...

@dataclass(frozen=True)
class _DispatchCacheBypassEntry:
    """
    Entry type for a negative cache entry.
    """
    reason: str
_DispatchCacheEntry = _DispatchCacheValidEntry | _DispatchCacheBypassEntry

@dataclass(frozen=True)
class _BypassDispatchCache(Exception):
    """
    Signals cases that should skip FakeTensor caching.
    """
    reason: str

@dataclass(frozen=True)
class DispatchCacheInfo:
    """
    Information about the state of the FakeTensor dispatch cache.
    """
    hits: int
    misses: int
    bypasses: dict[str, int]
    size: int

class FakeTensorMode(TorchDispatchMode):
    cache: dict[_DispatchCacheKey, _DispatchCacheEntry]
    cache_hits: int
    cache_misses: int
    cache_bypasses: dict[str, int]
    epoch: int
    in_kernel_invocation: bool
    static_shapes: bool
    shape_env: ShapeEnv | None
    _stack: str | None
    allow_meta: bool
    nt_tensor_id_counter: int
    nt_tensor_id_initial_count: int
    allow_fallback_kernels: Incomplete
    propagate_real_tensors: Incomplete
    fake_tensor_converter: Incomplete
    allow_scalar_outputs: bool
    _allow_unsafe_data_ptr_access: Incomplete
    cache_enabled: bool
    cache_crosscheck_enabled: Incomplete
    allow_non_fake_inputs: Incomplete
    enter_stack: list[tuple[bool, TorchDispatchMode | None, bool | None]]
    _stack_trace: Incomplete
    _mode_key: Incomplete
    def __init__(self, *, allow_fallback_kernels: bool = True, allow_non_fake_inputs: bool = False, shape_env: ShapeEnv | None = None, static_shapes: bool | None = None, export: bool = False) -> None: ...
    def reset_nt_tensor_id_counter(self) -> None: ...
    def is_our_fake(self, t: object) -> TypeGuard[FakeTensor]: ...
    @property
    def avoid_device_init(self) -> bool: ...
    @property
    def stack(self) -> str: ...
    @count
    def __torch_dispatch__(self, func: OpOverload, types: Sequence[type], args: Sequence[object] = (), kwargs: Mapping[str, object] = ...) -> object: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, a: type[BaseException] | None, b: BaseException | None, c: TracebackType | None) -> None: ...
    @classmethod
    def is_infra_mode(cls) -> bool: ...
    @classmethod
    def cache_info(cls) -> DispatchCacheInfo:
        """
        Query the state of the dispatch cache.
        """
    @classmethod
    def cache_clear(cls) -> None:
        """
        Clear the dispatch cache.
        """
    def _cached_dispatch_impl(self, func: OpOverload, types: Sequence[type], args: Sequence[object], kwargs: Mapping[str, object]) -> object:
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
    def _cache_key(self, state: _CacheKeyState, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]) -> _DispatchCacheKey:
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
    def _validate_cache_key(self, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object]) -> None:
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
    def _prep_args_for_hash(self, result: list[object], args: Mapping[str, object] | Sequence[object] | Iterable[object], state: _CacheKeyState, id_hashed_objects: list[object]) -> None:
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
    def _validate_output_for_cache_entry(self, state: _CacheKeyState, key: _DispatchCacheKey, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object], output: FakeTensor | None) -> None: ...
    def _get_output_info_for_cache_entry(self, state: _CacheKeyState, key: _DispatchCacheKey, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object], output: FakeTensor) -> _DispatchCacheEntryOutputInfo: ...
    def _make_cache_entry(self, state: _CacheKeyState, key: _DispatchCacheKey, func: OpOverload, args: Sequence[object], kwargs: Mapping[str, object], output: FakeTensor | None) -> _DispatchCacheValidEntry:
        """
        Make a cache entry object for the given 'output' Tensor. Raises
        _BypassDispatchCache if the output tensor has characteristics that
        prevent caching it.
        """
    def _get_output_tensor_from_cache_entry(self, state: _CacheKeyState, entry: _DispatchCacheEntryOutputInfo, key: _DispatchCacheKey, func: OpOverload, args: Sequence[object]) -> FakeTensor | None: ...
    def _output_from_cache_entry(self, state: _CacheKeyState, entry: _DispatchCacheValidEntry, key: _DispatchCacheKey, func: OpOverload, args: Sequence[object]) -> FakeTensor | None | tuple[FakeTensor | None, ...]:
        """
        Create a new FakeTensor from the cache entry.
        """
    def _crosscheck_cache_output(self, output: FakeTensor | None | tuple[FakeTensor | None, ...], func: OpOverload, types: Sequence[type], args: Sequence[object], kwargs: Mapping[str, object]) -> None:
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """
    def dispatch(self, func: OpOverload, types: Sequence[type], args: Sequence[object] = (), kwargs: Mapping[str, object] = ...) -> object: ...
    def _maybe_infer_fake(self, func: OpOverload, path: KeyPath, fake: object, real: object) -> tuple[object | None, bool]:
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched.
        Returns tuple of object & boolean, for whether or not it was overwrriten
        """
    def _maybe_infer_fake_kernel_from_pytree_out(self, func: OpOverload, fake_in: object, real_in: object, fake_out: object, real_out: object) -> object | None:
        """
        Helper to cross-check fake/real output properties & values,
        and create new fake vals if mismatched, but at the kernel level.
        Means this handles pytree outputs & checks aliasing.
        """
    def _dispatch_impl(self, func: OpOverload, types: Sequence[type], args: Sequence[object], kwargs: Mapping[str, object]) -> FakeTensor | None: ...
    _can_run_unsafe_fallback_allowed_namespaces: Incomplete
    def can_run_unsafe_fallback(self, func: OpOverload) -> bool: ...
    def validate_and_convert_non_fake_tensors(self, func: OpOverload, converter: FakeTensorConverter, flat_args: Sequence[object], args_spec: TreeSpec) -> tuple[list[object], list[FakeTensor]]:
        """
        Checks if the list of tensors are fake tensors.
        If not, try to convert them to fake tensors.
        Returns the original args, kwargs, and a flattened list of (args, kwargs) that are fake tensors.
        """
    def wrap_meta_outputs_with_default_device_logic(self, r: object, func: OpOverload, flat_args: Sequence[object], device: torch.device) -> PyTree: ...
    def create_symbolic_nested_int(self, *, nt_tensor_id: int | None = None) -> torch.SymInt: ...
    _cpp_meta_supports_symint: Incomplete
    def cpp_meta_supports_symint(self, func: OpOverload) -> bool: ...
    lift_fns: Incomplete
    def may_turn_const(self, t: Tensor) -> bool: ...
    def invalidate_written_to_constants(self, func: OpOverload, flat_arg_fake_tensors: Sequence[FakeTensor], args: Sequence[object], kwargs: Mapping[str, object]) -> None: ...
    def from_tensor(self, tensor: Tensor, *, static_shapes: bool | None = None, source: Source | None = None, symbolic_context: SymbolicContext | None = None, trace: bool = True) -> FakeTensor: ...
_StoragePointer = object

def _has_unrepresented_symbols(state: _CacheKeyState, output: FakeTensor | None) -> bool: ...
def run_fallback_kernel(fake_mode: FakeTensorMode, func: OpOverload, flat_args: Sequence[object], args_spec: PyTree, orig_not_implemented_exception: RuntimeError) -> FakeTensor: ...
def _set_cache_key_for_shape_env(cache: dict[_DispatchCacheKey, _DispatchCacheEntry], key: _DispatchCacheKey, entry: _DispatchCacheEntry) -> None: ...
def _set_cache_key(cache: dict[_DispatchCacheKey, _DispatchCacheEntry], key: _DispatchCacheKey, entry: _DispatchCacheEntry) -> None: ...

class FakeCopyMode(TorchFunctionMode):
    fake_mode: Incomplete
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...
    def __torch_function__(self, func: OpOverload, types: Sequence[type], args: Sequence[object] = (), kwargs: Mapping[str, object] | None = None) -> FakeTensor: ...

def _device_handler(args: Sequence[object]) -> torch.device: ...
def _check_for_subclass(flat_args: Sequence[object]) -> bool: ...
def _check_for_subclass_arg(x: object) -> bool: ...

_DISPATCH_META_HANDLERS: Incomplete
_DISPATCH_HANDLE_DIRECTLY: Incomplete

def evict_fake_tensor_cache_key(key: _DispatchCacheKey) -> None: ...
@atexit.register
def dump_cache_stats() -> None: ...
def _infer_fake_from_real_tensor(mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: torch.Tensor) -> torch.Tensor: ...
def inferred_fake_kernel_from_real_out(mode: FakeTensorMode, op: torch._ops.OpOverload, real_out: Any) -> Any: ...
