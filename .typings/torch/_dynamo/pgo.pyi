import dataclasses
import enum
import functools
import types
from _typeshed import Incomplete
from collections import defaultdict
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch._dynamo.utils import CompileEventLogger as CompileEventLogger, dynamo_timed as dynamo_timed, set_feature_use as set_feature_use, warn_once as warn_once
from torch._environment import is_fbcode as is_fbcode
from torch._inductor.remote_cache import JsonDataTy as JsonDataTy, RemoteCache as RemoteCache
from torch._logging._internal import trace_structured_artifact as trace_structured_artifact
from torch.compiler._cache import CacheArtifact as CacheArtifact, CacheArtifactFactory as CacheArtifactFactory, CacheArtifactManager as CacheArtifactManager
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import TypeVar
from typing_extensions import Self, override

class ReservedWorkflowIdUserError(ValueError): ...

log: Incomplete
LOCK_TIMEOUT: int

@functools.cache
def _hash_containing_file(filepath: str) -> str: ...

@dataclasses.dataclass(frozen=True)
class CodeId:
    filename: str
    firstlineno: int
    name: str
    file_hash: str
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    def __str__(self) -> str: ...
    @staticmethod
    def make(code: types.CodeType) -> CodeId: ...

@dataclasses.dataclass
class CodeState:
    automatic_dynamic: defaultdict[str, FrameStateSizeEntry] = dataclasses.field(default_factory=Incomplete)

_INIT_CODE_STATE: defaultdict[CodeId, CodeState] | None
_CODE_STATE: defaultdict[CodeId, CodeState] | None

@dataclasses.dataclass(frozen=True)
class InferStride:
    """
    Denotes the quantity stride[dim] * size[dim], which is what the stride would
    be for the next physical dimension that results in a contiguous layout.

    For example, given size = [2, 3], stride = [3, 1], we can replace this with
    stride = [InferStride(1), 1], because InferStride(1) = stride[1] * size[1] = 1 * 3 = 3

    Indirecting the representation in this way is important for the join operation
    on strides as if we join [2, 3][3, 1] and [2, 4][4, 1],
    we don't want [2, None][None, 1] which would get eventually symbolized into
    [2, s0][s1, 1] (notice that the relationship between s0 and s1 is broken).
    If we instead rewrite the expressions as InferStride so we have [2, 3][InferStride(1), 1]
    and [2, 4][InferStride(1), 1] we now join to [2, None][InferStride(1), 1] will
    result in [2, s0][s0, 1], as desired.
    """
    dim: int
_T = TypeVar('_T')

class AutoUnset(enum.Enum):
    '''
    The identity element of our semilattice, a generic "don\'t know" element that
    is always subsumed when we get more information.
    '''
    token = 0

auto_unset: Incomplete

class AutoDynamic(enum.Enum):
    """
    The top element of our (bounded) semilattice, whenever you merge this with
    any other element you always get it again
    """
    token = 0

auto_dynamic: Incomplete

@dataclasses.dataclass
class FrameStateSizeEntry:
    scalar: int | AutoDynamic | AutoUnset = dataclasses.field(default=auto_unset)
    size: AutoDynamic | AutoUnset | tuple[int | AutoDynamic, ...] = dataclasses.field(default=auto_unset)
    stride: AutoDynamic | AutoUnset | tuple[int | AutoDynamic | InferStride, ...] = dataclasses.field(default=auto_unset)
    def render(self) -> str: ...
    def __post_init__(self) -> None: ...
    def is_size_dynamic(self, dim: int) -> bool: ...
    def is_stride_dynamic(self, dim: int) -> bool: ...
    @staticmethod
    def _munge_symint(xs: tuple[int, ...]) -> tuple[AutoDynamic | int, ...]: ...
    @classmethod
    def make_scalar(cls, x: int) -> FrameStateSizeEntry: ...
    @classmethod
    def make_tensor(cls, size: tuple[int, ...], stride: tuple[int, ...]) -> FrameStateSizeEntry: ...
    @classmethod
    def make_size(cls, size: tuple[int, ...]) -> FrameStateSizeEntry: ...
    @staticmethod
    def _merge_atom(x: _T, y: _T) -> AutoDynamic | _T: ...
    @classmethod
    def _merge_atom_tup(cls, xs: AutoDynamic | AutoUnset | tuple[_T, ...], ys: AutoDynamic | AutoUnset | tuple[_T, ...]) -> AutoDynamic | AutoUnset | tuple[AutoDynamic | _T, ...]: ...
    def __ior__(self, other: Self) -> Self: ...

def update_automatic_dynamic(tx: InstructionTranslator, name: str, entry: FrameStateSizeEntry, *, is_unspecialized_nn_module: bool = False) -> FrameStateSizeEntry: ...
def process_automatic_dynamic(tx: InstructionTranslator, name: str, entry: FrameStateSizeEntry, *, is_unspecialized_nn_module: bool = False) -> FrameStateSizeEntry: ...
def get_cache_key() -> str | None: ...
def code_state_path(cache_key: str) -> str | None: ...
def should_use_remote_dynamo_pgo_cache() -> bool: ...
def get_remote_cache() -> RemoteCache[JsonDataTy] | None: ...
def _collect_dynamic_sources(code_state: CodeState) -> OrderedSet[str]: ...
def log_frame_dynamic_whitelist(f_code: types.CodeType) -> None: ...
def render_code_state(cs: defaultdict[CodeId, CodeState]) -> str: ...

class PGOCacheArtifact(CacheArtifact):
    @override
    def populate_cache(self) -> None: ...
    @override
    @staticmethod
    def type() -> str: ...
    @staticmethod
    def _rewrite_cache_key_for_mega_cache(original_key: str) -> str:
        """
        The PGO cache artifact key for a MAST job contains the job name and the version.
        When we want to use the cache artifact on a different MAST job, we need to
        update the key to use the new MAST job's name and version.
        """

def get_code_state() -> defaultdict[CodeId, CodeState]: ...
def put_code_state() -> None: ...
def write_local_impl(cache_key: str, pickled_code: bytes) -> tuple[str, int] | None: ...
def put_local_code_state(cache_key: str) -> None: ...
def put_remote_code_state(cache_key: str) -> None: ...
def reset_code_state() -> None: ...
