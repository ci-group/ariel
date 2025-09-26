import contextlib
import dataclasses
import functools
import types
from .bytecode_transformation import get_code_keys as get_code_keys
from _typeshed import Incomplete
from collections.abc import Generator
from torch._dynamo.precompile_context import PrecompileCacheArtifact as PrecompileCacheArtifact, PrecompileContext as PrecompileContext
from torch.compiler._cache import CacheArtifactFactory as CacheArtifactFactory
from typing import Any

logger: Incomplete

@dataclasses.dataclass(frozen=True)
class SerializedCode:
    co_argcount: int
    co_posonlyargcount: int
    co_kwonlyargcount: int
    co_nlocals: int
    co_stacksize: int
    co_flags: int
    co_code: bytes
    co_consts: tuple[Any, ...]
    co_names: tuple[str, ...]
    co_varnames: tuple[str, ...]
    co_filename: str
    co_name: str
    co_firstlineno: int
    co_cellvars: tuple[str, ...]
    co_freevars: tuple[str, ...]
    co_linetable: bytes | None = ...
    co_qualname: str | None = ...
    co_exceptiontable: bytes | None = ...
    co_lnotab: str | None = ...
    @classmethod
    @functools.cache
    def from_code_object(cls, code: types.CodeType) -> SerializedCode: ...
    @classmethod
    @functools.cache
    def to_code_object(cls, serialized_code: SerializedCode) -> types.CodeType: ...

@dataclasses.dataclass
class _GuardedCodeCacheEntry:
    """
    Contains the serializable information associated with a single compilation in dynamo.
    To restore an execution of compiled code, we will need to serialize the following data:
      - Dynamo bytecode for mapping Python inputs/outputs.
      - Dynamo guards.
    """
    guards_state: bytes
    dynamo_code: SerializedCode

_BackendId: Incomplete
_FunctionId: Incomplete

@dataclasses.dataclass
class _DynamoCodeCacheEntry:
    '''
    Contains the serializable information associated with a single code object
    in dynamo. To restore an execution of compiled code, we will need the following
    ingredients:
      1. The "original" code object, which serves as the entry point for eager
         execution, i.e. the code only executed when there\'s no cache entry hit.
      2. The python module name this code object belongs to, for identifying the
         enclosing global scope to inject compiled and resume functions.
      3. A list of function names that pointing to this code object. There could be
         multiple function objects pointing to the same code such as recursive functions.
      4. A list of guarded code that eval frame dispatches to.
      5. A list of imported module objects unioned from all compiled branches.
      6. A list of "backends" (compiled fx graph) unioned from all compield branches.
    '''
    python_code: SerializedCode
    python_module: str
    function_names: list[_FunctionId]
    guarded_codes: list[_GuardedCodeCacheEntry]
    import_sources: dict[str, str]
    backend_ids: list[_BackendId]

@dataclasses.dataclass
class _DynamoCacheEntry:
    codes: list[_DynamoCodeCacheEntry]
    python_version: str = ...
    torch_version: str = ...
    @property
    def backend_ids(self) -> set[_BackendId]: ...

class _DynamoCacheArtifact(PrecompileCacheArtifact[_DynamoCacheEntry]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> _DynamoCacheEntry: ...

class CompilePackage:
    """
    CompilePackage is considered a low level component and should not be directly exposed to
    end users. It has the following interface:

    1. `CompilePackage.__init__()` which optionally takes previously serialized dynamo states.
        a. when `dynamo` argument is None, it will construct a brand new CompilePackage object.
        b. when `dynamo` argument is not None, it will load a pre-compiled dynamo state.
    2. `package.save()` which dumps the dynamo and backend states to a DynamoCacheEntry object.
    3. `package.install(backends) which will handle all the side-effectful global scope
        updates with compiled functions and resume functions.
    """
    _innermost_fn: Incomplete
    _codes: dict[types.CodeType, _DynamoCodeCacheEntry]
    _current_entry: _DynamoCodeCacheEntry | None
    _installed_globals: dict[types.ModuleType, list[str]]
    _cached_backends: dict[_BackendId, Any]
    def __init__(self, fn: Any, dynamo: _DynamoCacheEntry | None = None) -> None: ...
    def _initialize(self, fn: Any, dynamo: _DynamoCacheEntry | None = None) -> None: ...
    def _add_function(self, python_code: types.CodeType, python_module: str, name: _FunctionId | None = None) -> None: ...
    @property
    def cached_backends(self) -> dict[_BackendId, Any]: ...
    @functools.cached_property
    def source_id(self) -> str: ...
    @contextlib.contextmanager
    def code_context(self, code: types.CodeType) -> Generator[None, None, None]: ...
    def add_guarded_code(self, guards_state: bytes, dynamo_code: types.CodeType) -> None: ...
    def add_resume_function(self, python_code: types.CodeType, python_module: str, name: str | None) -> None: ...
    def add_import_source(self, alias: str, module_name: str) -> None: ...
    def add_backend_id(self, backend_id: str, backend: Any | None = None) -> None: ...
    def validate(self) -> None: ...
    def _install_global(self, module: types.ModuleType, name: str, value: Any) -> None: ...
    def uninstall(self) -> None: ...
    def install(self, backends: dict[_BackendId, Any]) -> None:
        """
        Sync the package states to the compiled function. This includes the following actions:
          1. Clean up the previously installed states.
          2. Install the compiled functions to global scopes.
          3. Install the precompiled cache entries to ExtraStates on the code object.
        """
    def cache_entry(self) -> _DynamoCacheEntry: ...

class EagerCacheArtifact(PrecompileCacheArtifact[Any]):
    @staticmethod
    def type() -> str: ...
    def after_deserialization(self) -> Any: ...

class DynamoStore:
    """
    A DynamoStore tracks active CompilePackages, and provides methods to store and retrieve them.
    """
    def record_package(self, package: CompilePackage) -> None:
        """Records a package to PrecompileContext, so that it can be serialized later."""
    def record_eager_backend(self, backend_id: _BackendId, backend: Any) -> None:
        """Records eager fx graphs to PrecompileContext for testing purposes."""
    def save_package(self, package: CompilePackage, path: str) -> None:
        """Saves a package to a given path. Grabs backends from PrecompileContext."""
    def load_package(self, fn: Any, path: str) -> tuple[CompilePackage, dict[_BackendId, Any]]:
        """Loads a package from a given path and returns it plus a list of deserialized backends"""
