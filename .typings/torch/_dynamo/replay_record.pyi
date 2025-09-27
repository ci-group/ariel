import dataclasses
from _typeshed import Incomplete
from dataclasses import field
from torch.utils._import_utils import import_dill as import_dill
from types import CellType, CodeType, ModuleType
from typing import Any, IO
from typing_extensions import Self

dill: Incomplete

@dataclasses.dataclass
class ModuleRecord:
    module: ModuleType
    accessed_attrs: dict[str, Any] = field(default_factory=dict)

@dataclasses.dataclass
class DummyModule:
    name: str
    is_torch: bool = ...
    value: object = ...
    @property
    def __name__(self) -> str: ...

@dataclasses.dataclass
class ExecutionRecord:
    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    def dump(self, f: IO[str]) -> None: ...
    @classmethod
    def load(cls, f: IO[bytes]) -> Self: ...

@dataclasses.dataclass
class ExecutionRecorder:
    LOCAL_MOD_PREFIX = ...
    code: CodeType
    closure: tuple[CellType]
    globals: dict[str, Any] = field(default_factory=dict)
    locals: dict[str, Any] = field(default_factory=dict)
    builtins: dict[str, Any] = field(default_factory=dict)
    code_options: dict[str, Any] = field(default_factory=dict)
    name_to_modrec: dict[str, ModuleRecord] = field(default_factory=dict)
    def add_local_var(self, name: str, var: Any) -> None: ...
    def add_global_var(self, name: str, var: Any) -> None: ...
    def add_local_mod(self, name: str, mod: ModuleType) -> None: ...
    def record_module_access(self, mod: ModuleType, name: str, val: Any) -> None: ...
    def get_record(self) -> ExecutionRecord: ...
    def _add_mod(self, mod: ModuleType) -> ModuleRecord: ...
    @classmethod
    def _resolve_modules(cls, vars: dict[str, Any]) -> dict[str, Any]: ...
