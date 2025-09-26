import dataclasses
from _typeshed import Incomplete
from torch._export.serde import schema as schema
from torch._export.serde.union import _Union as _Union
from typing import Any

class SchemaUpdateError(Exception): ...

def _check(x, msg) -> None: ...

_CPP_TYPE_MAP: Incomplete
_THRIFT_TYPE_MAP: Incomplete

def _staged_schema(): ...
def _diff_schema(dst, src): ...
def _hash_content(s: str): ...

@dataclasses.dataclass
class _Commit:
    result: dict[str, Any]
    checksum_next: str
    yaml_path: str
    additions: dict[str, Any]
    subtractions: dict[str, Any]
    base: dict[str, Any]
    checksum_head: str | None
    cpp_header: str
    cpp_header_path: str
    thrift_checksum_head: str | None
    thrift_checksum_real: str | None
    thrift_checksum_next: str
    thrift_schema: str
    thrift_schema_path: str

def update_schema(): ...
def check(commit: _Commit, force_unsafe: bool = False): ...
