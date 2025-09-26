from _typeshed import Incomplete
from torch.utils import cpp_extension as cpp_extension
from torch.utils.benchmark.utils._stubs import CallgrindModuleType as CallgrindModuleType, TimeitModuleType as TimeitModuleType
from torch.utils.benchmark.utils.common import _make_temp_dir as _make_temp_dir
from typing import Any

LOCK: Incomplete
SOURCE_ROOT: Incomplete
_BUILD_ROOT: str | None

def _get_build_root() -> str: ...

CXX_FLAGS: list[str] | None
EXTRA_INCLUDE_PATHS: list[str]
CONDA_PREFIX: Incomplete
COMPAT_CALLGRIND_BINDINGS: CallgrindModuleType | None

def get_compat_bindings() -> CallgrindModuleType: ...
def _compile_template(*, stmt: str, setup: str, global_setup: str, src: str, is_standalone: bool) -> Any: ...
def compile_timeit_template(*, stmt: str, setup: str, global_setup: str) -> TimeitModuleType: ...
def compile_callgrind_template(*, stmt: str, setup: str, global_setup: str) -> str: ...
