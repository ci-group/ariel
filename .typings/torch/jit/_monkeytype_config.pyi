import monkeytype
from _typeshed import Incomplete
from collections.abc import Iterable
from monkeytype.db.base import CallTraceStore, CallTraceStoreLogger, CallTraceThunk as CallTraceThunk
from monkeytype.tracing import CallTrace as CallTrace, CodeFilter as CodeFilter
from types import CodeType

_IS_MONKEYTYPE_INSTALLED: bool

def is_torch_native_class(cls): ...
def get_type(type):
    """Convert the given type to a torchScript acceptable format."""
def get_optional_of_element_type(types):
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extracts the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
def get_qualified_name(func): ...

class JitTypeTraceStoreLogger(CallTraceStoreLogger):
    """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""
    def __init__(self, store: CallTraceStore) -> None: ...
    def log(self, trace: CallTrace) -> None: ...

class JitTypeTraceStore(CallTraceStore):
    trace_records: dict[str, list]
    def __init__(self) -> None: ...
    def add(self, traces: Iterable[CallTrace]): ...
    def filter(self, qualified_name: str, qualname_prefix: str | None = None, limit: int = 2000) -> list[CallTraceThunk]: ...
    def analyze(self, qualified_name: str) -> dict: ...
    def consolidate_types(self, qualified_name: str) -> dict: ...
    def get_args_types(self, qualified_name: str) -> dict: ...

class JitTypeTraceConfig(monkeytype.config.Config):
    s: Incomplete
    def __init__(self, s: JitTypeTraceStore) -> None: ...
    def trace_logger(self) -> JitTypeTraceStoreLogger:
        """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
    def trace_store(self) -> CallTraceStore: ...
    def code_filter(self) -> CodeFilter | None: ...

class JitTypeTraceStoreLogger:
    def __init__(self) -> None: ...

class JitTypeTraceStore:
    trace_records: Incomplete
    def __init__(self) -> None: ...

class JitTypeTraceConfig:
    def __init__(self) -> None: ...

def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
