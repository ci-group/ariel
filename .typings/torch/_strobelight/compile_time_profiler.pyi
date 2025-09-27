from _typeshed import Incomplete
from torch._strobelight.cli_function_profiler import StrobelightCLIFunctionProfiler as StrobelightCLIFunctionProfiler
from typing import Any

logger: Incomplete
console_handler: Incomplete
formatter: Incomplete

def get_fburl(url: str) -> str: ...
def get_strobelight_url(identifier: str) -> str: ...

class StrobelightCompileTimeProfiler:
    success_profile_count: int
    failed_profile_count: int
    ignored_profile_runs: int
    inside_profile_compile_time: bool
    enabled: bool
    frame_id_filter: str | None
    identifier: str | None
    current_phase: str | None
    profiler: Any | None
    max_stack_length: int
    max_profile_time: int
    sample_each: int
    @classmethod
    def get_frame(cls) -> str: ...
    @classmethod
    def enable(cls, profiler_class: Any = ...) -> None: ...
    @classmethod
    def _cls_init(cls) -> None: ...
    @classmethod
    def _log_stats(cls) -> None: ...
    @classmethod
    def profile_compile_time(cls, func: Any, phase_name: str, *args: Any, **kwargs: Any) -> Any: ...
