from _typeshed import Incomplete
from collections.abc import Sequence
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

logger: Incomplete
console_handler: Incomplete
formatter: Incomplete
_P = ParamSpec('_P')
_R = TypeVar('_R')

class StrobelightCLIProfilerError(Exception):
    """
    Raised when an error happens during strobelight profiling
    """

def _pid_namespace_link(pid: int | None = None) -> str:
    """Returns the link to the process's namespace, example: pid:[4026531836]"""
def _pid_namespace(pid: int | None = None) -> int:
    """Returns the process's namespace id"""
def _command_to_string(command: Sequence[str]) -> str: ...

class StrobelightCLIFunctionProfiler:
    """
    Note: this is a meta only tool.

    StrobelightCLIFunctionProfiler can be used to profile a python function and
    generate a strobelight link with the results. It works on meta servers but
    does not requries an fbcode target.
    When stop_at_error is false(default), error during profiling does not prevent
    the work function from running.

    Check function_profiler_example.py for an example.
    """
    _lock: Incomplete
    stop_at_error: Incomplete
    max_profile_duration_sec: Incomplete
    sample_each: Incomplete
    run_user_name: Incomplete
    timeout_wait_for_running_sec: Incomplete
    timeout_wait_for_finished_sec: Incomplete
    current_run_id: int | None
    sample_tags: Incomplete
    def __init__(self, *, stop_at_error: bool = False, max_profile_duration_sec: int = ..., sample_each: float = 10000000.0, run_user_name: str = 'pytorch-strobelight-ondemand', timeout_wait_for_running_sec: int = 60, timeout_wait_for_finished_sec: int = 60, recorded_env_variables: list[str] | None = None, sample_tags: list[str] | None = None, stack_max_len: int = 127, async_stack_max_len: int = 127) -> None: ...
    def _run_async(self) -> None: ...
    def _wait_for_running(self, counter: int = 0) -> None: ...
    def _stop_run(self) -> None: ...
    def _get_results(self) -> None: ...
    def _stop_strobelight_no_throw(self, collect_results: bool) -> None: ...
    def _start_strobelight(self) -> bool: ...
    def profile(self, work_function: Callable[_P, _R], *args: _P.args, **kwargs: _P.kwargs) -> _R | None: ...

def strobelight(profiler: StrobelightCLIFunctionProfiler | None = None, **kwargs: Any) -> Callable[[Callable[_P, _R]], Callable[_P, _R | None]]: ...
