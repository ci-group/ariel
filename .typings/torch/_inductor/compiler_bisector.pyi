import dataclasses
import functools
from dataclasses import dataclass, field
from torch._inductor.runtime.cache_dir_utils import cache_dir as cache_dir
from typing import Callable

SUBDIR_NAME: str

@dataclass
class Subsystem:
    name: str

@dataclass
class BisectSubsystem(Subsystem): ...
@dataclass
class BinarySubsystem(Subsystem): ...

@dataclass
class ConfigChange(BinarySubsystem):
    name: str = field(init=False)
    config_name: str
    config_field: str
    config_value: object
    def __post_init__(self) -> None: ...

BACKENDS: dict[str, list[Subsystem]]
subsystem_call_counter: dict[str, int]
call_counter_debug_info: dict[int, str]

def reset_counters() -> None: ...
@functools.cache
def get_env_val(env_str: str) -> str | None: ...

@dataclasses.dataclass
class BisectionResult:
    """
    backend: torch.compile backend responsible for failure
    subsystem: optional, registered component identified for failure
    bisect_number: optional, number of times the subsystem needed to be applied to trigger failure
    debug_info: associated info of the triggering bisect application of subsystem
    """
    backend: str
    subsystem: str | None = ...
    bisect_number: int | None = ...
    debug_info: str | None = ...

class CompilerBisector:
    """
    This class iteratively runs torch.compile backends (eager, aot_eager, inductor) to find the
    first backend that can repro an issue.

    Once it discovers the offending backend it will iteratively disable subsystems within the backend.
    For subsystems which are applied repeatedly, such as the number of post grad passes or number
    of lowering of nodes to inductor ir, it will bisect to find the offending application.

    The idiomatic way to run it is with `do_bisect`. You can also use it by setting the env flags
    `TORCH_BISECT_BACKEND`, `TORCH_BISECT_SUBSYSTEM` and `TORCH_BISECT_MAX`.

    It also supports a CLI interface, although this is less well tested.

    You must run python compiler_bisector.py [start | good | bad | end]
    """
    bisection_enabled: bool
    in_process_cache: str | None
    @classmethod
    def get_dir(cls) -> str: ...
    @classmethod
    def write_lines_to_file(cls, file_path: str, lines: list[str]) -> None: ...
    @classmethod
    def read_lines_from_file(cls, file_path: str) -> list[str]: ...
    @classmethod
    def update_run_state(cls, backend_name: str, subsystem: Subsystem, run_state: str) -> None: ...
    @classmethod
    def set_config_values(cls, backend: str, subsystem: str, config_data: dict[str, object]) -> None: ...
    @classmethod
    def update_bisect_status(cls, backend_name: str, subsystem_name: str) -> None: ...
    @classmethod
    def update_bisect_range(cls, backend_name: str, subsystem_name: str, low: int, high: int) -> None: ...
    @classmethod
    def get_backend(cls) -> str | None:
        """
        Returns the active backend, if any
        """
    @classmethod
    def get_subsystem(cls) -> str | None:
        """
        Returns the active subsystem, if any
        """
    @classmethod
    def get_subsystem_object(cls, backend_name: str, subsystem_name: str) -> Subsystem: ...
    @classmethod
    def get_run_state(cls, backend_name: str, subsystem_name: str) -> str | None:
        """
        Returns the current stage of bisecting, if Any
        """
    @classmethod
    def get_bisect_range(cls, backend_name: str, subsystem_name: str) -> tuple[int, int]: ...
    @classmethod
    def update_config_change(cls, backend: str, subsystem: ConfigChange) -> None: ...
    @classmethod
    def get_config_change(cls, config_name: str) -> dict[str, object] | None: ...
    @classmethod
    def delete_bisect_status(cls) -> None: ...
    @classmethod
    def get_system_counter(cls, name: str, increment: bool = True) -> int: ...
    @classmethod
    def disable_subsystem(cls, backend: str, subsystem: str, debug_info: Callable[[], str] | None = None) -> bool: ...
    @classmethod
    def advance_subsystem(cls, curr_backend: str, curr_subsystem: Subsystem) -> Subsystem | None:
        """
        Tries to move to the next subsystem within the current system.
        """
    @classmethod
    def advance_backend(cls, curr_backend: str) -> str | None:
        """
        Tries Move to the next backend.
        """
    @classmethod
    def process_subsystem(cls, curr_backend: str, curr_subsystem: Subsystem, fn: Callable[[], bool], cli_interface: bool = True) -> bool:
        """
        Process the current subsystem. Returns True if the issue is found, False otherwise.
        """
    @classmethod
    def initialize_system(cls) -> None: ...
    @classmethod
    def do_bisect(cls, fn: Callable[[], bool], cli_interface: bool = False) -> BisectionResult | None:
        """
        Run fn repeatedly attempting to bisect torch.compile. fn should return True on success and False on failure.
        """

def command_line_usage() -> None: ...
def get_is_bisection_enabled() -> bool: ...
