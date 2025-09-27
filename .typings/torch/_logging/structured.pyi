import traceback
from collections.abc import Sequence
from typing import Any

INTERN_TABLE: dict[str, int]
DUMPED_FILES: set[str]

def intern_string(s: str | None) -> int: ...
def dump_file(filename: str) -> None: ...
def from_traceback(tb: Sequence[traceback.FrameSummary]) -> list[dict[str, Any]]: ...
def get_user_stack(num_frames: int) -> list[dict[str, Any]]: ...
def get_framework_stack(num_frames: int = 25, cpp: bool = False) -> list[dict[str, Any]]:
    """
    Returns the traceback for the user stack and the framework stack
    """
