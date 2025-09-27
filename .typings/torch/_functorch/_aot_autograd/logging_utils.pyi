from collections.abc import Generator
from contextlib import contextmanager

graph_being_compiled: list[str]
nth_graph: int
model_name: str

def set_model_name(name) -> None: ...
def get_aot_compilation_context() -> tuple[list[str], str, int]: ...
def get_aot_graph_name() -> str:
    """
    Returns the name of the graph being compiled.
    """
get_graph_being_compiled = get_aot_graph_name

@contextmanager
def track_graph_compiling(aot_config, graph_name) -> Generator[None]: ...

callback_set: bool

def setup_stacktrace_preservation_hooks(roots: list): ...
def describe_input(i, aot_config): ...
def format_guard_bug_msg(aot_config, expected): ...
