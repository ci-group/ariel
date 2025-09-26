from collections.abc import Iterable
from typing import Any, Callable

__all__ = ['trace_dependencies']

def trace_dependencies(callable: Callable[[Any], Any], inputs: Iterable[tuple[Any, ...]]) -> list[str]:
    """Trace the execution of a callable in order to determine which modules it uses.

    Args:
        callable: The callable to execute and trace.
        inputs: The input to use during tracing. The modules used by 'callable' when invoked by each set of inputs
            are union-ed to determine all modules used by the callable for the purpooses of packaging.

    Returns: A list of the names of all modules used during callable execution.
    """
