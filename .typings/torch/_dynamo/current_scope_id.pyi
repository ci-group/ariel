import contextlib
from _typeshed import Incomplete
from collections.abc import Generator

_current_scope_id: Incomplete

def current_scope_id() -> int: ...
@contextlib.contextmanager
def enter_new_scope() -> Generator[None, None, None]: ...
