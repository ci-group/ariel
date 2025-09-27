from typing import Callable, TypeVar

F = TypeVar('F')

def exposed_in(module: str) -> Callable[[F], F]: ...
