from typing import TypeVar

T = TypeVar('T')

def not_none(obj: T | None) -> T: ...
