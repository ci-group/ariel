from _typeshed import DataclassInstance
from typing import TypeVar

__all__ = ['dataclass_slots']

_T = TypeVar('_T', bound='DataclassInstance')

def dataclass_slots(cls) -> type[DataclassInstance]: ...
