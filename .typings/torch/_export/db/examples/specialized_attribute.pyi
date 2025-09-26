import torch
from _typeshed import Incomplete
from enum import Enum

class Animal(Enum):
    COW = 'moo'

class SpecializedAttribute(torch.nn.Module):
    """
    Model attributes are specialized.
    """
    a: str
    b: int
    def __init__(self) -> None: ...
    def forward(self, x): ...

example_args: Incomplete
model: Incomplete
