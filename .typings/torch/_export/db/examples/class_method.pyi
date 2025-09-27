import torch
from _typeshed import Incomplete

class ClassMethod(torch.nn.Module):
    """
    Class methods are inlined during tracing.
    """
    @classmethod
    def method(cls, x): ...
    linear: Incomplete
    def __init__(self) -> None: ...
    def forward(self, x): ...

example_args: Incomplete
model: Incomplete
