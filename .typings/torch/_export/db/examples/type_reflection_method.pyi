import torch
from _typeshed import Incomplete

class A:
    @classmethod
    def func(cls, x): ...

class TypeReflectionMethod(torch.nn.Module):
    """
    type() calls on custom objects followed by attribute accesses are not allowed
    due to its overly dynamic nature.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
