import torch
from _typeshed import Incomplete

class NestedFunction(torch.nn.Module):
    """
    Nested functions are traced through. Side effects on global captures
    are not supported though.
    """
    def forward(self, a, b): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
