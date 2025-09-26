import torch
from _typeshed import Incomplete

class DynamicShapeSlicing(torch.nn.Module):
    """
    Slices with dynamic shape arguments should be captured into the graph
    rather than being baked in.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
