import torch
from _typeshed import Incomplete

class DynamicShapeConstructor(torch.nn.Module):
    """
    Tensor constructors should be captured with dynamic shape inputs rather
    than being baked in with static shape.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
