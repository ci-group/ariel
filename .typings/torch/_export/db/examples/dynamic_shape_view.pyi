import torch
from _typeshed import Incomplete

class DynamicShapeView(torch.nn.Module):
    """
    Dynamic shapes should be propagated to view arguments instead of being
    baked into the exported graph.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
