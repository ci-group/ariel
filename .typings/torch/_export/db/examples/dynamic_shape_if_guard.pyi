import torch
from _typeshed import Incomplete

class DynamicShapeIfGuard(torch.nn.Module):
    """
    `if` statement with backed dynamic shape predicate will be specialized into
    one particular branch and generate a guard. However, export will fail if the
    the dimension is marked as dynamic shape from higher level API.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
