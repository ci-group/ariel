import torch
from _typeshed import Incomplete
from torch.export import Dim as Dim

x: Incomplete
dim1_x: Incomplete

class ScalarOutput(torch.nn.Module):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    def __init__(self) -> None: ...
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
dynamic_shapes: Incomplete
model: Incomplete
