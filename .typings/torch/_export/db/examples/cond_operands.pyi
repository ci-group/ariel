import torch
from _typeshed import Incomplete
from torch.export import Dim as Dim

x: Incomplete
y: Incomplete
dim0_x: Incomplete

class CondOperands(torch.nn.Module):
    """
    The operands passed to cond() must be:
    - a list of tensors
    - match arguments of `true_fn` and `false_fn`

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    def forward(self, x, y): ...

example_args: Incomplete
tags: Incomplete
extra_inputs: Incomplete
dynamic_shapes: Incomplete
model: Incomplete
