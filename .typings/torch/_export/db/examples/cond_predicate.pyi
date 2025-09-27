import torch
from _typeshed import Incomplete

class CondPredicate(torch.nn.Module):
    """
    The conditional statement (aka predicate) passed to cond() must be one of the following:
      - torch.Tensor with a single element
      - boolean expression

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
