import torch
from _typeshed import Incomplete

class StaticIf(torch.nn.Module):
    """
    `if` statement with static predicate value should be traced through with the
    taken branch.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
