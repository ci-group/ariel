import torch
from _typeshed import Incomplete

class StaticForLoop(torch.nn.Module):
    """
    A for loop with constant number of iterations should be unrolled in the exported graph.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
