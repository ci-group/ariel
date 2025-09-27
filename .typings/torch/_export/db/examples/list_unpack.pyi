import torch
from _typeshed import Incomplete

class ListUnpack(torch.nn.Module):
    """
    Lists are treated as static construct, therefore unpacking should be
    erased after tracing.
    """
    def forward(self, args: list[torch.Tensor]):
        """
        Lists are treated as static construct, therefore unpacking should be
        erased after tracing.
        """

example_args: Incomplete
tags: Incomplete
model: Incomplete
