import torch
from _typeshed import Incomplete
from torch._export.db.case import SupportLevel as SupportLevel

class OptionalInput(torch.nn.Module):
    """
    Tracing through optional input is not supported yet
    """
    def forward(self, x, y=...): ...

example_args: Incomplete
tags: Incomplete
support_level: Incomplete
model: Incomplete
