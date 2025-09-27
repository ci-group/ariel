import torch
from _typeshed import Incomplete
from torch._export.db.case import SupportLevel as SupportLevel

class TorchSymMin(torch.nn.Module):
    """
    torch.sym_min operator is not supported in export.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
support_level: Incomplete
model: Incomplete
