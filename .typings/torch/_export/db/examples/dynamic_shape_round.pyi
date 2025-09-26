import torch
from _typeshed import Incomplete
from torch._export.db.case import SupportLevel as SupportLevel
from torch.export import Dim as Dim

class DynamicShapeRound(torch.nn.Module):
    """
    Calling round on dynamic shapes is not supported.
    """
    def forward(self, x): ...

x: Incomplete
dim0_x: Incomplete
example_args: Incomplete
tags: Incomplete
support_level: Incomplete
dynamic_shapes: Incomplete
model: Incomplete
