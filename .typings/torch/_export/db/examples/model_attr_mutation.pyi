import torch
from _typeshed import Incomplete
from torch._export.db.case import SupportLevel as SupportLevel

class ModelAttrMutation(torch.nn.Module):
    """
    Attribute mutation is not supported.
    """
    attr_list: Incomplete
    def __init__(self) -> None: ...
    def recreate_list(self): ...
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
support_level: Incomplete
model: Incomplete
