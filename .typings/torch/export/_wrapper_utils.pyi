import torch
from _typeshed import Incomplete

class _WrapperModule(torch.nn.Module):
    f: Incomplete
    def __init__(self, f) -> None: ...
    def forward(self, *args, **kwargs): ...
