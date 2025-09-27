import torch
from _typeshed import Incomplete

class TensorSetattr(torch.nn.Module):
    """
    setattr() call onto tensors is not supported.
    """
    def forward(self, x, attr): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
