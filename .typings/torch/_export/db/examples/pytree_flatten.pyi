import torch
from _typeshed import Incomplete

class PytreeFlatten(torch.nn.Module):
    """
    Pytree from PyTorch can be captured by TorchDynamo.
    """
    def forward(self, x): ...

example_args: Incomplete
model: Incomplete
