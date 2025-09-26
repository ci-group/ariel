import torch
from _typeshed import Incomplete

def test_decorator(func): ...

class Decorator(torch.nn.Module):
    """
    Decorators calls are inlined into the exported function during tracing.
    """
    @test_decorator
    def forward(self, x, y): ...

example_args: Incomplete
model: Incomplete
