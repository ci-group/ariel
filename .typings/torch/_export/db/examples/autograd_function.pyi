import torch
from _typeshed import Incomplete

class MyAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): ...
    @staticmethod
    def backward(ctx, grad_output): ...

class AutogradFunction(torch.nn.Module):
    """
    TorchDynamo does not keep track of backward() on autograd functions. We recommend to
    use `allow_in_graph` to mitigate this problem.
    """
    def forward(self, x): ...

example_args: Incomplete
model: Incomplete
