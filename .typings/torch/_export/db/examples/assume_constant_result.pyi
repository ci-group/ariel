import torch
import torch._dynamo as torchdynamo
from _typeshed import Incomplete

class AssumeConstantResult(torch.nn.Module):
    """
    Applying `assume_constant_result` decorator to burn make non-tracable code as constant.
    """
    @torchdynamo.assume_constant_result
    def get_item(self, y): ...
    def forward(self, x, y): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
