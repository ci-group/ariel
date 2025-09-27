import torch
from _typeshed import Incomplete

class ConstrainAsSizeExample(torch.nn.Module):
    """
    If the value is not known at tracing time, you can provide hint so that we
    can trace further. Please look at torch._check and torch._check_is_size APIs.
    torch._check_is_size is used for values that NEED to be used for constructing
    tensor.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
