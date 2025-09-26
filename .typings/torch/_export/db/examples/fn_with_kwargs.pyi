import torch
from _typeshed import Incomplete

class FnWithKwargs(torch.nn.Module):
    """
    Keyword arguments are not supported at the moment.
    """
    def forward(self, pos0, tuple0, *myargs, mykw0, **mykwargs): ...

example_args: Incomplete
example_kwargs: Incomplete
tags: Incomplete
model: Incomplete
