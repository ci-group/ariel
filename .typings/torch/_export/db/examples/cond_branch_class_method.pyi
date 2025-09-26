import torch
from _typeshed import Incomplete

class MySubModule(torch.nn.Module):
    def foo(self, x): ...
    def forward(self, x): ...

class CondBranchClassMethod(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables


    This example demonstrates using class method in cond().

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    subm: Incomplete
    def __init__(self) -> None: ...
    def bar(self, x): ...
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
