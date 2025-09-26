import torch
from _typeshed import Incomplete

class CondBranchNestedFunction(torch.nn.Module):
    """
    The branch functions (`true_fn` and `false_fn`) passed to cond() must follow these rules:
      - both branches must take the same args, which must also match the branch args passed to cond.
      - both branches must return a single tensor
      - returned tensor must have the same tensor metadata, e.g. shape and dtype
      - branch function can be free function, nested function, lambda, class methods
      - branch function can not have closure variables
      - no inplace mutations on inputs or global variables

    This example demonstrates using nested function in cond().

    NOTE: If the `pred` is test on a dim with batch size < 2, it will be specialized.
    """
    def forward(self, x): ...

example_args: Incomplete
tags: Incomplete
model: Incomplete
