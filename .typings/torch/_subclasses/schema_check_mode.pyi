import torch
from _typeshed import Incomplete
from torch.fx.operator_schemas import normalize_function as normalize_function
from torch.utils._python_dispatch import TorchDispatchMode as TorchDispatchMode
from torch.utils._pytree import tree_map as tree_map
from typing import NamedTuple

class Mutation(NamedTuple):
    op_name: Incomplete
    arg_name: Incomplete

class Aliasing(NamedTuple):
    op_name: Incomplete
    arg_name: Incomplete
    output_number: Incomplete

SchemaArgument: Incomplete
SchemaArgType: Incomplete
SchemaInfo = torch._C._SchemaInfo

def is_iterable_of_tensors(iterable): ...
def clone_inputs(args): ...

class SchemaCheckMode(TorchDispatchMode):
    ops: Incomplete
    mutated: Incomplete
    aliasing: Incomplete
    def __init__(self) -> None: ...
    def reset_cache(self) -> None: ...
    def display_ops(self) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...
