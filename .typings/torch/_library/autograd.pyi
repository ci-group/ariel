import dataclasses
from . import utils as utils
from torch import Tensor as Tensor, _C as _C, _ops as _ops, autograd as autograd
from torch.utils import _pytree as _pytree
from typing import Any, Callable, Protocol

class InfoProtocol(Protocol):
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

@dataclasses.dataclass
class Info:
    _backward_fn: Callable | None
    _setup_context_fn: Callable | None

def make_autograd_impl(op: _ops.OpOverload, info: InfoProtocol) -> Callable: ...
def supports_tensorlist(cls) -> Any:
    """Allows a given autograd.Function class to support List[Tensor] inputs/outputs.

    Regular autograd.Function has a constraint that it only directly supports autograd for
    Tensors. Applying @supports_tensorlist enables an autograd.Function to support
    autograd for List[Tensor] inputs and outputs.
    """
def not_list_of_tensor(tree): ...
def not_list_of_optional_tensor(tree): ...
flatten = _pytree.tree_flatten
unflatten = _pytree.tree_unflatten
spec_t = _pytree.TreeSpec
