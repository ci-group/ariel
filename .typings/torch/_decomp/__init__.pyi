import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._ops import OpOverloadPacket
from torch.export.decomp_utils import CustomDecompTable
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

__all__ = ['decomposition_table', 'pre_autograd_decomposition_table', 'meta_table', 'register_decomposition', 'get_decompositions', 'core_aten_decompositions', '_should_decompose_because_unsafe_op']

_T = TypeVar('_T')
_P = ParamSpec('_P')
decomposition_table: Incomplete
pre_autograd_decomposition_table: Incomplete
meta_table: Incomplete

def _should_decompose_because_unsafe_op(op: torch._ops.OperatorBase) -> bool:
    """
    Returns True if the op must always decompose in export/compile tracing system

    In export, we always decompose certain CIA ops that are tagged with
    maybe_aliasing_or_mutating because we statically need to know if the op is
    mutating or not. But these CIA ops could have different behaviour in runtime.

    native_batch_norm is a prim op which has a wrong schema and it needs to be replaced
    with correct schema. But until then, we will force decompose it via this tag.
    """
def register_decomposition(aten_op, registry=None, *, type: str = 'post_autograd', unsafe: bool = False) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    """
    A decorator to register a function as a decomposition to the Python
    decomposition table.  Use it like this::

        @register_decomposition(torch.ops.aten.clamp_min)
        def clamp_min(x):
            return torch.clamp(self, min=min)

    If you are writing a new decomposition, consider contributing it
    directly to PyTorch in torch._decomp.decompositions.

    This API is experimental; we are almost certainly going to extend
    the API when we make decompositions eligible for use in transforms (e.g.,
    autograd) and not just backend tracing, where we then need to know if a
    decomposition can be used to simulate a transform.

    By default, we also will register it to the Meta key of dispatcher,
    and replace the c++ Meta implementation if there is already one.

    unsafe kwarg is for reuse of this function for registering non-function
    things
    """
def get_decompositions(aten_ops: Sequence[torch._ops.OperatorBase | OpOverloadPacket], type: str = 'post_autograd') -> dict[torch._ops.OperatorBase, Callable]:
    """
    Retrieve a dictionary of decompositions corresponding to the list of
    operator overloads and overload packets passed as input.  Overload
    packets will include all decomposed overloads in the packet.  If there is
    no decomposition for a requested operator, it is silently ignored.

    This API is experimental; we are almost certainly going to give an alternate,
    more recommended formulation, where a user provides the set of operators
    they know how to implement, and we provide decompositions for everything
    not in this set.
    """
def core_aten_decompositions() -> CustomDecompTable: ...
