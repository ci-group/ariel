from _typeshed import Incomplete
from dataclasses import dataclass
from torch._ops import HigherOrderOperator as HigherOrderOperator
from typing import Callable

def is_graphable(val) -> bool:
    """Definition: a graphable type is a type that that is an acceptable input/output type to a FX node."""
def is_graphable_type(typ) -> bool:
    """Return whether the given type is graphable"""
def to_graphable(stuff):
    """Flattens stuff into a flat list of graphable types."""
def from_graphable(flat_args, spec):
    """The inverse of to_graphable."""
def func_to_graphable(func):
    """
    Pack and flatten a function type into graphable types.
    This is useful for legalizing the function argument of `flat_apply`.
    """

@dataclass(frozen=True)
class _ConstantFunction:
    func: Callable
    def __call__(self, *args, **kwargs): ...

_op_types: Incomplete

class FlatApply(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, func, in_spec, *flat_args, **_unused):
        """
        Functions that take in non-graphable types cannot directly be put into FX graph.

        Given func(*args, **kwargs), if all of the non-graphable types are pytrees,
        then we're able to store a call to flat_apply(func, in_spec, *flat_args) in the FX graph.

        The semantics of flat_apply(func, in_spec, *flat_args) are roughly equivalent to:

        >>> def flat_apply_impl(func, in_spec, *flat_args):
        >>>     args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
        >>>     output = func(*args, **kwargs)
        >>>     return output

        flat_apply supports the following two cases:
        - an input type is a container type (e.g. of tensors) registered as a pytree.
        We'll tree_flatten the input type and store the spec.
        - an input type is a constant type (i.e. torch.compile will specialize on it)
        registered with pytree.register_constant. The constant type goes directly
        into the spec.

        """

def impl(func, in_spec, *flat_args): ...

flat_apply: Incomplete
