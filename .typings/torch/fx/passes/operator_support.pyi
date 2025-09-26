import abc
import torch
import torch.fx
import typing as t

__all__ = ['OperatorSupportBase', 'OperatorSupport', 'create_op_support', 'chain', 'OpSupports', 'any_chain']

TargetTypeName = str
SupportedArgumentDTypes = tuple[t.Sequence[t.Sequence[torch.dtype]], dict[str, t.Sequence[torch.dtype]]] | None
SupportDict = t.Mapping[TargetTypeName, SupportedArgumentDTypes]

class OperatorSupportBase(abc.ABC, metaclass=abc.ABCMeta):
    """Interface for determining if a fx.Node is supported by a backend"""
    @abc.abstractmethod
    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool: ...

class OperatorSupport(OperatorSupportBase):
    '''
    `_support_dict` maps node.target typename to supported inputs dtypes.

    node.target typename is retrieved using helper function `get_node_target()`

    If supported inputs dtypes is None, it means any dtype is supported, else
    we should see a tuple like (([dtypes], ...), {"name":[dtypes], ...}).

    The first tuple ([dtypes], ...) indicates what dtypes are supported for
    inputs in node.args and the second dict {"name": [dtypes], ...} indicates
    what dtypes are supported for inputs in node.kwargs.

    For inputs in args, if we don\'t want to check it, we can put None there,
    e.g. (None, [torch.float]) indicates that we don\'t care about the type of
    the first input in args. And for inputs in kwargs, if not listed, will not
    be checked.
    '''
    _support_dict: SupportDict
    def __init__(self, support_dict: SupportDict | None = None) -> None: ...
    def is_node_supported(self, submodules: t.Mapping[str, torch.nn.Module], node: torch.fx.Node) -> bool:
        """
        Args:
            `submodules`: mapping from module name to the module. This can be
                          retrieved by calling model.named_modules().

            `node`: a Fx node that we want to determine whether it's supported.

        Returns:
            `is_supported`: whether the arg `node` is supported.
        """
IsNodeSupported = t.Callable[[t.Mapping[str, torch.nn.Module], torch.fx.Node], bool]

def create_op_support(is_node_supported: IsNodeSupported) -> OperatorSupportBase:
    """Wraps a `IsNodeSupported` function into an `OperatorSupportBase` instance

    `IsNodeSupported` has the same call signature as
    `OperatorSupportBase.is_node_supported`
    """
def chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns False if
    any of it reports False.
    """
def any_chain(*op_support: OperatorSupportBase) -> OperatorSupportBase:
    """Combines a sequence of `OperatorSupportBase` instances to form a single `OperatorSupportBase`
    instance by evaluating each input `OperatorSupportBase` instance, and returns True if
    any of it reports True.
    """

class OpSupports:
    """A set of atomic `OperatorSupportBase` instances that can be combined together
    to form more complex operator support logic.
    """
    @classmethod
    def decline_if_input_dtype(cls, dtype: torch.dtype) -> OperatorSupportBase:
        """Report a node as non-supported, if any of its arguments is of dtype"""
    @classmethod
    def decline_if_node_in_names(cls, disallow_set: set[str]) -> OperatorSupportBase:
        """
        If a node has a name that is in the disallow set, reported it as non-supported.
        """
