from .fake_impl import FakeImplHolder
from .utils import RegistrationHandle
from _typeshed import Incomplete
from typing import Callable

__all__ = ['SimpleLibraryRegistry', 'SimpleOperatorEntry', 'singleton']

class SimpleLibraryRegistry:
    '''Registry for the "simple" torch.library APIs

    The "simple" torch.library APIs are a higher-level API on top of the
    raw PyTorch DispatchKey registration APIs that includes:
    - fake impl

    Registrations for these APIs do not go into the PyTorch dispatcher\'s
    table because they may not directly involve a DispatchKey. For example,
    the fake impl is a Python function that gets invoked by FakeTensor.
    Instead, we manage them here.

    SimpleLibraryRegistry is a mapping from a fully qualified operator name
    (including the overload) to SimpleOperatorEntry.
    '''
    _data: Incomplete
    def __init__(self) -> None: ...
    def find(self, qualname: str) -> SimpleOperatorEntry: ...

singleton: SimpleLibraryRegistry

class SimpleOperatorEntry:
    """This is 1:1 to an operator overload.

    The fields of SimpleOperatorEntry are Holders where kernels can be
    registered to.
    """
    qualname: str
    fake_impl: FakeImplHolder
    torch_dispatch_rules: GenericTorchDispatchRuleHolder
    def __init__(self, qualname: str) -> None: ...
    @property
    def abstract_impl(self): ...

class GenericTorchDispatchRuleHolder:
    _data: Incomplete
    qualname: Incomplete
    def __init__(self, qualname) -> None: ...
    def register(self, torch_dispatch_class: type, func: Callable) -> RegistrationHandle: ...
    def find(self, torch_dispatch_class): ...
