import abc
import numbers
from _typeshed import Incomplete
from abc import ABCMeta
from torch.utils.data.datapipes._hook_iterator import _SnapshotState as _SnapshotState, hook_iterator as hook_iterator
from typing import TypeVar, _tp_cache

class GenericMeta(ABCMeta): ...
class Integer(numbers.Integral, metaclass=abc.ABCMeta): ...
class Boolean(numbers.Integral, metaclass=abc.ABCMeta): ...

TYPE2ABC: Incomplete

def issubtype(left, right, recursive: bool = True):
    """
    Check if the left-side type is a subtype of the right-side type.

    If any of type is a composite type like `Union` and `TypeVar` with
    bounds, it would be expanded into a list of types and check all
    of left-side types are subtypes of either one from right-side types.
    """
def _decompose_type(t, to_list: bool = True): ...
def _issubtype_with_constraints(variant, constraints, recursive: bool = True):
    """
    Check if the variant is a subtype of either one from constraints.

    For composite types like `Union` and `TypeVar` with bounds, they
    would be expanded for testing.
    """
def issubinstance(data, data_type): ...

class _DataPipeType:
    """Save type annotation in `param`."""
    param: Incomplete
    def __init__(self, param) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
    def issubtype(self, other): ...
    def issubtype_of_instance(self, other): ...
_T_co = TypeVar('_T_co', covariant=True)
_DEFAULT_TYPE: Incomplete

class _DataPipeMeta(GenericMeta):
    """
    Metaclass for `DataPipe`.

    Add `type` attribute and `__init_subclass__` based on the type, and validate the return hint of `__iter__`.

    Note that there is subclass `_IterDataPipeMeta` specifically for `IterDataPipe`.
    """
    type: _DataPipeType
    def __new__(cls, name, bases, namespace, **kwargs): ...
    def __init__(self, name, bases, namespace, **kwargs) -> None: ...
    @_tp_cache
    def _getitem_(self, params): ...
    def _eq_(self, other): ...
    def _hash_(self): ...

class _IterDataPipeMeta(_DataPipeMeta):
    """
    Metaclass for `IterDataPipe` and inherits from `_DataPipeMeta`.

    Add various functions for behaviors specific to `IterDataPipe`.
    """
    def __new__(cls, name, bases, namespace, **kwargs): ...

def _dp_init_subclass(sub_cls, *args, **kwargs) -> None: ...
def reinforce_type(self, expected_type):
    """
    Reinforce the type for DataPipe instance.

    And the 'expected_type' is required to be a subtype of the original type
    hint to restrict the type requirement of DataPipe instance.
    """
