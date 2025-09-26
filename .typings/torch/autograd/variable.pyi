import torch
from _typeshed import Incomplete

__all__ = ['VariableMeta', 'Variable']

class VariableMeta(type):
    def __instancecheck__(cls, other): ...

class Variable(torch._C._LegacyVariableBase, metaclass=VariableMeta):
    _execution_engine: Incomplete
