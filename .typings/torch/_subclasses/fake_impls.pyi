import functools
import torch
from _typeshed import Incomplete

__all__ = ['op_implementations_checks', 'get_fast_op_impls', 'stride_incorrect_op', 'has_meta']

pytree = torch.utils._pytree
op_implementations_checks: Incomplete

def stride_incorrect_op(op): ...
def has_meta(func): ...
@functools.cache
def get_fast_op_impls(): ...
