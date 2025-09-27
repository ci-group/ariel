import torch
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch.types import Number as Number
from typing import Callable, TypeVar
from typing_extensions import ParamSpec

aten: Incomplete
decomposition_table: dict[str, torch.jit.ScriptFunction]
function_name_set: set[str]
_T = TypeVar('_T')
_P = ParamSpec('_P')

def check_decomposition_has_type_annotations(f) -> None: ...
def signatures_match(decomposition_sig, torch_op_sig): ...
def register_decomposition(aten_op: torch._ops.OpOverload, registry: dict[str, torch.jit.ScriptFunction] | None = None) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def var_decomposition(input: Tensor, dim: list[int] | None = None, correction: Number | None = None, keepdim: bool = False) -> Tensor: ...
def var(input: Tensor, unbiased: bool = True) -> Tensor: ...
