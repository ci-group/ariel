import torch
from _typeshed import Incomplete
from torch._ops import OpOverload as OpOverload
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode, MetadataMismatchError as MetadataMismatchError, UnsupportedFakeTensorException as UnsupportedFakeTensorException, tree_flatten_only as tree_flatten_only
from torch.utils._python_dispatch import TorchDispatchMode as TorchDispatchMode
from typing import Any, Callable

aten: Incomplete

def outputs_alias_inputs(outputs, inputs): ...
def outputs_are_inputs(outputs, inputs): ...
def output_alias_each_other(outputs): ...
def _check_alias_info(context, real_out, real_in, fake_out, fake_in) -> None: ...
def is_sdpa_error(func, idx, e): ...
def try_convert_fake_to_real(ten_list: list[FakeTensor | Any]) -> list[FakeTensor | torch.Tensor | Any]:
    """
    Attempt to convert fake tensors to a corresponding real tensor with the correct underlying storage by looking up
    the FakeTensorMode meta to real storage mapping. On failure to find the storage mapping, the FakeTensor will
    remain in the list.

    Note: this is not currently optimized (makes copies of the meta converter internal dictionaries)
    """
def _check_fake_real_tensors(real_out: torch.Tensor, fake_out: FakeTensor, context: str = '', sizes: bool = True, strides: bool = False, storage_offset: bool = True, requires_grad: bool = True): ...

class CrossRefFakeMode(TorchDispatchMode):
    ignore_op_fn: Incomplete
    check_strides: Incomplete
    check_aliasing: Incomplete
    only_check_ops_with_meta: Incomplete
    def __init__(self, ignore_op_fn: Callable[[OpOverload], bool] | None = None, *, check_strides: bool = True, check_aliasing: bool = True, only_check_ops_with_meta: bool = True) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...
