import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor as Tensor, _prims as _prims

log: Incomplete

def make_prim(schema: str, impl_aten, return_type=..., doc: str = '', tags: Sequence[torch.Tag] | None = None): ...
def eager_force_stride(input_tensor: Tensor, stride) -> Tensor: ...
def eager_prepare_softmax(x: Tensor, dim: int) -> tuple[Tensor, Tensor]: ...

seed: Incomplete
seeds: Incomplete
lookup_seed: Incomplete
random: Incomplete
randint: Incomplete
force_stride_order: Incomplete
_unsafe_index_put_: Incomplete
fma: Incomplete
prepare_softmax_online: Incomplete

def _flattened_index_to_nd(indices, width): ...
def _flatten_index(indices, width): ...
def _low_memory_max_pool_with_offsets_aten(self, kernel_size, stride, padding, dilation, ceil_mode): ...
def _low_memory_max_pool_offsets_to_indices_aten(offsets, kernel_size, input_size, stride, padding, dilation): ...

_low_memory_max_pool_with_offsets: Incomplete
_low_memory_max_pool_offsets_to_indices: Incomplete
