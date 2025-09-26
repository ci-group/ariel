from .triton_compat import _log2 as _log2, builtins_use_semantic_kwarg as builtins_use_semantic_kwarg, libdevice as libdevice, math as math, tl as tl, triton as triton
from typing import Any, Callable, TypeVar

_T = TypeVar('_T')
_LOG_2_E: tl.constexpr

def set_driver_to_cpu() -> None: ...
def set_driver_to_gpu() -> None: ...
def get_backend_options(): ...
@triton.jit
def promote_to_tensor(x): ...
@triton.jit
def div_floor_integer(a, b): ...
@triton.jit
def remainder_integer(a, b): ...
@triton.jit
def is_floating(x): ...
@triton.jit
def _prod_accumulate(a, b): ...
@triton.jit
def prod(input, axis): ...
@triton.jit
def minimum(a, b): ...
@triton.jit
def maximum(a, b): ...
@triton.jit
def min2(a, dim): ...
@triton.jit
def max2(a, dim): ...
@triton.jit
def minimum_with_index(a_value, a_index, b_value, b_index): ...
@triton.jit
def maximum_with_index(a_value, a_index, b_value, b_index): ...
@triton.jit
def min_with_index(value, index, dim): ...
@triton.jit
def max_with_index(value, index, dim): ...
@triton.jit
def exp(x, use_fast_math: tl.constexpr): ...
@triton.jit
def online_softmax_reduce(lhs_max, lhs_sum, dim, use_fast_math: tl.constexpr): ...
@triton.jit
def online_softmax_combine(lhs_max, lhs_sum, rhs_max, use_fast_math: tl.constexpr):
    """
    When we do combine, we assume lhs is the accumulator and rhs is the next
    block of data.
    Then rhs_sum is always 1. With that assumption, we can save some registers
    and computation.
    """
@triton.jit
def welford_reduce(value, mean, m2, weight, first_iteration): ...
@triton.jit
def welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2): ...
@triton.jit
def welford(mean, m2, weight, dim): ...
@triton.jit
def device_assert_then(cond, msg, r): ...
@triton.jit
def randint64(seed, offset, low, high): ...
@triton.jit
def _any_combine(a, b): ...
@triton.jit
def any(a, dim): ...
@triton.jit
def bucketize_binary_search(values: tl.tensor, boundaries_ptr: tl.tensor, BOUNDARIES_SIZE: int, BOUNDARIES_UNDERLYING_NUMEL: int, BOUNDARIES_STRIDE: int, boundary_indices: tl.tensor, indexing_dtype: tl.dtype, right: bool, sorter_ptr: tl.tensor, SORTER_STRIDE: int, sorter_indices: tl.tensor):
    '''
    See [Note: Inductor bucketize op]

    Inputs:
    -------
    values: the values to bucketize.
    boundaries_ptr: a pointer to the beginning of the boundaries tensor, in 1-D.
    BOUNDARIES_SIZE: the length of the last dimension of the boundaries tensor (i.e. one
    individual set of boundaries).
    BOUNDARIES_UNDERLYING_NUMEL: the length of the boundaries tensor, in 1-D, ignoring
    any striding.
    BOUNDARIES_STRIDE: the stride of the last dimension of the boundaries tensor
    boundary_indices: a tensor of the same size as "values"; each element is an index
    into a 1-D, un-strided boundaries tensor, pointing to the first element in the set
    of boundaries used for that value.
    indexing_dtype: the dtype used for indexing into the boundaries tensor, and the
    return dtype.
    right: if true, use boundary intervals closed on the left; otherwise use intervals
    closed on the right.
    sorter_ptr: an optional pointer to a sorter tensor of the same shape as boundaries,
    but potentially different striding.  If present, this allows us to treat boundaries
    as sorted even if the elements of boundaries are unsorted.
    SORTER_STRIDE: must be present if sorter_ptr is non-None; the stride of the last
    dimension of the sorter tensor.
    sorter_indices: must be present if sorter_ptr is non-None; see "boundary_indices".
    BLOCK_SHAPE: the shape of the data block being processed.
    '''
@triton.jit
def pack_value_flag(value, flag, DTYPE_VALUE_AS_UINT: tl.constexpr, DTYPE_PACK: tl.constexpr): ...
@triton.jit
def unpack_value(pack, DTYPE_VALUE, DTYPE_VALUE_AS_UINT): ...
@triton.jit
def unpack_flag(pack, DTYPE_FLAG): ...
@triton.jit
def exclusive_scan_decoupled_lookback(scratch_base, block_value, index, combine_fn, DTYPE_VALUE_AS_UINT: tl.constexpr, DTYPE_PACK: tl.constexpr):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    DTYPE_VALUE_AS_UINT: A tl.uint{n} type equal in size to ``block_value``
    DTYPE_PACK: Unsigned type twice the width of block_value

    NOTE: This function is limited to values which are 32-bits or less because
    we need to pack (value, flag) into a single unsigned int.
    """
@triton.jit
def exclusive_scan_decoupled_lookback_64(scratch_base, block_value, index, combine_fn):
    """Compute exclusive scan of a scalar value between blocks

    Ref: https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    scratch_base: Pointer to scratch space in global memory
    block_value: Scalar value for this block, must be 64-bits wide
    index: Scalar index of this block relative to the current scan
    combine_fn: Function ``(value, value) -> value`` which is scanned over
    init: Scalar value equal to the identity of combine_fn
    """
@triton.jit
def frexp(x): ...
@triton.jit
def _compare_and_swap_with_index(x, idxs, rnumel, flip, i: tl.constexpr, n_dims: tl.constexpr, stable: tl.constexpr, descending: tl.constexpr): ...
@triton.jit
def _bitonic_merge_with_index(x, idxs, rnumel, stage: tl.constexpr, alternating: tl.constexpr, n_dims: tl.constexpr, stable: tl.constexpr, descending: tl.constexpr): ...
@triton.jit
def sort_with_index(x, idxs, rnumel, dim: tl.constexpr = None, stable: tl.constexpr = ..., descending: tl.constexpr = ...): ...
@triton.jit
def select_one(x, mask, dim, keep_dims: bool = False): ...
@triton.jit
def x_grid_barrier(sem) -> None:
    """
    Wait for all other thread blocks in grid sharing same y/z program_id
    to reach this barrier before returning.

    Args:
        sem: an uint32 semaphores, zero or 0x80000000 initialized.  Must be unique to each y/z program ID.
    """
def triton_builtin(f: Callable[..., _T]) -> Callable[..., _T]:
    """
    Decorator to mark a function as a Triton built-in function.  These functions
    are evaluated at compile time.

    Args:
        f (function): The function to be marked as a Triton built-in.

    Returns:
        function: The same function, marked as a Triton built-in.
    """
@triton_builtin
def constexpr_next_power_of_2(n: tl.constexpr, *, _builder: object = None) -> tl.constexpr:
    """
    A version triton.next_power_of_two that can be used within a kernel on constants.
    """
@triton_builtin
def if_mask(mask: Any, val, *, _builder: object = None) -> tl.constexpr:
    """
    Work around triton compile error: `ValueError: `other` cannot be provided without `mask``
    A compile-time to check to return either `val` or `None` depending on the value of mask.
    """
