import torch
import triton
import triton.language as tl
from ._triton_ops_meta import get_meta as get_meta
from _typeshed import Incomplete
from collections.abc import Generator
from torch._dynamo.utils import warn_once as warn_once
from torch.utils._triton import has_triton as has_triton

TORCH_SPARSE_BSR_SCATTER_MM_LRU_CACHE_SIZE: Incomplete

def check(cond, msg) -> None: ...
def check_bsr_layout(f_name, t) -> None: ...
def check_device(f_name, t, device) -> None: ...
def check_mm_compatible_shapes(f_name, lhs, rhs) -> None: ...
def check_dtype(f_name, t, dtype, *additional_dtypes) -> None: ...
def check_blocksize(f_name, blocksize): ...
def make_triton_contiguous(t):
    """Return input as a triton-contiguous tensor.

    A triton-contiguous tensor is defined as a tensor that has strides
    with minimal value smaller than or equal to 1.

    While triton kernels support triton-non-contiguous tensors (all
    strides being greater than 1) arguments, a considerable slow-down
    occurs because tensor data is copied element-wise rather than
    chunk-wise. Zero strides is assumed to not have this defect.
    """
def broadcast_batch_dims(f_name, *tensors): ...
def slicer(dim, slice_range, *tensors) -> Generator[Incomplete]: ...
def multidim_slicer(dims, slices, *tensors) -> Generator[Incomplete]: ...
def ptr_stride_extractor(*tensors) -> Generator[Incomplete, Incomplete]: ...
def grid_partitioner(full_grid, grid_blocks, tensor_dims_map) -> Generator[Incomplete]: ...
def launch_kernel(kernel, tensor_dims_map, full_grid, grid_blocks=None): ...
def prepare_inputs(bsr, *dense_tensors): ...
def broadcast_batch_dims_bsr(f_name, bsr, *tensors): ...
def tile_to_blocksize(t, blocksize): ...
def as1Dbatch(tensor):
    """Return tensor as 3D tensor by either prepending new dimensions to
    the tensor shape (when ``tensor.ndim < 3``), or by collapsing
    starting dimensions into the first dimension (when ``tensor.ndim >
    3``).
    """
def scatter_mm(blocks, others, indices_data, *, accumulators=None):
    '''Scattered matrix multiplication of tensors.

    A scattered matrix multiplication is defined as a series of matrix
    multiplications applied to input tensors according to the input
    and output mappings specified by indices data.

    The following indices data formats are supported for defining a
    scattered matrix multiplication operation (:attr:`indices_data[0]`
    holds the name of the indices data format as specified below):

    - ``"scatter_mm"`` - matrix multiplications scattered in batches
      of tensors.

      If :attr:`blocks` is a :math:`(* \times M \times K) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor,
      and :attr:`indices = indices_data[\'indices\']` is a :math:`(*
      \times 3)` tensor, then the operation is equivalent to the
      following code::

        c_offsets, pq = indices_data[1:]
        for r in range(len(c_offsets) - 1):
            for g in range(c_offsets[r], c_offsets[r + 1]):
                p, q = pq[g]
                accumulators[r] += blocks[p] @ others[q]

    - ``"bsr_strided_mm"`` - matrix multiplications scattered in
      batches of tensors and a tensor.

      If :attr:`blocks` is a :math:`(Ms \times Ks) tensor,
      :attr:`others` is a :math:`(* \times K \times N)` tensor,
      :attr:`accumulators` is a :math:`(* \times M \times N)` tensor, then
      the operation is equivalent to the following code::

        c_indices, r_offsets, p_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for i, r in enumerate(r_offsets):
                r0, r1 = divmod(r, N)
                acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                for g in range(c_indices[i], c_indices[i+1]):
                    p = p_offsets[g]
                    q0, q1 = divmod(q_offsets[g], N)
                    acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]

      where ``Ns = N // meta[\'SPLIT_N\']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

    - ``"bsr_strided_mm_compressed"`` - matrix multiplications
      scattered in batches of tensors and a tensor. A memory and
      processor efficient version of ``"bsr_strided_mm"`` format.  If
      :attr:`blocks` is a :math:`(Ms \times Ks) tensor, :attr:`others`
      is a :math:`(* \times K \times N)` tensor, :attr:`accumulators`
      is a :math:`(* \times M \times N)` tensor, then the operation is
      equivalent to the following code::

        c_indices, r_offsets, q_offsets, meta = indices_data[1:]
        for b in range(nbatches):
            for r in r_offsets:
                m = (r // N) // Ms
                n = (r % N) // Ns
                r0, r1 = divmod(r, N)
                c0, c1 = c_indices[m], c_indices[m + 1]
                acc = accumulators[b, r0:r0 + Ms, r1:r1 + Ns]
                for i, p in enumerate(range(c0, c1)):
                    q = q_offsets[n * c1 + (SPLIT_N - n) * c0 + i]
                    q0, q1 = divmod(q, N)
                    acc += blocks[p] @ others[b, q0:q0 + Ks, q1:q1 + Ns]

      where ``Ns = N // meta[\'SPLIT_N\']``, and ``M`` and ``K`` are
      integer multiples of ``Ms`` and ``Ks``, respectively.

      Notice that the order of ``r_offsets`` items can be arbitrary;
      this property enables defining swizzle operators via
      rearrangements of ``r_offsets`` items..

    Auxiliary functions are provided for pre-computing
    :attr:`indices_data`. For example,
    :func:`bsr_scatter_mm_indices_data` is used to define indices data
    for matrix multiplication of BSR and strided tensors.

    Parameters
    ----------
    blocks (Tensor): a 3-D tensor of first matrices to be multiplied

    others (Tensor): a tensor of second matrices to be multiplied. If
      ``indices_data[0]=="scatter_mm"``, the tensor is a 1-D batch
      tensor of second input matrices to be multiplied. Otherwise, the
      second input matrices are slices of the :attr:`others` tensor.
    indices_data (tuple): a format data that defines the inputs and
      outputs of scattered matrix multiplications.

    Keyword arguments
    -----------------

    accumulators (Tensor, optional): a tensor of matrix product
      accumulators. If ``indices_data[0]=="scatter_mm"``, the tensor
      is a 1-D batch tensor of output matrices. Otherwise, output
      matrices are slices of the :attr:`accumulators` tensor.
    '''
def scatter_mm_meta(M, K, N, Ms, Ks, GROUP_SIZE=None, TILE_M=None, TILE_N=None, SPLIT_N=None, num_warps=None, num_stages=None, **extra): ...
def bsr_dense_addmm_meta(M, K, N, Ms, Ks, beta, alpha, SPLIT_N=None, GROUP_SIZE_ROW=None, num_warps=None, num_stages=None, sparsity=None, dtype=None, out_dtype=None, _version: int = 0, **extra): ...

class TensorAsKey:
    """A light-weight wrapper of a tensor that enables storing tensors as
    keys with efficient memory reference based comparison as an
    approximation to data equality based keys.

    Motivation: the hash value of a torch tensor is tensor instance
    based that does not use data equality and makes the usage of
    tensors as keys less useful. For instance, the result of
    ``len({a.crow_indices(), a.crow_indices()})`` is `2`, although,
    the tensor results from `crow_indices` method call are equal, in
    fact, these share the same data storage.
    On the other hand, for efficient caching of tensors we want to
    avoid calling torch.equal that compares tensors item-wise.

    TensorAsKey offers a compromise in that it guarantees key equality
    of tensors that references data in the same storage in the same
    manner and without accessing underlying data. However, this
    approach does not always guarantee correctness. For instance, for
    a complex tensor ``x``, we have ``TensorAsKey(x) ==
    TensorAsKey(x.conj())`` while ``torch.equal(x, x.conj())`` would
    return False.
    """
    _obj_ref: Incomplete
    key: Incomplete
    _hash: Incomplete
    def __init__(self, obj) -> None: ...
    def __hash__(self): ...
    def __eq__(self, other): ...
    @property
    def obj(self):
        """Return object if alive, otherwise None."""

def _bsr_scatter_mm_indices_data(indices_format, M, K, N, Ms, Ks, nbatches, SPLIT_N, compressed_sparse_tensor_as_key): ...
def bsr_scatter_mm_indices_data(bsr, other, indices_format: str = 'bsr_strided_mm_compressed', **meta_input):
    """Computes indices data for :func:`scatter_mm` used in BSR and
    strided tensor matrix multiplication.
    """
def bsr_scatter_mm(bsr, other, indices_data=None, out=None):
    """BSR @ strided -> strided"""
def _int_bsr_dense_addmm(input: torch.Tensor, bsr: torch.Tensor, dense: torch.Tensor, *, beta: int = 1, alpha: int = 1, left_alpha: torch.Tensor | None = None, right_alpha: torch.Tensor | None = None, out: torch.Tensor | None = None, skip_checks: bool = False, max_grid: tuple[int | None, int | None, int | None] | None = None, meta: dict | None = None): ...
def bsr_dense_addmm(input: torch.Tensor, bsr: torch.Tensor, dense: torch.Tensor, *, beta: int = 1, alpha: int = 1, left_alpha: torch.Tensor | None = None, right_alpha: torch.Tensor | None = None, out: torch.Tensor | None = None, skip_checks: bool = False, max_grid: tuple[int | None, int | None, int | None] | None = None, meta: dict | None = None):
    """Compute

      out = beta * input + left_alpha.reshape(-1, 1) * (alpha * (bsr @ dense)) * right_alpha.reshape(1, -1)

    where left_alpha, right_alpha are (* + 1)-D tensors when
    specified, otherwise, these are treated as tensors filled with
    ones.
    """
@triton.jit
def _sampled_addmm_kernel(alpha, beta, IS_BETA_ZERO: tl.constexpr, BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, k, TILE_K: tl.constexpr, values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, mat1_ptr, mat1_batch_stride, mat1_tiled_row_stride, mat1_tiled_col_stride, mat1_row_block_stride, mat1_col_block_stride, mat2_ptr, mat2_batch_stride, mat2_tiled_row_stride, mat2_tiled_col_stride, mat2_row_block_stride, mat2_col_block_stride, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr): ...
@triton.jit
def _bsr_strided_dense_rowspace_kernel(values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, dense_ptr, dense_batch_stride, dense_tiled_row_stride, dense_tiled_col_stride, dense_row_block_stride, dense_col_block_stride, output_ptr, output_batch_stride, output_tiled_row_stride, output_tiled_col_stride, output_row_block_stride, output_col_block_stride, BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr, GROUP_SIZE_ROW: tl.constexpr): ...
def _run_sampled_addmm_kernel(alpha, beta, is_beta_zero, blocksize, k, tile_k, values, crow_indices, col_indices, mat1, mat2, max_grid) -> None: ...
def sampled_addmm(input: torch.Tensor, mat1: torch.Tensor, mat2: torch.Tensor, *, beta: float = 1.0, alpha: float = 1.0, out: torch.Tensor | None = None, skip_checks: bool = False, max_grid: tuple[int | None, int | None, int | None] | None = None): ...
def bsr_dense_mm(bsr: torch.Tensor, dense: torch.Tensor, *, out: torch.Tensor | None = None, skip_checks: bool = False, max_grid: tuple[int | None, int | None, int | None] | None = None, meta: dict | None = None): ...
@triton.jit
def _bsr_softmax_kernel(crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, values_ptr, values_batch_stride, values_row_block_stride, values_nnz_col_block_stride, row_block, col_block, MAX_ROW_NNZ: tl.constexpr, TILE: tl.constexpr): ...
def bsr_softmax(input, max_row_nnz=None): ...
def _scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: torch.Tensor | None, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None): ...
@triton.jit
def _scatter_mm2_kernel(M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_Q, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_R, accumulators_stride_M, accumulators_stride_N, pq_offsets_ptr, pq_offsets_stride, pq_ptr, pq_stride_T, pq_stride_1, dot_out_dtype: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, allow_tf32: tl.constexpr): ...
def _scatter_mm2(blocks: torch.Tensor, others: torch.Tensor, pq_offsets: torch.Tensor, pq_indices: torch.Tensor, accumulators: torch.Tensor): ...
@triton.jit
def _scatter_mm6_kernel(nbatches, Ms, Ks: tl.constexpr, N, blocks_ptr, blocks_stride_P, blocks_stride_M, blocks_stride_K, others_ptr, others_stride_B, others_stride_K, others_stride_N, accumulators_ptr, accumulators_stride_B, accumulators_stride_M, accumulators_stride_N, c_indices_ptr, r_offsets_ptr, p_offsets_ptr, q_offsets_ptr, is_compressed: tl.constexpr, dot_out_dtype: tl.constexpr, SPLIT_N: tl.constexpr, TILE_M: tl.constexpr, TILE_N: tl.constexpr, GROUP_SIZE: tl.constexpr, allow_tf32: tl.constexpr): ...
def _scatter_mm6(blocks: torch.Tensor, others: torch.Tensor, c_indices: torch.Tensor, r_offsets: torch.Tensor, p_offsets: torch.Tensor, q_offsets: torch.Tensor, meta: dict, accumulators: torch.Tensor, force_contiguous: bool = True): ...
@triton.jit
def _bsr_strided_addmm_kernel(values_ptr, values_batch_stride, values_nnz_stride, values_row_block_stride, values_col_block_stride, crow_indices_ptr, crow_indices_batch_stride, crow_indices_stride, col_indices_ptr, col_indices_batch_stride, col_indices_stride, input_ptr, input_batch_stride, input_tiled_row_stride, input_tiled_col_stride, input_row_block_stride, input_col_block_stride, dense_ptr, dense_batch_stride, dense_tiled_row_stride, dense_tiled_col_stride, dense_row_block_stride, dense_col_block_stride, left_alpha_ptr, left_alpha_batch_stride, left_alpha_tiled_row_stride, left_alpha_tiled_col_stride: tl.constexpr, left_alpha_row_block_stride, left_alpha_col_block_stride: tl.constexpr, right_alpha_ptr, right_alpha_batch_stride, right_alpha_tiled_row_stride: tl.constexpr, right_alpha_tiled_col_stride, right_alpha_row_block_stride: tl.constexpr, right_alpha_col_block_stride, output_ptr, output_batch_stride, output_tiled_row_stride, output_tiled_col_stride, output_row_block_stride, output_col_block_stride, beta, alpha, beta_is_one: tl.constexpr, beta_is_nonzero: tl.constexpr, alpha_is_one: tl.constexpr, left_alpha_is_one: tl.constexpr, right_alpha_is_one: tl.constexpr, BLOCKSIZE_ROW: tl.constexpr, BLOCKSIZE_COL: tl.constexpr, BLOCKSIZE_INNER: tl.constexpr, acc_dtype: tl.constexpr, allow_tf32: tl.constexpr, GROUP_SIZE_ROW: tl.constexpr, SPLIT_N: tl.constexpr): ...

bsr_softmax: Incomplete
bsr_dense_mm: Incomplete
sampled_addmm: Incomplete
_scaled_dot_product_attention: Incomplete
_scatter_mm2: Incomplete
_scatter_mm6: Incomplete
_bsr_strided_addmm_kernel: Incomplete
