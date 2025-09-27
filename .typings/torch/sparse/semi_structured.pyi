import torch
from _typeshed import Incomplete
from typing import Any, Callable, NamedTuple

__all__ = ['SparseSemiStructuredTensor', 'SparseSemiStructuredTensorCUTLASS', 'SparseSemiStructuredTensorCUSPARSELT', 'to_sparse_semi_structured']

class _SEMI_STRUCTURED_SPARSE_CONFIG(NamedTuple):
    sparse_min_rows: Incomplete
    sparse_min_cols: Incomplete
    dense_min_rows: Incomplete
    dense_min_cols: Incomplete

class SparseSemiStructuredTensor(torch.Tensor):
    """
    This class implements semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    There are two backends available for semi_structred sparsity, either cuSPARSELt or CUTLASS.
    This class is meant to serve as a base class for both implementations. SparseSemiStructuredCUTLASS
    and SparseSemiStructuredCUSPARSELT both inherit from this class and define three backend-specific items.
    Note that as such, this class cannot be instantiated directly.

    -`_DTYPE_SHAPE_CONSTRAINTS` - A dictionary holding backend specific dense/sparse min shape constraints
    - `def from_dense()` - backend specific compression routines
    - `def _mm()` - backend specific mm op (either torch._cslt_sparse_mm or torch._sparse_semi_structured_(mm|addmm))
    """
    _DEFAULT_ALG_ID: int
    _DTYPE_SHAPE_CONSTRAINTS: dict[torch.dtype, _SEMI_STRUCTURED_SPARSE_CONFIG]
    _FORCE_CUTLASS: bool
    _FUSE_TRANSPOSE: bool
    _PROTOTYPE_WARNING_SHOWN: bool
    BACKEND: str
    SPARSE_DISPATCH: dict[Callable, Callable]
    packed: torch.Tensor | None
    meta: torch.Tensor | None
    packed_t: torch.Tensor | None
    meta_t: torch.Tensor | None
    compressed_swizzled_bitmask: torch.Tensor | None
    fuse_transpose_cusparselt: bool
    alg_id_cusparselt: int
    __slots__: Incomplete
    @staticmethod
    def __new__(cls, shape: torch.Size, packed: torch.Tensor | None, meta: torch.Tensor | None, packed_t: torch.Tensor | None, meta_t: torch.Tensor | None, compressed_swizzled_bitmask: torch.Tensor | None, fuse_transpose_cusparselt: bool = False, alg_id_cusparselt: int = 0, requires_grad: bool = False):
        """
        Create a new instance of the tensor subclass from the compressed sparse representation.

        We have the option to create the subclass with the compressed representations of both X and X', for training.
        For inference, we only need a single representation (either X or X'), while the corresponding other set will be None.

        Depending on the backend selected, certain fields will be set to None. (CUSPARSELT vs CUTLASS)

        Args:
            shape: The shape of the original dense tensor
            packed: The compressed representation of the original dense tensor
            meta: The metadata of the original dense tensor, if it is stored separately
            packed_t: The compressed representation of the transposed original dense tensor
            meta_t: The metadata of the transposed original dense tensor, if it is stored separately
            compressed_swizzled_bitmask: The masks used by the CUTLASS backend to determine which threads should
                                         participate in the computation. Used for pointwise ops.
            fuse_transpose_cusparselt: When running with cuSPARSELt, we have the option to fuse a transposition
                                       with a matmul, which is useful in the case of 2:4 sparse training.
            alg_id_cusparselt: The algorithm id to use when using cuSPARSELT, will have effect on performance

        Returns:
            torch.Tensor: A torch.Tensor wrapper subclass.

        Raises:
            ValueError: If all of the tensor arguments are None.
        """
    def __repr__(self) -> str: ...
    def __tensor_flatten__(self) -> tuple[list[str], tuple[torch.Size, bool, int, bool]]: ...
    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, tensor_meta: tuple[torch.Size, bool, int, bool], outer_size, outer_stride) -> torch.Tensor: ...
    __torch_function__ = torch._C._disabled_torch_function_impl
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any: ...
    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None) -> None:
        """
        Loads the op overload sparse dispatch table for the current class.
        """
    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor: torch.Tensor) -> None:
        """
        Assert that the given tensor is valid for semi-structured sparse compression.
        """
    @classmethod
    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
    def to_dense(self): ...
    @classmethod
    def from_dense(cls, original_tensor: torch.Tensor) -> SparseSemiStructuredTensor: ...
    def _mm(self, B: torch.Tensor, *, bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor: ...

def to_sparse_semi_structured(original_tensor: torch.Tensor, transposed: bool = False) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of the minimum sparse block size, given in
    `_DTYPE_TO_SHAPE_CONSTRAINTS` for each dtype (float32, float16, bfloat16, int8).

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): deprecated arg to be removed in another release. Do not use.
    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor
    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]))
        >>> A_sparse.values()
        tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
        >>> A_sparse.indices()
        tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0', dtype=torch.int16))
    """

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    """
    This class implements semi-structured sparsity for the CUTLASS backend.


    In this implementation, the specified elements and metadata are stored separately,
    in packed and meta respectively.

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_(mm|addmm) and
    sparse_semi_structured_from_dense for conversion to the compressed format.
    """
    BACKEND: str
    _DTYPE_SHAPE_CONSTRAINTS: Incomplete
    @classmethod
    def from_dense(cls, original_tensor: torch.Tensor) -> SparseSemiStructuredTensorCUTLASS: ...
    def to_dense(self): ...
    @classmethod
    def prune_dense_static_sort(cls, original_tensor: torch.Tensor, algorithm: str = '') -> SparseSemiStructuredTensor:
        """
        This function takes in a unpruned dense tensor and runs a (branchless) static sort across a 4x4 tile.

        It greedily picks the largest values in the tile, upholding the 2:4 sparsity constraint across both rows and columns.
        The algorithm used to prune the matrix is implemented in `_sparse_semi_structured_tile`.

        Then it creates the packed and meta tensors for the compressed sparse representation of the pruned dense tensor.
        It also calculates the packed_t and meta_t tensors for the compressed sparse representation of the transposed
        pruned dense tensor.
        Since we cannot transpose the compressed representations, we store both for the fw/bw pass respectively.

        Finally, this function also computes a compressed swizzled bitmask that encodes the sparsity pattern
        This can be used in the backward pass to mask the gradients.

        [9 1 7 4]                       [9 0 7 0]
        [1 2 3 0]                       [0 2 0 0]
        [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to CUTLASS semi-structured -> packed
        [1 2 6 2]                       [0 0 6 2]                                    -> metadata

                                                  -> pack to transposed CUTLASS      -> packed_t
                                                     semi-structured representation  -> metadata_t

                                                  -> compute swizzled bitmask        -> compressed_swizzled_bitmask


        The equivalent PyTorch code to create the same five outputs from the dense tensor can be found below:
        ```
        from torch.sparse import SparseSemiStructuredTensorCUTLASS
        from torch.sparse._semi_structured_conversions import _sparse_semi_structured_tile, _compute_compressed_swizzled_bitmask

        pruned = _sparse_semi_structured_tile(dense)
        packed_cutlass, meta_cutlass = sparse_semi_structured_from_dense_cutlass(pruned)
        packed_t_cutlass, meta_t_cutlass = sparse_semi_structured_from_dense_cutlass(pruned.t().contiguous())
        bitmask = _compute_compressed_swizzled_bitmask(pruned)

        SparseSemiStructuredTensorCUTLASS(dense.shape, packed_cutlass, meta_cutlass, packed_t_cutlass, meta_t_cutlass, bitmask)
        ```
        """
    def _mm(self, B: torch.Tensor, *, bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor: ...

class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    """
    The cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
    packed = [ specified elements of original tensor | metadata ]
    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. Since there is only one tensor, we only use the packed and packed_t
    attributes respectively.

    cuSPARSELt also supports transposition fusion, which is necessary for performant 2:4 sparse training, as well
    as specifying alg_id, a config that affects the performance of the matmul depending on matmul sizes.
    """
    BACKEND: str
    _DTYPE_SHAPE_CONSTRAINTS: Incomplete
    @classmethod
    def from_dense(cls, original_tensor: torch.Tensor) -> SparseSemiStructuredTensorCUSPARSELT: ...
    @classmethod
    def prune_dense_static_sort(cls, original_tensor: torch.Tensor, algorithm: str = '') -> SparseSemiStructuredTensor:
        """
        This function does the same thing as described in SparseSemiStructuredCUTLASS, but uses the cuSPASRELt metadata
        layout and sparse matmul.

        The only functional difference is that cuSPARSELt stores `metadata` and `packed` together into a single tensor.

        [9 1 7 4]                       [9 0 7 0]
        [1 2 3 0]                       [0 2 0 0]
        [8 3 5 4] -> prune 4x4 tile  -> [8 0 0 4] -> pack to cuSPARSELT semi-structured -> packed
        [1 2 6 2]                       [0 0 6 2]

                                                  -> pack to transposed cuSPARSELt      -> packed_t
                                                     semi-structured representation

                                                  -> compute swizzled bitmask           -> compressed_swizzled_bitmask


        The equivalent PyTorch code to create the same three outputs from the dense tensor can be found below:
        ```
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT
        from torch.sparse._semi_structured_conversions import _sparse_semi_structured_tile, _compute_compressed_swizzled_bitmask

        pruned = _sparse_semi_structured_tile(dense)
        packed_cusparselt = torch._cslt_compress(pruned)
        packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
        bitmask = _compute_compressed_swizzled_bitmask(pruned)

        SparseSemiStructuredTensorCUSPARSELT(dense.shape, packed_cutlass, None, packed_t_cutlass, None, bitmask)
        ```
        """
    def _mm(self, B: torch.Tensor, *, bias: torch.Tensor | None = None, **kwargs) -> torch.Tensor: ...
