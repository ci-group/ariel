def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    """
    This is PyTorch implementation of main part of reorder_meta()
    function, from tools/util/include/cutlass/util/host_reorder.h file
    of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
    GEMM decides upon layout of this matrix, and at the moment for the
    sparse GEMM executed on tensor cores, this is layout described by
    ColumnMajorInterleaved<2> data structure, in
    include/cutlass/layout/matrix.h of CUTLASS source tree.  The
    reordering of meta matrix into meta_reordered matrix calculated
    according to these segments of CUTLASS code is re-implemented here.
    Note that this calculation produces offsets for scattering metadata
    matrix elements into reordered metadata matrix elements (or,
    equivalently, for gathering reordered metadata matrix element back
    into metadata matrix elements).
    """
def sparse_semi_structured_from_dense_cutlass(dense):
    '''
    This function converts dense matrix into sparse semi-structured
    representation, producing "compressed" matrix, in the layout used by
    CUTLASS backend, and corresponding metadata matrix.
    '''
def sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    '''
    This function performs reverse of the function above - it
    reconstructs dense matrix from a pair of "compressed" matrix, given
    in the layout used by CUTLASS backend, and accompanying metadata
    matrix.
    '''
def _sparse_semi_structured_tile(dense):
    """
    This function computes a 2:4 sparse tile by greedily taking the largest values.

    Since we take the largest values greedily, how the sorting algorithm handles duplicates affects
    the ultimate sparsity pattern.

    Note that this function does not have the same sorting semantics as our CUDA backend,
    which is exposed via `torch._sparse_semi_structured_tile` and thus returns a different pattern.
    """
def _compute_compressed_swizzled_bitmask(dense):
    """
    Calculates the compressed swizzled bitmask from a dense tensor
    """
