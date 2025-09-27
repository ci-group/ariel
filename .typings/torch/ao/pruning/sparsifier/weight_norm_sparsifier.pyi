from .base_sparsifier import BaseSparsifier
from _typeshed import Incomplete
from typing import Callable

__all__ = ['WeightNormSparsifier']

class WeightNormSparsifier(BaseSparsifier):
    '''Weight-Norm Sparsifier

    This sparsifier computes the norm of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.

    Args:

        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block (see note below)
        zeros_per_block: Number of zeros in a sparse block
        norm: Norm to use. Could be either `int` or a callable.
            If `int`, only L1 and L2 are implemented.

    Note::
        The `sparse_block_shape` is tuple representing (block_ROWS, block_COLS),
        irrespective of what the rows / cols mean in the data tensor. That means,
        if you were to sparsify a weight tensor in the nn.Linear, which has a
        weight shape `(Cout, Cin)`, the `block_ROWS` would refer to the output
        channels, while the `block_COLS` would refer to the input channels.

    Note::
        All arguments to the WeightNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `prepare` step.
    '''
    norm_fn: Incomplete
    def __init__(self, sparsity_level: float = 0.5, sparse_block_shape: tuple[int, int] = (1, 4), zeros_per_block: int | None = None, norm: Callable | int | None = None) -> None: ...
    def _scatter_fold_block_mask(self, output_shape, dim, indices, block_shape, mask=None, input_shape=None, device=None):
        """Creates patches of size `block_shape` after scattering the indices."""
    def _make_tensor_mask(self, data, input_shape, sparsity_level, sparse_block_shape, mask=None):
        '''Creates a tensor-level mask.

        Tensor-level mask is described as a mask, where the granularity of sparsification of the
        smallest patch is the sparse_block_shape. That means, that for a given mask and a
        sparse_block_shape, the smallest "patch" of zeros/ones could be the sparse_block_shape.

        In this context, `sparsity_level` describes the fraction of sparse patches.
        '''
    def _make_block_mask(self, data, sparse_block_shape, zeros_per_block, mask=None):
        """Creates a block-level mask.

        Block-level mask is described as a mask, where the granularity of sparsification of the
        largest patch is the sparse_block_shape. That means that for a given mask and a
        sparse_block_shape, the sparsity is computed only within a patch of a size sparse_block_shape.

        In this context the `zeros_per_block` describes the number of zeroed-out elements within a patch.
        """
    def update_mask(self, module, tensor_name, sparsity_level, sparse_block_shape, zeros_per_block, **kwargs) -> None: ...
