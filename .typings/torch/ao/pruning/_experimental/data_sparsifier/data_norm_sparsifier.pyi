from .base_data_sparsifier import BaseDataSparsifier
from _typeshed import Incomplete
from typing import Any

__all__ = ['DataNormSparsifier']

class DataNormSparsifier(BaseDataSparsifier):
    '''L1-Norm Sparsifier
    This sparsifier computes the *L1-norm* of every sparse block and "zeroes-out" the
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
        sparse_block_shape: The shape of a sparse block
        zeros_per_block: Number of zeros in a sparse block
    Note::
        All arguments to the DataNormSparsifier constructor are "default"
        arguments and could be overridden by the configuration provided in the
        `add_data` step.
    '''
    norm: Incomplete
    def __init__(self, data_list: list[tuple[str, Any]] | None = None, sparsity_level: float = 0.5, sparse_block_shape: tuple[int, int] = (1, 4), zeros_per_block: int | None = None, norm: str = 'L1') -> None: ...
    def __get_scatter_folded_mask(self, data, dim, indices, output_size, sparse_block_shape): ...
    def __get_block_level_mask(self, data, sparse_block_shape, zeros_per_block): ...
    def __get_data_level_mask(self, data, sparsity_level, sparse_block_shape): ...
    def update_mask(self, name, data, sparsity_level, sparse_block_shape, zeros_per_block, **kwargs) -> None: ...
