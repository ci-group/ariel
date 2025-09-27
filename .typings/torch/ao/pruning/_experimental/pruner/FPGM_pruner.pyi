from .base_structured_sparsifier import BaseStructuredSparsifier
from _typeshed import Incomplete
from typing import Callable

__all__ = ['FPGMPruner']

class FPGMPruner(BaseStructuredSparsifier):
    """Filter Pruning via Geometric Median (FPGM) Structured Pruner
    This sparsifier prune fliter (row) in a tensor according to distances among filters according to
    `Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration <https://arxiv.org/abs/1811.00250>`_.

    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of filters (rows) that are zeroed-out.
    2. `dist` defines the distance measurement type. Default: 3 (L2 distance).
    Available options are: [1, 2, (custom callable distance function)].

    Note::
        Inputs should be a 4D convolutional tensor of shape (N, C, H, W).
            - N: output channels size
            - C: input channels size
            - H: height of kernel
            - W: width of kernel
    """
    dist_fn: Incomplete
    def __init__(self, sparsity_level: float = 0.5, dist: Callable | int | None = None) -> None: ...
    def _compute_distance(self, t):
        """Compute distance across all entries in tensor `t` along all dimension
        except for the one identified by dim.
        Args:
            t (torch.Tensor): tensor representing the parameter to prune
        Returns:
            distance (torch.Tensor): distance computed across filtters
        """
    def update_mask(self, module, tensor_name, sparsity_level, **kwargs) -> None: ...
