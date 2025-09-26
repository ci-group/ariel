import torch
from _typeshed import Incomplete
from torch import Tensor
from torch._jit_internal import Optional

__all__ = ['EmbeddingPackedParams', 'Embedding', 'EmbeddingBag']

class EmbeddingPackedParams(torch.nn.Module):
    _version: int
    dtype: Incomplete
    def __init__(self, num_embeddings, embedding_dim, dtype=...) -> None: ...
    _packed_weight: Incomplete
    @torch.jit.export
    def set_weight(self, weight: torch.Tensor) -> None: ...
    @torch.jit.export
    def _weight(self): ...
    def forward(self, x): ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    def __repr__(self) -> str: ...

class Embedding(torch.nn.Module):
    """
    A quantized Embedding module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html for documentation.

    Similar to :class:`~torch.nn.Embedding`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\\text{num\\_embeddings}, \\text{embedding\\_dim})`.

    Examples::
        >>> m = nn.quantized.Embedding(num_embeddings=10, embedding_dim=12)
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        >>> output = m(indices)
        >>> print(output.size())
        torch.Size([9, 12])

    """
    _version: int
    num_embeddings: Incomplete
    embedding_dim: Incomplete
    dtype: Incomplete
    _packed_params: Incomplete
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: Optional[Tensor] = None, dtype=...) -> None: ...
    def forward(self, indices: Tensor) -> Tensor: ...
    def _get_name(self): ...
    def __repr__(self) -> str: ...
    def extra_repr(self): ...
    def set_weight(self, w: torch.Tensor) -> None: ...
    def weight(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a quantized embedding module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
    @classmethod
    def from_reference(cls, ref_embedding): ...

class EmbeddingBag(Embedding):
    """
    A quantized EmbeddingBag module with quantized packed weights as inputs.
    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html for documentation.

    Similar to :class:`~torch.nn.EmbeddingBag`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module of
                         shape :math:`(\\text{num\\_embeddings}, \\text{embedding\\_dim})`.

    Examples::
        >>> m = nn.quantized.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, mode='sum')
        >>> indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        >>> offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        >>> output = m(indices, offsets)
        >>> print(output.size())
        torch.Size([5, 12])

    """
    _version: int
    mode: Incomplete
    pruned_weights: bool
    include_last_offset: Incomplete
    dtype: Incomplete
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, mode: str = 'sum', sparse: bool = False, _weight: Optional[Tensor] = None, include_last_offset: bool = False, dtype=...) -> None: ...
    def forward(self, indices: Tensor, offsets: Optional[Tensor] = None, per_sample_weights: Optional[Tensor] = None, compressed_indices_mapping: Optional[Tensor] = None) -> Tensor: ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a quantized embedding_bag module from a float module

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by user
        """
    @classmethod
    def from_reference(cls, ref_embedding_bag): ...
