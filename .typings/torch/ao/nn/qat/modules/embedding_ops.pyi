import torch.nn as nn
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['Embedding', 'EmbeddingBag']

class Embedding(nn.Embedding):
    """
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.Embedding`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
    for documentation.

    Similar to `torch.nn.Embedding`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.Embedding
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, _weight=None, device=None, dtype=None, qconfig=None) -> None: ...
    def forward(self, input) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
    def to_float(self): ...

class EmbeddingBag(nn.EmbeddingBag):
    """
    An embedding bag module attached with FakeQuantize modules for weight,
    used for quantization aware training.

    We adopt the same interface as `torch.nn.EmbeddingBag`, please see
    https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html#torch.nn.EmbeddingBag
    for documentation.

    Similar to `torch.nn.EmbeddingBag`, with FakeQuantize modules initialized to
    default.

    Attributes:
        weight: fake quant module for weight
    """
    _FLOAT_MODULE = nn.EmbeddingBag
    qconfig: Incomplete
    weight_fake_quant: Incomplete
    def __init__(self, num_embeddings, embedding_dim, max_norm=None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, mode: str = 'mean', sparse: bool = False, _weight=None, include_last_offset: bool = False, padding_idx=None, qconfig=None, device=None, dtype=None) -> None: ...
    def forward(self, input, offsets=None, per_sample_weights=None) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a qat module from a float module

        Args: `mod` a float module, either produced by torch.ao.quantization utilities
        or directly from user
        """
    def to_float(self): ...
