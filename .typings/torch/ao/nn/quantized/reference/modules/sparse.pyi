import torch.nn as nn
from .utils import ReferenceQuantizedModule
from torch import Tensor
from typing import Any

__all__ = ['Embedding', 'EmbeddingBag']

class Embedding(nn.Embedding, ReferenceQuantizedModule):
    """A reference quantized Embedding module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None, max_norm: float | None = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: Tensor | None = None, device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams): ...

class EmbeddingBag(nn.EmbeddingBag, ReferenceQuantizedModule):
    """A reference quantized EmbeddingBag module that fits into the
    FX Graph Mode Quantization workflow, activation will be floating point Tensor,
    we will store floating point weight as well in the module, but in forward we'll
    quantize and dequantize the weight before running the floating point functional
    embedding operator.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, max_norm: float | None = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, mode: str = 'mean', sparse: bool = False, _weight: Tensor | None = None, include_last_offset: bool = False, padding_idx: int | None = None, device=None, dtype=None, weight_qparams: dict[str, Any] | None = None) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, offsets: Tensor | None = None, per_sample_weights: Tensor | None = None) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, weight_qparams, use_precomputed_fake_quant: bool = False): ...
