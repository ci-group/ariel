import torch.jit
from _typeshed import Incomplete
from torch import Tensor, nn

__all__ = ['MultiheadAttention']

class MultiheadAttention(nn.MultiheadAttention):
    _FLOAT_MODULE = nn.MultiheadAttention
    __constants__: Incomplete
    linear_Q: Incomplete
    linear_K: Incomplete
    linear_V: Incomplete
    out_proj: Incomplete
    q_scaling_product: Incomplete
    quant_attn_output: Incomplete
    quant_attn_output_weights: Incomplete
    dequant_q: Incomplete
    dequant_k: Incomplete
    dequant_v: Incomplete
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True, add_bias_kv: bool = False, add_zero_attn: bool = False, kdim: int | None = None, vdim: int | None = None, batch_first: bool = False, device=None, dtype=None) -> None: ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, other): ...
    @torch.jit.unused
    def dequantize(self):
        """Utility to convert the quantized MHA back to float.

        The motivation for this is that it is not trivial to convert the weights
        from the format that is used in the quantized version back to the
        float.
        """
    @classmethod
    def from_observed(cls, other) -> None: ...
    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None, need_weights: bool = True, attn_mask: Tensor | None = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[Tensor, Tensor | None]:
        '''
        Note::
            Please, refer to :func:`~torch.nn.MultiheadAttention.forward` for more
            information

        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored.
            need_weights: output attn_output_weights.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.

        Shape:
            - Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
              the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
              If a BoolTensor is provided, the positions with the
              value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
            - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
              3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
              S is the source sequence length. attn_mask ensure that position i is allowed to attend the unmasked
              positions. If a BoolTensor is provided, positions with ``True``
              is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
              is provided, it will be added to the attention weight.
            - is_causal: If specified, applies a causal mask as attention mask. Mutually exclusive with providing attn_mask.
              Default: ``False``.
            - average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
              heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
              effect when ``need_weights=True.``. Default: True (i.e. average weights across heads)

            - Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
              E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: If ``average_attn_weights=True``, returns attention weights averaged
              across heads of shape :math:`(N, L, S)`, where N is the batch size, L is the target sequence length,
              S is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
              head of shape :math:`(N, num_heads, L, S)`.
        '''
    def _forward_impl(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Tensor | None = None, need_weights: bool = True, attn_mask: Tensor | None = None, average_attn_weights: bool = True, is_causal: bool = False) -> tuple[Tensor, Tensor | None]: ...
