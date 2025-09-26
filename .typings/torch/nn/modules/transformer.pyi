import torch
from .module import Module
from _typeshed import Incomplete
from torch import Tensor
from typing import Any, Callable

__all__ = ['Transformer', 'TransformerEncoder', 'TransformerDecoder', 'TransformerEncoderLayer', 'TransformerDecoderLayer']

class Transformer(Module):
    '''A basic transformer layer.


    This Transformer layer implements the original Transformer architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build an efficient transformer layer from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of encoder/decoder intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples:
        >>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
        >>> src = torch.rand((10, 32, 512))
        >>> tgt = torch.rand((20, 32, 512))
        >>> out = transformer_model(src, tgt)

    Note: A full example to apply nn.Transformer module for the word language model is available in
    https://github.com/pytorch/examples/tree/master/word_language_model
    '''
    encoder: Incomplete
    decoder: Incomplete
    d_model: Incomplete
    nhead: Incomplete
    batch_first: Incomplete
    def __init__(self, d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str | Callable[[Tensor], Tensor] = ..., custom_encoder: Any | None = None, custom_decoder: Any | None = None, layer_norm_eps: float = 1e-05, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None: ...
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None, tgt_key_padding_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None, src_is_causal: bool | None = None, tgt_is_causal: bool | None = None, memory_is_causal: bool = False) -> Tensor:
        """Take in and process masked source/target sequences.

        .. note::

            If a boolean tensor is provided for any of the [src/tgt/memory]_mask arguments, positions with a ``True`` value are
            not allowed to participate in the attention,
            which is the opposite of the definition for :attr:`attn_mask`
            in :func:`torch.nn.functional.scaled_dot_product_attention`.

        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
            src_mask: the additive mask for the src sequence (optional).
            tgt_mask: the additive mask for the tgt sequence (optional).
            memory_mask: the additive mask for the encoder output (optional).
            src_key_padding_mask: the Tensor mask for src keys per batch (optional).
            tgt_key_padding_mask: the Tensor mask for tgt keys per batch (optional).
            memory_key_padding_mask: the Tensor mask for memory keys per batch (optional).
            src_is_causal: If specified, applies a causal mask as ``src_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``src_is_causal`` provides a hint that ``src_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            tgt_is_causal: If specified, applies a causal mask as ``tgt_mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory_mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            - src: :math:`(S, E)` for unbatched input, :math:`(S, N, E)` if `batch_first=False` or
              `(N, S, E)` if `batch_first=True`.
            - tgt: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.
            - src_mask: :math:`(S, S)` or :math:`(N\\cdot\\text{num\\_heads}, S, S)`.
            - tgt_mask: :math:`(T, T)` or :math:`(N\\cdot\\text{num\\_heads}, T, T)`.
            - memory_mask: :math:`(T, S)`.
            - src_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.
            - tgt_key_padding_mask: :math:`(T)` for unbatched input otherwise :math:`(N, T)`.
            - memory_key_padding_mask: :math:`(S)` for unbatched input otherwise :math:`(N, S)`.

            Note: [src/tgt/memory]_mask ensures that position :math:`i` is allowed to attend the unmasked
            positions. If a BoolTensor is provided, positions with ``True``
            are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
            is provided, it will be added to the attention weight.
            [src/tgt/memory]_key_padding_mask provides specified elements in the key to be ignored by
            the attention. If a BoolTensor is provided, the positions with the
            value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.

            - output: :math:`(T, E)` for unbatched input, :math:`(T, N, E)` if `batch_first=False` or
              `(N, T, E)` if `batch_first=True`.

            Note: Due to the multi-head attention architecture in the transformer model,
            the output sequence length of a transformer is same as the input sequence
            (i.e. target) length of the decoder.

            where :math:`S` is the source sequence length, :math:`T` is the target sequence length, :math:`N` is the
            batch size, :math:`E` is the feature number

        Examples:
            >>> # xdoctest: +SKIP
            >>> output = transformer_model(
            ...     src, tgt, src_mask=src_mask, tgt_mask=tgt_mask
            ... )
        """
    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> Tensor:
        """Generate a square causal mask for the sequence.

        The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
        """
    def _reset_parameters(self) -> None:
        """Initiate parameters in the transformer model."""

class TransformerEncoder(Module):
    """TransformerEncoder is a stack of N encoder layers.

    This TransformerEncoder layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    .. warning::
        All layers in the TransformerEncoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerEncoder instance.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__: Incomplete
    layers: Incomplete
    num_layers: Incomplete
    norm: Incomplete
    enable_nested_tensor: Incomplete
    use_nested_tensor: Incomplete
    mask_check: Incomplete
    def __init__(self, encoder_layer: TransformerEncoderLayer, num_layers: int, norm: Module | None = None, enable_nested_tensor: bool = True, mask_check: bool = True) -> None: ...
    def forward(self, src: Tensor, mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None, is_causal: bool | None = None) -> Tensor:
        """Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """

class TransformerDecoder(Module):
    """TransformerDecoder is a stack of N decoder layers.

    This TransformerDecoder layer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    .. warning::
        All layers in the TransformerDecoder are initialized with the same parameters.
        It is recommended to manually initialize the layers after creating the TransformerDecoder instance.

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__: Incomplete
    layers: Incomplete
    num_layers: Incomplete
    norm: Incomplete
    def __init__(self, decoder_layer: TransformerDecoderLayer, num_layers: int, norm: Module | None = None) -> None: ...
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None, tgt_key_padding_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None, tgt_is_causal: bool | None = None, memory_is_causal: bool = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """

class TransformerEncoderLayer(Module):
    '''TransformerEncoderLayer is made up of self-attn and feedforward network.

    This TransformerEncoderLayer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    TransformerEncoderLayer can handle either traditional torch.tensor inputs,
    or Nested Tensor inputs.  Derived classes are expected to similarly accept
    both input formats.  (Not all combinations of inputs are currently
    supported by TransformerEncoderLayer while Nested Tensor is in prototype
    state.)

    If you are implementing a custom layer, you may derive it either from
    the Module or TransformerEncoderLayer class.  If your custom layer
    supports both torch.Tensors and Nested Tensors inputs, make its
    implementation a derived class of TransformerEncoderLayer. If your custom
    Layer supports only torch.Tensor inputs, derive its implementation from
    Module.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it\'s done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(
        ...     d_model=512, nhead=8, batch_first=True
        ... )
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation described in
        `FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`_ if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.

        .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    '''
    __constants__: Incomplete
    self_attn: Incomplete
    linear1: Incomplete
    dropout: Incomplete
    linear2: Incomplete
    norm_first: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    dropout1: Incomplete
    dropout2: Incomplete
    activation_relu_or_gelu: int
    activation: Incomplete
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str | Callable[[Tensor], Tensor] = ..., layer_norm_eps: float = 1e-05, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, src: Tensor, src_mask: Tensor | None = None, src_key_padding_mask: Tensor | None = None, is_causal: bool = False) -> Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
    def _sa_block(self, x: Tensor, attn_mask: Tensor | None, key_padding_mask: Tensor | None, is_causal: bool = False) -> Tensor: ...
    def _ff_block(self, x: Tensor) -> Tensor: ...

class TransformerDecoderLayer(Module):
    '''TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.

    This TransformerDecoderLayer implements the original architecture described
    in the `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_ paper. The
    intent of this layer is as a reference implementation for foundational understanding
    and thus it contains only limited features relative to newer Transformer architectures.
    Given the fast pace of innovation in transformer-like architectures, we recommend
    exploring this `tutorial <https://pytorch.org/tutorials/intermediate/transformer_building_blocks.html>`_
    to build efficient layers from building blocks in core or using higher
    level libraries from the `PyTorch Ecosystem <https://landscape.pytorch.org/>`_.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it\'s done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(
        ...     d_model=512, nhead=8, batch_first=True
        ... )
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    '''
    __constants__: Incomplete
    self_attn: Incomplete
    multihead_attn: Incomplete
    linear1: Incomplete
    dropout: Incomplete
    linear2: Incomplete
    norm_first: Incomplete
    norm1: Incomplete
    norm2: Incomplete
    norm3: Incomplete
    dropout1: Incomplete
    dropout2: Incomplete
    dropout3: Incomplete
    activation: Incomplete
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str | Callable[[Tensor], Tensor] = ..., layer_norm_eps: float = 1e-05, batch_first: bool = False, norm_first: bool = False, bias: bool = True, device=None, dtype=None) -> None: ...
    def __setstate__(self, state) -> None: ...
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor | None = None, memory_mask: Tensor | None = None, tgt_key_padding_mask: Tensor | None = None, memory_key_padding_mask: Tensor | None = None, tgt_is_causal: bool = False, memory_is_causal: bool = False) -> Tensor:
        """Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
    def _sa_block(self, x: Tensor, attn_mask: Tensor | None, key_padding_mask: Tensor | None, is_causal: bool = False) -> Tensor: ...
    def _mha_block(self, x: Tensor, mem: Tensor, attn_mask: Tensor | None, key_padding_mask: Tensor | None, is_causal: bool = False) -> Tensor: ...
    def _ff_block(self, x: Tensor) -> Tensor: ...
