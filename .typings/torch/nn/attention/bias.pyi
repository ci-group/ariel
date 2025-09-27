import torch
from _typeshed import Incomplete
from enum import IntEnum

__all__ = ['causal_upper_left', 'causal_lower_right', 'CausalVariant', 'CausalBias']

class CausalVariant(IntEnum):
    """
    Enum for causal variants used in attention mechanisms.

    Defines two types of causal biases:

    ``UPPER_LEFT``: Represents upper-left triangular bias for standard causal attention.
    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]


    ``LOWER_RIGHT``: Represents lower-right triangular bias, the include values are aligned to the lower
    right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with ``shape=(3,4)``, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Note that these variants are equivalent to each other when the sequence lengths of the query and key/value
    tensors are equal since the triangular matrix is square.

    .. warning:: This enum is a prototype and subject to change.
    """
    UPPER_LEFT = ...
    LOWER_RIGHT = ...

class CausalBias(torch.Tensor):
    '''
    A bias representing causal attention patterns. For an overview of the bias structure, see the :class:`CausalVariant` enum.

    This class is used for defining causal (triangular) attention biases. For construing the bias, there exist
    two factory functions: :func:`causal_upper_left` and :func:`causal_lower_right`.

    Example:

    .. code-block:: python

        from torch.nn.attention.bias import causal_lower_right

        bsz, num_heads, seqlen_q, seqlen_kv, head_dim = 32, 8, 4, 12, 8

        # Create a lower-right causal bias
        attn_bias = causal_lower_right(seqlen_q, seqlen_kv)

        q = torch.randn(
            bsz, num_heads, seqlen_q, head_dim, device="cuda", dtype=torch.float16
        )
        k = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            bsz, num_heads, seqlen_kv, head_dim, device="cuda", dtype=torch.float16
        )

        out = F.scaled_dot_product_attention(q, k, v, attn_bias)

    .. warning:: This class is a prototype and subject to change.
    '''
    variant: Incomplete
    seq_len_q: Incomplete
    seq_len_kv: Incomplete
    def __init__(self, variant: CausalVariant, seq_len_q: int, seq_len_kv: int) -> None:
        """
        Initializes the CausalBias instance with a specified variant and sequence lengths.

        Args:
            variant (CausalVariant): The type of causal bias to use (either UPPER_LEFT or LOWER_RIGHT).
            seq_len_q (int): The sequence length of the query tensor.
            seq_len_kv (int): The sequence length of the key/value tensor.

        Raises a warning if the LOWER_RIGHT variant is used with seq_len_q > seq_len_kv, as it may produce NaNs.
        """
    def _upper_left(self, device: torch.device) -> torch.Tensor:
        """Upper left causal bias"""
    def _lower_right(self, device: torch.device) -> torch.Tensor:
        """Lower right causal bias"""
    def _materialize(self, device: torch.device | None = None) -> torch.Tensor:
        """
        Materializes the causal bias into a tensor form.

        Depending on the variant, this method generates either an upper-left or lower-right
        triangular matrix to represent the causal bias.

        Args:
            device (Optional[torch.device]): The device on which to create the tensor. Defaults to CPU.

        Returns:
            torch.Tensor: The materialized bias tensor.
        """
    @staticmethod
    def _dispatch(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_mask: CausalBias, dropout_p: float = 0.0, is_causal: bool = False, scale: float | None = None, enable_gqa: bool = False) -> torch.Tensor:
        """
        Handles the logic for computing attention with the specified causal bias.

        Args:
            query (Tensor): Query tensor; shape :math:`(N, ..., L, E)`.
            key (Tensor): Key tensor; shape :math:`(N, ..., S, E)`.
            value (Tensor): Value tensor; shape :math:`(N, ..., S, Ev)`.
            attn_mask (CausalBias): The type of causal attention to apply.
                A boolean mask where a value of True indicates that the element *should* take part in attention.
                A float mask of the same type as query, key, value that is added to the attention score.
            dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
            is_causal (bool): If true, assumes upper left causal attention masking and errors if both attn_mask and is_causal
                are set.
            scale (optional float): Scaling factor applied prior to softmax. If None, the default value is set
                to :math:`\\frac{1}{\\sqrt{E}}`.
            enable_gqa (optional bool): If set to True, Grouped Query Attention (GQA) is enabled, by default it is set to False.

        Returns:
            output (Tensor): Attention output; shape :math:`(N, ..., L, Ev)`.

        Raises:
            ValueError: If the causal bias variant is not a CausalVariant type.

        """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """Defines the behavior of torch.nn.functional.scaled_dot_product_attention when the attn_bias is an AttnBias"""
    def __repr__(self) -> str: ...

def causal_upper_left(*size) -> CausalBias:
    """
    Creates an upper-left triangular causal bias.

    This function generates a upper-left triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the upper left corner of the matrix.
    This equivalent to the `is_causal=True` argument in `scaled_dot_product_attention`.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        torch.tril(torch.ones(size, dtype=torch.bool))

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The UPPER_LEFT triangular causal bias variant.
    """
def causal_lower_right(*size) -> CausalBias:
    """
    Creates a lower-right triangular causal bias.

    This function generates a lower-right triangular matrix to represent causal attention bias with a
    diagonal offset set so that the inclusive values are aligned to the lower right corner of the matrix.

    The equivalent pytorch code for constructing this bias is:

    .. code-block:: python

        diagonal_offset = size[1] - size[0]
        torch.tril(
            torch.ones(size, dtype=torch.bool),
            diagonal=diagonal_offset,
        )

    For instance, with `shape=(3,4)`, the materialized bias tensor will be:

    .. code-block:: text

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    Args:
        size: The size of the bias matrix.

    Returns:
        CausalBias: The LOWER_RIGHT triangular causal bias variant.
    """
