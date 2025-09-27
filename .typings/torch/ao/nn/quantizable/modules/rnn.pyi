import torch
from _typeshed import Incomplete
from torch import Tensor

__all__ = ['LSTMCell', 'LSTM']

class LSTMCell(torch.nn.Module):
    """A quantizable long short-term memory (LSTM) cell.

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTMCell`

    `split_gates`: specify True to compute the input/forget/cell/output gates separately
    to avoid an intermediate tensor which is subsequently chunk'd. This optimization can
    be beneficial for on-device inference latency. This flag is cascaded down from the
    parent classes.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTMCell(10, 20)
        >>> input = torch.randn(6, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """
    _FLOAT_MODULE = torch.nn.LSTMCell
    __constants__: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    bias: Incomplete
    split_gates: Incomplete
    igates: torch.nn.Module
    hgates: torch.nn.Module
    gates: torch.nn.Module
    input_gate: Incomplete
    forget_gate: Incomplete
    cell_gate: Incomplete
    output_gate: Incomplete
    fgate_cx: Incomplete
    igate_cgate: Incomplete
    fgate_cx_igate_cgate: Incomplete
    ogate_cy: Incomplete
    initial_hidden_state_qparams: tuple[float, int]
    initial_cell_state_qparams: tuple[float, int]
    hidden_state_dtype: torch.dtype
    cell_state_dtype: torch.dtype
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, device=None, dtype=None, *, split_gates: bool = False) -> None: ...
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]: ...
    def initialize_hidden(self, batch_size: int, is_quantized: bool = False) -> tuple[Tensor, Tensor]: ...
    def _get_name(self): ...
    @classmethod
    def from_params(cls, wi, wh, bi=None, bh=None, split_gates: bool = False):
        """Uses the weights and biases to create a new LSTM cell.

        Args:
            wi, wh: Weights for the input and hidden layers
            bi, bh: Biases for the input and hidden layers
        """
    @classmethod
    def from_float(cls, other, use_precomputed_fake_quant: bool = False, split_gates: bool = False): ...

class _LSTMSingleLayer(torch.nn.Module):
    """A single one-directional LSTM layer.

    The difference between a layer and a cell is that the layer can process a
    sequence, while the cell only expects an instantaneous value.
    """
    cell: Incomplete
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, device=None, dtype=None, *, split_gates: bool = False) -> None: ...
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None): ...
    @classmethod
    def from_params(cls, *args, **kwargs): ...

class _LSTMLayer(torch.nn.Module):
    """A single bi-directional LSTM layer."""
    batch_first: Incomplete
    bidirectional: Incomplete
    layer_fw: Incomplete
    layer_bw: Incomplete
    def __init__(self, input_dim: int, hidden_dim: int, bias: bool = True, batch_first: bool = False, bidirectional: bool = False, device=None, dtype=None, *, split_gates: bool = False) -> None: ...
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None): ...
    @classmethod
    def from_float(cls, other, layer_idx: int = 0, qconfig=None, **kwargs):
        """
        There is no FP equivalent of this class. This function is here just to
        mimic the behavior of the `prepare` within the `torch.ao.quantization`
        flow.
        """

class LSTM(torch.nn.Module):
    """A quantizable long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples below.

    Examples::

        >>> import torch.ao.nn.quantizable as nnqa
        >>> rnn = nnqa.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
        >>> # To get the weights:
        >>> # xdoctest: +SKIP
        >>> print(rnn.layers[0].weight_ih)
        tensor([[...]])
        >>> print(rnn.layers[0].weight_hh)
        AssertionError: There is no reverse path in the non-bidirectional layer
    """
    _FLOAT_MODULE = torch.nn.LSTM
    input_size: Incomplete
    hidden_size: Incomplete
    num_layers: Incomplete
    bias: Incomplete
    batch_first: Incomplete
    dropout: Incomplete
    bidirectional: Incomplete
    training: bool
    layers: Incomplete
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None, *, split_gates: bool = False) -> None: ...
    def forward(self, x: Tensor, hidden: tuple[Tensor, Tensor] | None = None): ...
    def _get_name(self): ...
    @classmethod
    def from_float(cls, other, qconfig=None, split_gates: bool = False): ...
    @classmethod
    def from_observed(cls, other) -> None: ...
