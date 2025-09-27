import torch
import torch.nn as nn
from _typeshed import Incomplete
from torch import Tensor
from torch._jit_internal import Optional
from torch.nn.utils.rnn import PackedSequence

__all__ = ['pack_weight_bias', 'PackedParameter', 'RNNBase', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell', 'GRUCell', 'apply_permutation']

def apply_permutation(tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor: ...
def pack_weight_bias(qweight, bias, dtype): ...

class PackedParameter(torch.nn.Module):
    param: Incomplete
    def __init__(self, param) -> None: ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...

class RNNBase(torch.nn.Module):
    _FLOAT_MODULE = nn.RNNBase
    _version: int
    mode: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    num_layers: Incomplete
    bias: Incomplete
    batch_first: Incomplete
    dropout: Incomplete
    bidirectional: Incomplete
    dtype: Incomplete
    version: int
    training: bool
    _all_weight_values: Incomplete
    def __init__(self, mode, input_size, hidden_size, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, dtype=...) -> None: ...
    def _get_name(self): ...
    def extra_repr(self): ...
    def __repr__(self) -> str: ...
    def check_input(self, input: Tensor, batch_sizes: Optional[Tensor]) -> None: ...
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Optional[Tensor]) -> tuple[int, int, int]: ...
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: tuple[int, int, int], msg: str = 'Expected hidden size {}, got {}') -> None: ...
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None: ...
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    def set_weight_bias(self, weight_bias_dict): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    def _weight_bias(self): ...
    def get_weight(self): ...
    def get_bias(self): ...

class LSTM(RNNBase):
    """
    A dynamic quantized LSTM module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.LSTM`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """
    _FLOAT_MODULE = nn.LSTM
    __overloads__: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_name(self): ...
    def forward_impl(self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]], batch_sizes: Optional[Tensor], max_batch_size: int, sorted_indices: Optional[Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    @torch.jit.export
    def forward_tensor(self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    @torch.jit.export
    def forward_packed(self, input: PackedSequence, hx: Optional[tuple[Tensor, Tensor]] = None) -> tuple[PackedSequence, tuple[Tensor, Tensor]]: ...
    def permute_hidden(self, hx: tuple[Tensor, Tensor], permutation: Optional[Tensor]) -> tuple[Tensor, Tensor]: ...
    def check_forward_args(self, input: Tensor, hidden: tuple[Tensor, Tensor], batch_sizes: Optional[Tensor]) -> None: ...
    @torch.jit.ignore
    def forward(self, input, hx=None): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_mod): ...

class GRU(RNNBase):
    """Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.


    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \\begin{array}{ll}
            r_t = \\sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\\\\n            z_t = \\sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\\\\n            n_t = \\tanh(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{(t-1)}+ b_{hn})) \\\\\n            h_t = (1 - z_t) \\odot n_t + z_t \\odot h_{(t-1)}
        \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is the input
    at time `t`, :math:`h_{(t-1)}` is the hidden state of the layer
    at time `t-1` or the initial hidden state at time `0`, and :math:`r_t`,
    :math:`z_t`, :math:`n_t` are the reset, update, and new gates, respectively.
    :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.

    In a multilayer GRU, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l >= 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\\delta^{(l-1)}_t` where each :math:`\\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        - **input** of shape `(seq_len, batch, input_size)`: tensor containing the features
          of the input sequence. The input can also be a packed variable length
          sequence. See :func:`torch.nn.utils.rnn.pack_padded_sequence`
          for details.
        - **h_0** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the initial hidden state for each element in the batch.
          Defaults to zero if not provided. If the RNN is bidirectional,
          num_directions should be 2, else it should be 1.

    Outputs: output, h_n
        - **output** of shape `(seq_len, batch, num_directions * hidden_size)`: tensor
          containing the output features h_t from the last layer of the GRU,
          for each `t`. If a :class:`torch.nn.utils.rnn.PackedSequence` has been
          given as the input, the output will also be a packed sequence.
          For the unpacked case, the directions can be separated
          using ``output.view(seq_len, batch, num_directions, hidden_size)``,
          with forward and backward being direction `0` and `1` respectively.

          Similarly, the directions can be separated in the packed case.
        - **h_n** of shape `(num_layers * num_directions, batch, hidden_size)`: tensor
          containing the hidden state for `t = seq_len`

          Like *output*, the layers can be separated using
          ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

    Shape:
        - Input1: :math:`(L, N, H_{in})` tensor containing input features where
          :math:`H_{in}=\\text{input\\_size}` and `L` represents a sequence length.
        - Input2: :math:`(S, N, H_{out})` tensor
          containing the initial hidden state for each element in the batch.
          :math:`H_{out}=\\text{hidden\\_size}`
          Defaults to zero if not provided. where :math:`S=\\text{num\\_layers} * \\text{num\\_directions}`
          If the RNN is bidirectional, num_directions should be 2, else it should be 1.
        - Output1: :math:`(L, N, H_{all})` where :math:`H_{all}=\\text{num\\_directions} * \\text{hidden\\_size}`
        - Output2: :math:`(S, N, H_{out})` tensor containing the next hidden state
          for each element in the batch

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\\text{k}^{th}` layer
            (W_ir|W_iz|W_in), of shape `(3*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(3*hidden_size, num_directions * hidden_size)`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\\text{k}^{th}` layer
            (W_hr|W_hz|W_hn), of shape `(3*hidden_size, hidden_size)`
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\\text{k}^{th}` layer
            (b_ir|b_iz|b_in), of shape `(3*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\\text{k}^{th}` layer
            (b_hr|b_hz|b_hn), of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    .. note::
        The calculation of new gate :math:`n_t` subtly differs from the original paper and other frameworks.
        In the original implementation, the Hadamard product :math:`(\\odot)` between :math:`r_t` and the
        previous hidden state :math:`h_{(t-1)}` is done before the multiplication with the weight matrix
        `W` and addition of bias:

        .. math::
            \\begin{aligned}
                n_t = \\tanh(W_{in} x_t + b_{in} + W_{hn} ( r_t \\odot h_{(t-1)} ) + b_{hn})
            \\end{aligned}

        This is in contrast to PyTorch implementation, which is done after :math:`W_{hn} h_{(t-1)}`

        .. math::
            \\begin{aligned}
                n_t = \\tanh(W_{in} x_t + b_{in} + r_t \\odot (W_{hn} h_{(t-1)}+ b_{hn}))
            \\end{aligned}

        This implementation differs on purpose for efficiency.

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """
    _FLOAT_MODULE = nn.GRU
    __overloads__: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_name(self): ...
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Optional[Tensor]) -> None: ...
    def forward_impl(self, input: Tensor, hx: Optional[Tensor], batch_sizes: Optional[Tensor], max_batch_size: int, sorted_indices: Optional[Tensor]) -> tuple[Tensor, Tensor]: ...
    @torch.jit.export
    def forward_tensor(self, input: Tensor, hx: Optional[Tensor] = None) -> tuple[Tensor, Tensor]: ...
    @torch.jit.export
    def forward_packed(self, input: PackedSequence, hx: Optional[Tensor] = None) -> tuple[PackedSequence, Tensor]: ...
    def permute_hidden(self, hx: Tensor, permutation: Optional[Tensor]) -> Tensor: ...
    @torch.jit.ignore
    def forward(self, input, hx=None): ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_mod): ...

class RNNCellBase(torch.nn.Module):
    __constants__: Incomplete
    input_size: Incomplete
    hidden_size: Incomplete
    bias: Incomplete
    weight_dtype: Incomplete
    bias_ih: Incomplete
    bias_hh: Incomplete
    _packed_weight_ih: Incomplete
    _packed_weight_hh: Incomplete
    def __init__(self, input_size, hidden_size, bias: bool = True, num_chunks: int = 4, dtype=...) -> None: ...
    def _get_name(self): ...
    def extra_repr(self): ...
    def check_forward_input(self, input) -> None: ...
    def check_forward_hidden(self, input: Tensor, hx: Tensor, hidden_label: str = '') -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
    @classmethod
    def from_reference(cls, ref_mod): ...
    def _weight_bias(self): ...
    def get_weight(self): ...
    def get_bias(self): ...
    def set_weight_bias(self, weight_bias_dict) -> None: ...
    def _save_to_state_dict(self, destination, prefix, keep_vars) -> None: ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...

class RNNCell(RNNCellBase):
    """An Elman RNN cell with tanh or ReLU non-linearity.
    A dynamic quantized RNNCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.RNNCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.RNNCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    __constants__: Incomplete
    nonlinearity: Incomplete
    def __init__(self, input_size, hidden_size, bias: bool = True, nonlinearity: str = 'tanh', dtype=...) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell.

    A dynamic quantized LSTMCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.LSTMCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.LSTMCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, hx: Optional[tuple[Tensor, Tensor]] = None) -> tuple[Tensor, Tensor]: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...

class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell

    A dynamic quantized GRUCell module with floating point tensor as inputs and outputs.
    Weights are quantized to 8 bits. We adopt the same interface as `torch.nn.GRUCell`,
    please see https://pytorch.org/docs/stable/nn.html#torch.nn.GRUCell for documentation.

    Examples::

        >>> # xdoctest: +SKIP
        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    def __init__(self, input_size, hidden_size, bias: bool = True, dtype=...) -> None: ...
    def _get_name(self): ...
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False): ...
