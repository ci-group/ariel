import torch
import weakref
from .module import Module
from _typeshed import Incomplete
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import PackedSequence
from typing import overload

__all__ = ['RNNBase', 'RNN', 'LSTM', 'GRU', 'RNNCellBase', 'RNNCell', 'LSTMCell', 'GRUCell']

class RNNBase(Module):
    """Base class for RNN modules (RNN, LSTM, GRU).

    Implements aspects of RNNs shared by the RNN, LSTM, and GRU classes, such as module initialization
    and utility methods for parameter storage management.

    .. note::
        The forward method is not implemented by the RNNBase class.

    .. note::
        LSTM and GRU classes override some methods implemented by RNNBase.
    """
    __constants__: Incomplete
    __jit_unused_properties__: Incomplete
    mode: str
    input_size: int
    hidden_size: int
    num_layers: int
    bias: bool
    batch_first: bool
    dropout: float
    bidirectional: bool
    proj_size: int
    _flat_weight_refs: list[weakref.ReferenceType[Parameter] | None]
    _flat_weights_names: Incomplete
    _all_weights: Incomplete
    def __init__(self, mode: str, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, proj_size: int = 0, device=None, dtype=None) -> None: ...
    _flat_weights: Incomplete
    def _init_flat_weights(self) -> None: ...
    def __setattr__(self, attr, value) -> None: ...
    def flatten_parameters(self) -> None:
        """Reset parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """
    def _apply(self, fn, recurse: bool = True): ...
    def reset_parameters(self) -> None: ...
    def check_input(self, input: Tensor, batch_sizes: Tensor | None) -> None: ...
    def get_expected_hidden_size(self, input: Tensor, batch_sizes: Tensor | None) -> tuple[int, int, int]: ...
    def check_hidden_size(self, hx: Tensor, expected_hidden_size: tuple[int, int, int], msg: str = 'Expected hidden size {}, got {}') -> None: ...
    def _weights_have_changed(self): ...
    def check_forward_args(self, input: Tensor, hidden: Tensor, batch_sizes: Tensor | None): ...
    def permute_hidden(self, hx: Tensor, permutation: Tensor | None): ...
    def extra_repr(self) -> str: ...
    def _update_flat_weights(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, d) -> None: ...
    @property
    def all_weights(self) -> list[list[Parameter]]: ...
    def _replicate_for_data_parallel(self): ...

class RNN(RNNBase):
    '''__init__(input_size,hidden_size,num_layers=1,nonlinearity=\'tanh\',bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer Elman RNN with :math:`\\tanh` or :math:`\\text{ReLU}`
    non-linearity to an input sequence. For each element in the input sequence,
    each layer computes the following function:

    .. math::
        h_t = \\tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})

    where :math:`h_t` is the hidden state at time `t`, :math:`x_t` is
    the input at time `t`, and :math:`h_{(t-1)}` is the hidden state of the
    previous layer at time `t-1` or the initial hidden state at time `0`.
    If :attr:`nonlinearity` is ``\'relu\'``, then :math:`\\text{ReLU}` is used instead of :math:`\\tanh`.

    .. code-block:: python

        # Efficient implementation equivalent to the following with bidirectional=False
        rnn = nn.RNN(input_size, hidden_size, num_layers)
        params = dict(rnn.named_parameters())
        def forward(x, hx=None, batch_first=False):
            if batch_first:
                x = x.transpose(0, 1)
            seq_len, batch_size, _ = x.size()
            if hx is None:
                hx = torch.zeros(rnn.num_layers, batch_size, rnn.hidden_size)
            h_t_minus_1 = hx.clone()
            h_t = hx.clone()
            output = []
            for t in range(seq_len):
                for layer in range(rnn.num_layers):
                    input_t = x[t] if layer == 0 else h_t[layer - 1]
                    h_t[layer] = torch.tanh(
                        input_t @ params[f"weight_ih_l{layer}"].T
                        + h_t_minus_1[layer] @ params[f"weight_hh_l{layer}"].T
                        + params[f"bias_hh_l{layer}"]
                        + params[f"bias_ih_l{layer}"]
                    )
                output.append(h_t[-1].clone())
                h_t_minus_1 = h_t.clone()
            output = torch.stack(output)
            if batch_first:
                output = output.transpose(0, 1)
            return output, h_t

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two RNNs together to form a `stacked RNN`,
            with the second RNN taking in outputs of the first RNN and
            computing the final results. Default: 1
        nonlinearity: The non-linearity to use. Can be either ``\'tanh\'`` or ``\'relu\'``. Default: ``\'tanh\'``
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``

    Inputs: input, hx
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **hx**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the initial hidden
          state for the input sequence batch. Defaults to zeros if not provided.

        where:

        .. math::
            \\begin{aligned}
                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{out} ={} & \\text{hidden\\_size}
            \\end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the RNN, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the final hidden state
          for each element in the batch.

    Attributes:
        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size, input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size, num_directions * hidden_size)`
        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size, hidden_size)`
        bias_ih_l[k]: the learnable input-hidden bias of the k-th layer,
            of shape `(hidden_size)`
        bias_hh_l[k]: the learnable hidden-hidden bias of the k-th layer,
            of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    .. note::
        For bidirectional RNNs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.RNN(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    '''
    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: Tensor, hx: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: PackedSequence, hx: Tensor | None = None) -> tuple[PackedSequence, Tensor]: ...

class LSTM(RNNBase):
    """__init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,proj_size=0,device=None,dtype=None)

    Apply a multi-layer long short-term memory (LSTM) RNN to an input sequence.
    For each element in the input sequence, each layer computes the following
    function:

    .. math::
        \\begin{array}{ll} \\\\\n            i_t = \\sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\\n            f_t = \\sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\\n            g_t = \\tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\\n            o_t = \\sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\\n            c_t = f_t \\odot c_{t-1} + i_t \\odot g_t \\\\\n            h_t = o_t \\odot \\tanh(c_t) \\\\\n        \\end{array}

    where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
    state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
    is the hidden state of the layer at time `t-1` or the initial hidden
    state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
    :math:`o_t` are the input, forget, cell, and output gates, respectively.
    :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.

    In a multilayer LSTM, the input :math:`x^{(l)}_t` of the :math:`l` -th layer
    (:math:`l \\ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
    dropout :math:`\\delta^{(l-1)}_t` where each :math:`\\delta^{(l-1)}_t` is a Bernoulli random
    variable which is :math:`0` with probability :attr:`dropout`.

    If ``proj_size > 0`` is specified, LSTM with projections will be used. This changes
    the LSTM cell in the following way. First, the dimension of :math:`h_t` will be changed from
    ``hidden_size`` to ``proj_size`` (dimensions of :math:`W_{hi}` will be changed accordingly).
    Second, the output hidden state of each layer will be multiplied by a learnable projection
    matrix: :math:`h_t = W_{hr}h_t`. Note that as a consequence of this, the output
    of LSTM network will be of different shape as well. See Inputs/Outputs sections below for exact
    dimensions of all variables. You can find more details in https://arxiv.org/abs/1402.1128.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        proj_size: If ``> 0``, will use LSTM with projections of corresponding size. Default: 0

    Inputs: input, (h_0, c_0)
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the
          initial hidden state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **c_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the
          initial cell state for each element in the input sequence.
          Defaults to zeros if (h_0, c_0) is not provided.

        where:

        .. math::
            \\begin{aligned}
                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{cell} ={} & \\text{hidden\\_size} \\\\\n                H_{out} ={} & \\text{proj\\_size if } \\text{proj\\_size}>0 \\text{ otherwise hidden\\_size} \\\\\n            \\end{aligned}

    Outputs: output, (h_n, c_n)
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the LSTM, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence. When ``bidirectional=True``, `output` will contain
          a concatenation of the forward and reverse hidden states at each time step in the sequence.
        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the
          final hidden state for each element in the sequence. When ``bidirectional=True``,
          `h_n` will contain a concatenation of the final forward and reverse hidden states, respectively.
        * **c_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{cell})` for unbatched input or
          :math:`(D * \\text{num\\_layers}, N, H_{cell})` containing the
          final cell state for each element in the sequence. When ``bidirectional=True``,
          `c_n` will contain a concatenation of the final forward and reverse cell states, respectively.

    Attributes:
        weight_ih_l[k] : the learnable input-hidden weights of the :math:`\\text{k}^{th}` layer
            `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
            Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
            ``proj_size > 0`` was specified, the shape will be
            `(4*hidden_size, num_directions * proj_size)` for `k > 0`
        weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\\text{k}^{th}` layer
            `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
            was specified, the shape will be `(4*hidden_size, proj_size)`.
        bias_ih_l[k] : the learnable input-hidden bias of the :math:`\\text{k}^{th}` layer
            `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
        bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\\text{k}^{th}` layer
            `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
        weight_hr_l[k] : the learnable projection weights of the :math:`\\text{k}^{th}` layer
            of shape `(proj_size, hidden_size)`. Only present when ``proj_size > 0`` was
            specified.
        weight_ih_l[k]_reverse: Analogous to `weight_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hh_l[k]_reverse:  Analogous to `weight_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_ih_l[k]_reverse:  Analogous to `bias_ih_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        bias_hh_l[k]_reverse:  Analogous to `bias_hh_l[k]` for the reverse direction.
            Only present when ``bidirectional=True``.
        weight_hr_l[k]_reverse:  Analogous to `weight_hr_l[k]` for the reverse direction.
            Only present when ``bidirectional=True`` and ``proj_size > 0`` was specified.

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    .. note::
        For bidirectional LSTMs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        For bidirectional LSTMs, `h_n` is not equivalent to the last element of `output`; the
        former contains the final forward and reverse hidden states, while the latter contains the
        final forward hidden state and the initial reverse hidden state.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

    .. note::
        ``proj_size`` should be smaller than ``hidden_size``.

    .. include:: ../cudnn_rnn_determinism.rst

    .. include:: ../cudnn_persistent_rnn.rst

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> c0 = torch.randn(2, 3, 20)
        >>> output, (hn, cn) = rnn(input, (h0, c0))
    """
    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, proj_size: int = 0, device=None, dtype=None) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    def get_expected_cell_size(self, input: Tensor, batch_sizes: Tensor | None) -> tuple[int, int, int]: ...
    def check_forward_args(self, input: Tensor, hidden: tuple[Tensor, Tensor], batch_sizes: Tensor | None): ...
    def permute_hidden(self, hx: tuple[Tensor, Tensor], permutation: Tensor | None) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: PackedSequence, hx: tuple[Tensor, Tensor] | None = None) -> tuple[PackedSequence, tuple[Tensor, Tensor]]: ...

class GRU(RNNBase):
    """__init__(input_size,hidden_size,num_layers=1,bias=True,batch_first=False,dropout=0.0,bidirectional=False,device=None,dtype=None)

    Apply a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
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
    (:math:`l \\ge 2`) is the hidden state :math:`h^{(l-1)}_t` of the previous layer multiplied by
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
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            GRU layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``

    Inputs: input, h_0
        * **input**: tensor of shape :math:`(L, H_{in})` for unbatched input,
          :math:`(L, N, H_{in})` when ``batch_first=False`` or
          :math:`(N, L, H_{in})` when ``batch_first=True`` containing the features of
          the input sequence.  The input can also be a packed variable length sequence.
          See :func:`torch.nn.utils.rnn.pack_padded_sequence` or
          :func:`torch.nn.utils.rnn.pack_sequence` for details.
        * **h_0**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` or
          :math:`(D * \\text{num\\_layers}, N, H_{out})`
          containing the initial hidden state for the input sequence. Defaults to zeros if not provided.

        where:

        .. math::
            \\begin{aligned}
                N ={} & \\text{batch size} \\\\\n                L ={} & \\text{sequence length} \\\\\n                D ={} & 2 \\text{ if bidirectional=True otherwise } 1 \\\\\n                H_{in} ={} & \\text{input\\_size} \\\\\n                H_{out} ={} & \\text{hidden\\_size}
            \\end{aligned}

    Outputs: output, h_n
        * **output**: tensor of shape :math:`(L, D * H_{out})` for unbatched input,
          :math:`(L, N, D * H_{out})` when ``batch_first=False`` or
          :math:`(N, L, D * H_{out})` when ``batch_first=True`` containing the output features
          `(h_t)` from the last layer of the GRU, for each `t`. If a
          :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input, the output
          will also be a packed sequence.
        * **h_n**: tensor of shape :math:`(D * \\text{num\\_layers}, H_{out})` or
          :math:`(D * \\text{num\\_layers}, N, H_{out})` containing the final hidden state
          for the input sequence.

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
        For bidirectional GRUs, forward and backward are directions 0 and 1 respectively.
        Example of splitting the output layers when ``batch_first=False``:
        ``output.view(seq_len, batch, num_directions, hidden_size)``.

    .. note::
        ``batch_first`` argument is ignored for unbatched inputs.

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

        >>> rnn = nn.GRU(10, 20, 2)
        >>> input = torch.randn(5, 3, 10)
        >>> h0 = torch.randn(2, 3, 20)
        >>> output, hn = rnn(input, h0)
    """
    @overload
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bias: bool = True, batch_first: bool = False, dropout: float = 0.0, bidirectional: bool = False, device=None, dtype=None) -> None: ...
    @overload
    def __init__(self, *args, **kwargs) -> None: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: Tensor, hx: Tensor | None = None) -> tuple[Tensor, Tensor]: ...
    @overload
    @torch._jit_internal._overload_method
    def forward(self, input: PackedSequence, hx: Tensor | None = None) -> tuple[PackedSequence, Tensor]: ...

class RNNCellBase(Module):
    __constants__: Incomplete
    input_size: int
    hidden_size: int
    bias: bool
    weight_ih: Tensor
    weight_hh: Tensor
    bias_ih: Incomplete
    bias_hh: Incomplete
    def __init__(self, input_size: int, hidden_size: int, bias: bool, num_chunks: int, device=None, dtype=None) -> None: ...
    def extra_repr(self) -> str: ...
    def reset_parameters(self) -> None: ...

class RNNCell(RNNCellBase):
    """An Elman RNN cell with tanh or ReLU non-linearity.

    .. math::

        h' = \\tanh(W_{ih} x + b_{ih}  +  W_{hh} h + b_{hh})

    If :attr:`nonlinearity` is `'relu'`, then ReLU is used in place of tanh.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`.
            Default: ``True``
        nonlinearity: The non-linearity to use. Can be either ``'tanh'`` or ``'relu'``. Default: ``'tanh'``

    Inputs: input, hidden
        - **input**: tensor containing input features
        - **hidden**: tensor containing the initial hidden state
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** of shape `(batch, hidden_size)`: tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    Examples::

        >>> rnn = nn.RNNCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    __constants__: Incomplete
    nonlinearity: str
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh', device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor: ...

class LSTMCell(RNNCellBase):
    """A long short-term memory (LSTM) cell.

    .. math::

        \\begin{array}{ll}
        i = \\sigma(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\\\\n        f = \\sigma(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\\\\n        g = \\tanh(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\\\\n        o = \\sigma(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\\\\n        c' = f \\odot c + i \\odot g \\\\\n        h' = o \\odot \\tanh(c') \\\\\n        \\end{array}

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, (h_0, c_0)
        - **input** of shape `(batch, input_size)` or `(input_size)`: tensor containing input features
        - **h_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial hidden state
        - **c_0** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the initial cell state

          If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.

    Outputs: (h_1, c_1)
        - **h_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next hidden state
        - **c_1** of shape `(batch, hidden_size)` or `(hidden_size)`: tensor containing the next cell state

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(4*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(4*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(4*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(4*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.LSTMCell(10, 20)  # (input_size, hidden_size)
        >>> input = torch.randn(2, 3, 10)  # (time_steps, batch, input_size)
        >>> hx = torch.randn(3, 20)  # (batch, hidden_size)
        >>> cx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(input.size()[0]):
        ...     hx, cx = rnn(input[i], (hx, cx))
        ...     output.append(hx)
        >>> output = torch.stack(output, dim=0)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, hx: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]: ...

class GRUCell(RNNCellBase):
    """A gated recurrent unit (GRU) cell.

    .. math::

        \\begin{array}{ll}
        r = \\sigma(W_{ir} x + b_{ir} + W_{hr} h + b_{hr}) \\\\\n        z = \\sigma(W_{iz} x + b_{iz} + W_{hz} h + b_{hz}) \\\\\n        n = \\tanh(W_{in} x + b_{in} + r \\odot (W_{hn} h + b_{hn})) \\\\\n        h' = (1 - z) \\odot n + z \\odot h
        \\end{array}

    where :math:`\\sigma` is the sigmoid function, and :math:`\\odot` is the Hadamard product.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        bias: If ``False``, then the layer does not use bias weights `b_ih` and
            `b_hh`. Default: ``True``

    Inputs: input, hidden
        - **input** : tensor containing input features
        - **hidden** : tensor containing the initial hidden
          state for each element in the batch.
          Defaults to zero if not provided.

    Outputs: h'
        - **h'** : tensor containing the next hidden state
          for each element in the batch

    Shape:
        - input: :math:`(N, H_{in})` or :math:`(H_{in})` tensor containing input features where
          :math:`H_{in}` = `input_size`.
        - hidden: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the initial hidden
          state where :math:`H_{out}` = `hidden_size`. Defaults to zero if not provided.
        - output: :math:`(N, H_{out})` or :math:`(H_{out})` tensor containing the next hidden state.

    Attributes:
        weight_ih: the learnable input-hidden weights, of shape
            `(3*hidden_size, input_size)`
        weight_hh: the learnable hidden-hidden weights, of shape
            `(3*hidden_size, hidden_size)`
        bias_ih: the learnable input-hidden bias, of shape `(3*hidden_size)`
        bias_hh: the learnable hidden-hidden bias, of shape `(3*hidden_size)`

    .. note::
        All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k = \\frac{1}{\\text{hidden\\_size}}`

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Examples::

        >>> rnn = nn.GRUCell(10, 20)
        >>> input = torch.randn(6, 3, 10)
        >>> hx = torch.randn(3, 20)
        >>> output = []
        >>> for i in range(6):
        ...     hx = rnn(input[i], hx)
        ...     output.append(hx)
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, device=None, dtype=None) -> None: ...
    def forward(self, input: Tensor, hx: Tensor | None = None) -> Tensor: ...
