import torch
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase, QConfig as QConfig, QConfigMapping as QConfigMapping, default_weight_fake_quant as default_weight_fake_quant, default_weight_observer as default_weight_observer
from torch.ao.quantization.backend_config import BackendConfig as BackendConfig
from torch.ao.quantization.observer import _PartialWrapper as _PartialWrapper
from torch.ao.quantization.quantize_fx import convert_to_reference_fx as convert_to_reference_fx, prepare_fx as prepare_fx
from typing import Any

def _get_lstm_with_individually_observed_parts(float_lstm: torch.nn.LSTM, example_inputs: tuple[Any, ...], backend_config: BackendConfig | None = None, linear_output_obs_ctr: _PartialWrapper | None = None, sigmoid_obs_ctr: _PartialWrapper | None = None, tanh_obs_ctr: _PartialWrapper | None = None, cell_state_obs_ctr: _PartialWrapper | None = None, hidden_state_obs_ctr: _PartialWrapper | None = None, split_gates: bool = False) -> torch.ao.nn.quantizable.LSTM:
    """
    Return an observed `torch.ao.nn.quantizable.LSTM` created from a `torch.nn.LSTM`
    with specific observers or fake quantizes assigned to the inner ops or submodules.

    In both eager and FX graph mode quantization, `torch.ao.nn.quantizable.LSTM` is
    used as an observed custom module, which is responsible for inserting its own
    observers. By default, all inner ops inherit the parent custom module's QConfig.
    Users who wish to override this behavior may extend `torch.ao.nn.quantizable.LSTM`
    and use this helper function to customize the observer insertion logic.

    This is meant to be used to convert a float module to an observed module in the
    custom module flow.

    Args:
        `float_lstm`: The float LSTM module
        `example_inputs`: example inputs for the forward function of the LSTM module
        `backend_config`: BackendConfig to use to observe the LSTM module
        `linear_output_obs_ctr`: observer or fake quantize for linear outputs Wx + b,
            where W is the weight matrix, b is the bias, and x is either the inputs
            or the hidden state from the previous layer (if any)
        `sigmoid_obs_ctr`: observer or fake quantize for sigmoid activations
        `tanh_obs_ctr`: observer or fake quantize for tanh activations
        `cell_state_obs_ctr`: observer or fake quantize for the cell state
        `hidden_state_obs_ctr`: observer or fake quantize for the hidden state and
            the output

    Return:
        A `torch.ao.nn.quantizable.LSTM` with the specified observers or fake quantizes
        assigned to the inner ops.
    """
def _get_reference_quantized_lstm_module(observed_lstm: torch.ao.nn.quantizable.LSTM, backend_config: BackendConfig | None = None) -> torch.ao.nn.quantized.LSTM:
    """
    Return a `torch.ao.nn.quantized.LSTM` created from a `torch.ao.nn.quantizable.LSTM`
    with observers or fake quantizes inserted through `prepare_fx`, e.g. from
    `_get_lstm_with_individually_observed_parts`.

    This is meant to be used to convert an observed module to a quantized module in the
    custom module flow.

    Args:
        `observed_lstm`: a `torch.ao.nn.quantizable.LSTM` observed through `prepare_fx`
        `backend_config`: BackendConfig to use to produce the reference quantized model

    Return:
        A reference `torch.ao.nn.quantized.LSTM` module.
    """
