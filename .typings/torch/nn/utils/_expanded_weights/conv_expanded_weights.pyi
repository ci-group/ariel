import torch
from .conv_utils import conv_args_and_kwargs as conv_args_and_kwargs, conv_backward as conv_backward, conv_input_for_string_padding as conv_input_for_string_padding, conv_picker as conv_picker
from .expanded_weights_impl import ExpandedWeight as ExpandedWeight, implements_per_sample_grads as implements_per_sample_grads
from .expanded_weights_utils import forward_helper as forward_helper
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec

_P = ParamSpec('_P')
_R = TypeVar('_R')

class ConvPerSampleGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, kwarg_names: list[str], conv_fn: Callable[_P, _R], *expanded_args_and_kwargs: Any) -> torch.Tensor: ...
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any: ...
