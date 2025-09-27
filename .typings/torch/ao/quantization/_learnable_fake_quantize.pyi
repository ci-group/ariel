import torch
from _typeshed import Incomplete
from torch.nn.parameter import Parameter as Parameter

__all__: list[str]

class _LearnableFakeQuantize(torch.ao.quantization.FakeQuantizeBase):
    """Generalized extension of the FakeQuantize module in fake_quantize.py.

    This is an extension of the FakeQuantize module in fake_quantize.py, which
    supports more generalized lower-bit quantization and supports learning of the scale
    and zero point parameters through backpropagation.

    In addition to the attributes in the original FakeQuantize module, the _LearnableFakeQuantize
    module also includes the following attributes to support quantization parameter learning.

    * :attr:`channel_len` defines the length of the channel when initializing scale and zero point
      for the per channel case.

    * :attr:`use_grad_scaling` defines the flag for whether the gradients for scale and zero point are
      normalized by the constant, which is proportional to the square root of the number of
      elements in the tensor. The related literature justifying the use of this particular constant
      can be found here: https://openreview.net/pdf?id=rkgO66VKDS.

    * :attr:`fake_quant_enabled` defines the flag for enabling fake quantization on the output.

    * :attr:`static_enabled` defines the flag for using observer's static estimation for
      scale and zero point.

    * :attr:`learning_enabled` defines the flag for enabling backpropagation for scale and zero point.
    """
    quant_min: Incomplete
    quant_max: Incomplete
    use_grad_scaling: Incomplete
    scale: Incomplete
    zero_point: Incomplete
    activation_post_process: Incomplete
    dtype: Incomplete
    qscheme: Incomplete
    ch_axis: Incomplete
    bitwidth: Incomplete
    def __init__(self, observer, quant_min: int = 0, quant_max: int = 255, scale: float = 1.0, zero_point: float = 0.0, channel_len: int = -1, use_grad_scaling: bool = False, **observer_kwargs) -> None: ...
    @torch.jit.export
    def enable_param_learning(self):
        """Enable parameter learning over static observer estimates.

        Enables learning of quantization parameters and
        disables static observer estimates. Forward path returns fake quantized X.
        """
    @torch.jit.export
    def enable_static_estimate(self) -> None:
        """Enable static estimates of quantization parameters.

        Enables static observer estimates and disables learning of
        quantization parameters. Forward path returns fake quantized X.
        """
    @torch.jit.export
    def enable_static_observation(self) -> None:
        """Enable accumulation of data without updating quantization parameters.

        Enables static observer accumulating data from input but doesn't
        update the quantization parameters. Forward path returns the original X.
        """
    @torch.jit.export
    def toggle_observer_update(self, enabled: bool = True): ...
    @torch.jit.export
    def enable_observer(self, enabled: bool = True) -> None: ...
    @torch.jit.export
    def toggle_qparam_learning(self, enabled: bool = True): ...
    @torch.jit.export
    def toggle_fake_quant(self, enabled: bool = True): ...
    @torch.jit.export
    def observe_quant_params(self) -> None: ...
    @torch.jit.export
    def calculate_qparams(self): ...
    def forward(self, X): ...
