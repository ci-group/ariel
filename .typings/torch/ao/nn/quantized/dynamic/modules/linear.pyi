import torch.ao.nn.quantized as nnq

__all__ = ['Linear']

class Linear(nnq.Linear):
    """
    A dynamic quantized linear module with floating point tensor as inputs and outputs.
    We adopt the same interface as `torch.nn.Linear`, please see
    https://pytorch.org/docs/stable/nn.html#torch.nn.Linear for documentation.

    Similar to :class:`torch.nn.Linear`, attributes will be randomly
    initialized at module creation time and will be overwritten later

    Attributes:
        weight (Tensor): the non-learnable quantized weights of the module which are of
                         shape :math:`(\\text{out\\_features}, \\text{in\\_features})`.
        bias (Tensor): the non-learnable floating point bias of the module of shape
                       :math:`(\\text{out\\_features})`. If :attr:`bias` is ``True``,
                       the values are initialized to zero.

    Examples::

        >>> # xdoctest: +SKIP
        >>> m = nn.quantized.dynamic.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    _version: int
    version: int
    def __init__(self, in_features, out_features, bias_: bool = True, dtype=...) -> None: ...
    def forward(self, x): ...
    def _get_name(self): ...
    def extra_repr(self): ...
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant: bool = False):
        """Create a dynamic quantized module from a float module or qparams_dict

        Args:
            mod (Module): a float module, either produced by torch.ao.quantization
                          utilities or provided by the user
        """
    @classmethod
    def from_reference(cls, ref_qlinear):
        """Create a (fbgemm/qnnpack) dynamic quantized module from a reference quantized
        module
        Args:
            ref_qlinear (Module): a reference quantized  module, either produced by
            torch.ao.quantization functions or provided by the user
        """
