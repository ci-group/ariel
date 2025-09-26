import torch

__all__ = ['LSTM']

class LSTM(torch.ao.nn.quantizable.LSTM):
    """A quantized long short-term memory (LSTM).

    For the description and the argument types, please, refer to :class:`~torch.nn.LSTM`

    Attributes:
        layers : instances of the `_LSTMLayer`

    .. note::
        To access the weights and biases, you need to access them per layer.
        See examples in :class:`~torch.ao.nn.quantizable.LSTM`

    Examples::
        >>> # xdoctest: +SKIP
        >>> custom_module_config = {
        ...     'float_to_observed_custom_module_class': {
        ...         nn.LSTM: nn.quantizable.LSTM,
        ...     },
        ...     'observed_to_quantized_custom_module_class': {
        ...         nn.quantizable.LSTM: nn.quantized.LSTM,
        ...     }
        ... }
        >>> tq.prepare(model, prepare_custom_module_class=custom_module_config)
        >>> tq.convert(model, convert_custom_module_class=custom_module_config)
    """
    _FLOAT_MODULE = torch.ao.nn.quantizable.LSTM
    def _get_name(self): ...
    @classmethod
    def from_float(cls, *args, **kwargs) -> None: ...
    @classmethod
    def from_observed(cls, other): ...
