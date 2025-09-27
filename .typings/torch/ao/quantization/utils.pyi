import torch
from torch.fx import Node
from typing import Any, Callable

__all__ = ['NodePattern', 'Pattern', 'MatchAllNode', 'check_node', 'get_combined_dict', 'is_per_tensor', 'is_per_channel', 'getattr_from_fqn', 'get_qparam_dict', 'get_swapped_custom_module_class', 'activation_dtype', 'weight_dtype', 'activation_is_statically_quantized', 'activation_is_dynamically_quantized', 'activation_is_int8_quantized', 'activation_is_int32_quantized', 'weight_is_quantized', 'weight_is_statically_quantized', 'op_is_int8_dynamically_quantized', 'get_qconfig_dtypes', 'get_quant_type', 'check_min_max_valid', 'calculate_qmin_qmax', 'has_no_children_ignoring_parametrizations', 'get_fqn_to_example_inputs', 'to_underlying_dtype', 'determine_qparams', 'validate_qmin_qmax', 'DEPRECATION_WARNING']

NodePattern = tuple[Node, Node] | tuple[Node, tuple[Node, Node]] | Any
QuantizerCls = Any
Pattern = Callable | tuple[Callable, Callable] | tuple[Callable, tuple[Callable, Callable]] | Any

class MatchAllNode:
    """A node pattern that matches all nodes, used in defining
    fusion patterns in FX Graph Mode Quantization
    """

def check_node(node, modules): ...
def get_combined_dict(default_dict, additional_dict):
    """
    Combines two dictionaries.

    This function takes two dictionaries as input and returns a new dictionary
    that contains all the key-value pairs from both input dictionaries.
    If there are any duplicate keys in the `additional_dict`, the values
    from the `additional_dict` will overwrite those in the `default_dict`.
    Args:
        default_dict (dict): The main dictionary that will be used as the base
        additional_dict (dict): The dictionary used to update `default_dict`

    Returns:
        dict: The resulting dictionary
    Example:
        >>> x = dict(a=1, b=1)
        >>> y = dict(b=2, c=3)
        >>> get_combined_dict(x, y)
        {'a': 1, 'b': 2, 'c': 3}
    """
def is_per_tensor(qscheme): ...
def is_per_channel(qscheme): ...
def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    '''
    Given an obj and a fqn such as "foo.bar.baz", returns gm.foo.bar.baz.
    '''
def to_underlying_dtype(qdtype): ...
def get_qparam_dict(observer_or_fake_quant): ...
def get_swapped_custom_module_class(custom_module, custom_module_class_mapping, qconfig):
    """Get the observed/quantized custom module class that we need
    to swap `custom_module` to
    Input:
        custom_module: input, can be an instance of either a float or observed custom module
        custom_module_class_mapping: the float to observed or observed to quantized custom module class mapping
        qconfig: qconfig configured for the custom module

    Output:
        corresponding observed/quantized custom module class for input custom module instance
    """
def activation_dtype(qconfig): ...
def weight_dtype(qconfig): ...
def activation_is_statically_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized or not, this includes quantizing to quint8, qint8 and qint32 and float16
    """
def activation_is_dynamically_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    dynamically quantized or not, this includes dynamically quantizing to
    quint8, qint8 and float16
    """
def activation_is_int8_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized to int8 or not, this includes quantizing to quint8, qint8
    """
def activation_is_int32_quantized(qconfig):
    """Given a qconfig, decide if the activation needs to be
    quantized to int32 or not
    """
def weight_is_quantized(qconfig):
    """Given a qconfig, decide if the weight needs to be
    quantized or not
    """
def weight_is_statically_quantized(qconfig):
    """Given a qconfig, decide if the weight needs to be statically
    quantized or not
    """
def op_is_int8_dynamically_quantized(qconfig) -> bool:
    """Given a qconfig, returns True if this op is using int8 dynamic
    quantization
    """
def get_qconfig_dtypes(qconfig):
    """returns the qconfig tuple for qconfig:
    (activation_dtype, weight_dtype, activation_is_dynamic)
    """
def get_quant_type(qconfig): ...
def check_min_max_valid(min_val: torch.Tensor, max_val: torch.Tensor) -> bool:
    """Checks if the given minimum and maximum values are valid, meaning that
    they exist and the min value is less than the max value.
    """
def calculate_qmin_qmax(quant_min: int, quant_max: int, has_customized_qrange: bool, dtype: torch.dtype, reduce_range: bool) -> tuple[int, int]:
    """Calculates actual qmin and qmax based on the quantization range,
    observer datatype and if range is reduced.
    """
def has_no_children_ignoring_parametrizations(module):
    """
    Checks if module._modules is empty or
    if module is a parametrization, checks that module._modules only has
    the 'parametrizations' module
    """
def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    """Validates that the user-specified quantization range is properly initialized
    and within the given bound supported by the observer dtype.

    To accommodate lower-bit quantization with respect to the existing torch.qint8 and
    torch.quint8 datatypes, the user can choose to use dynamic quantization range by passing
    in a tuple of initial qmin and qmax values. One use case is these customized qmin and qmax
    values are used to calculate static estimates of the scale and zero point for aggressive lower-bit
    fake quantization. These estimates are compared against parameters learned through backpropagation.
    The related literatures for scale and zero point via backpropagation are as follows:

    Learned Step Size Quantization: https://openreview.net/pdf?id=rkgO66VKDS
    Trained Quantization Thresholds: https://arxiv.org/pdf/1903.08066.pdf
    """
def determine_qparams(min_val: torch.Tensor, max_val: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype, eps: torch.Tensor, has_customized_qrange: bool, qscheme: torch.qscheme = ...) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculates the quantization parameters, given min and max
    value tensors. Works for both per tensor and per channel cases

    Args:
        min_val: Minimum values per channel
        max_val: Maximum values per channel

    Returns:
        scales: Scales tensor of shape (#channels,)
        zero_points: Zero points tensor of shape (#channels,)
    """
def get_fqn_to_example_inputs(model: torch.nn.Module, example_inputs: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    '''Given a model and its example inputs, return a dictionary from
    fully qualified name of submodules to example_inputs for that submodule,
    e.g. {"linear1": (tensor1,), "linear2": (tensor2,), "sub": (tensor3,),
          "sub.linear1": (tensor4,), ...}

    Used to make quantizing submodules easier now that FX Graph Mode Quantization requires
    example inputs.

    Also works for keyword arguments with default values, we would flatten keyword
    arguments as positional arguments and fill in the missing keyword args with default
    values, e.g. if we have a forward function:
    def forward(self, x, key1=3, key2=3):
        ...

    and we call it with self.submodule(x, key2=6)
    we\'ll get example_inputs: (x, 3, 6)

    user can also override `key1` with positional arguments as well:
    for self.submodule(x, 5, key2=6)
    we\'ll get: (x, 5, 6)

    variable positional arguments and variable positional keyword arguments in forward
    function are not supported currently, so please make sure no submodules is using
    them.
    '''

DEPRECATION_WARNING: str
