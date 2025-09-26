import torch
import torch.nn as nn
from .utils import get_new_attr_name_with_prefix as get_new_attr_name_with_prefix, maybe_get_next_module as maybe_get_next_module, node_arg_is_weight as node_arg_is_weight
from _typeshed import Incomplete
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr as _get_observed_graph_module_attr
from torch.ao.quantization.observer import ObserverBase as ObserverBase, PerChannelMinMaxObserver as PerChannelMinMaxObserver, _with_args as _with_args
from torch.ao.quantization.utils import _parent_name as _parent_name, check_min_max_valid as check_min_max_valid
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node
from typing import Any, NamedTuple

CUSTOM_MODULE_SUPP_LIST: list[Any]

def reshape_scale(scale: torch.Tensor, axis: int, input: torch.Tensor) -> torch.Tensor:
    """Reshapes the scale so that we can multiply it to the input by the given axis."""

qsheme_mapping_per_tensor_to_per_channel: Incomplete

class _InputEqualizationObserver(nn.Module):
    """Observer for tracking the running min/max values of input columns, and
    computing the quantization parameters for the overall min/max input values.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    The running minimum/maximum :math:`x_\\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`,
    with the difference that the running min/max values are stored per column.
    This observer is intended to be used along with a WeightEqualizationObserver
    to calculate the equalization scale.
    """
    dtype: Incomplete
    qscheme: Incomplete
    input_obs: Incomplete
    equalization_scale: Incomplete
    equalization_shape: list[int]
    def __init__(self, dtype=..., qscheme=..., quant_min=None, quant_max=None, factory_kwargs=None) -> None: ...
    def forward(self, x_orig): ...
    def get_input_minmax(self): ...
    def set_equalization_scale(self, equalization_scale) -> None: ...
    def calculate_scaled_minmax(self):
        """Returns the scaled min/max inputs"""
    with_args: Incomplete

class _WeightEqualizationObserver(nn.Module):
    """Observer for tracking the running min/max values of weight columns and
    rows, and computing the quantization parameters for the weight rows.

    Args:
        dtype: Quantized data type
        qscheme: Quantization scheme
        quant_min: Minimum quantization value. If unspecified, it will
            follow the 8-bit setup.
        quant_max: Maximum quantization value. If unspecified, it will
            follow the 8-bit setup.

    This observer is made up of 1 PerChannelMinMaxObserver `weight_col_obs` used
    to record the running minimum and maximum of columns of incoming weight
    tensors. This observer is intended to be used along with an
    InputEqualizationObserver to calculate the equalization scale.

    The running minimum/maximum :math:`w_\\text{min/max}` are computed in the
    same way as :class:`~torch.ao.quantization.observer.PerChannelMinMaxObserver`.
    """
    dtype: Incomplete
    qscheme: Incomplete
    ch_axis: int
    weight_col_obs: Incomplete
    equalization_scale: Incomplete
    def __init__(self, dtype=..., qscheme=..., quant_min=None, quant_max=None, factory_kwargs=None) -> None: ...
    def forward(self, w_orig): ...
    def get_weight_col_minmax(self): ...
    def set_equalization_scale(self, equalization_scale) -> None: ...
    with_args: Incomplete

def calculate_equalization_scale(input_obs: _InputEqualizationObserver, weight_obs: _WeightEqualizationObserver) -> torch.Tensor:
    """Calculates the equalization scale and sets the equalization_scale value
    in the observers.

    Args:
        input_obs: Observer that tracks the ranges for the input columns
        weight_obs: Observer that tracks the ranges for the weight columns
    """

class EqualizationQConfig(NamedTuple('EqualizationQConfig', [('input_activation', Incomplete), ('weight', Incomplete)])):
    """
    Describes how to quantize a layer or a part of the network specifically for
    input-weight equalization by providing settings (observer classes) for
    inputs, outputs, and weights.

    Note that EqualizationQConfig needs to contain observer **classes** (like
    MinMaxObserver) or a callable that returns instances on invocation, not the
    concrete observer instances themselves.
    Quantization function will instantiate observers multiple times for each of
    the layers.

    Observer classes have usually reasonable default arguments, but they can be
    overwritten with `with_args` method (that behaves like functools.partial):

    my_qconfig = EqualizationQConfig(input_activation=_InputEqualizationObserver.with_args(dtype=torch.qint8),
                                    weight=_WeightEqualizationObserver.with_args(dtype=torch.qint8))
    """
    __slots__: Incomplete
    def __new__(cls, input_activation=..., weight=...): ...

input_equalization_observer: Incomplete
weight_equalization_observer: Incomplete
default_equalization_qconfig: Incomplete

def fused_module_supports_equalization(module) -> bool:
    """Checks if the fused node supports equalization."""
def nn_module_supports_equalization(module) -> bool:
    """Checks if the torch.nn node supports equalization."""
def custom_module_supports_equalization(module) -> bool:
    """Checks if the custom node supports equalization."""
def node_supports_equalization(node: Node, modules) -> bool:
    """Checks if the current node supports equalization
    Currently we only support nn.Linear/F.Linear and nn.Conv/F.conv layers
    """
def is_equalization_observer(observer: nn.Module) -> bool: ...
def get_op_node_and_weight_eq_obs(input_eq_obs_node: Node, model: GraphModule, modules: dict[str, nn.Module]) -> tuple[Node | None, _WeightEqualizationObserver | None]:
    """Gets the following weight equalization observer. There should always
    exist a weight equalization observer after an input equalization observer.

    Returns the operation node that follows the input equalization observer node
    and the weight equalization observer
    """
def maybe_get_weight_eq_obs_node(op_node: Node, modules: dict[str, nn.Module]) -> Node | None:
    """Gets the weight equalization observer node if it exists."""
def maybe_get_next_input_eq_obs(node: Node, modules: dict[str, nn.Module]) -> _InputEqualizationObserver | None:
    """Gets the following input equalization observer if it exists.

    For example, in the case of connecting linear layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    If the node being passed in is the linear1 node, then we want to return eq_obs2,
    the following equalization observer for linear2.

    However, if there are no connecting layers:
        x -> inp_obs1 -> eq_obs1 -> linear1 -> out_obs1 -> add
    Then we want to return None.

    In the case of an unfused linear-relu layer with a connecting linear layer:
        linear1 -> relu -> out_obs1 -> eq_obs2 -> linear2 -> out_obs2
    Since it is unfused, we want to skip over the relu layer and return eq_obs2,
    the following equalization observer for linear2.
    """
def maybe_get_next_equalization_scale(node: Node, modules: dict[str, nn.Module]) -> torch.Tensor | None:
    """If the next next node is an InputEqualizationObserver then we want to
    return its equalization scale, else we return 1

    This is used in the case where there are two connecting linear layers:
        linear1 -> LinearOutObs -> InputEqObs -> linear2
    In this case, the node given is linear1 and we want to locate the InputEqObs.
    """
def scale_input_observer(node: Node, modules: dict[str, nn.Module]) -> None:
    """Scales the following input quantization observer's min/max values by
    updating the values with the scaled min/max values calculated by the input
    equalization observer
    """
def scale_weight_node(node: Node, modules: dict[str, nn.Module], equalization_scale: torch.Tensor, next_equalization_scale: torch.Tensor | None) -> None:
    """Scale the weights for input-weight equalization by multiplying the
    weight by 1/equalization_scale and next_equalization_scale

    Args:
        node: Current node whose weights we want to scale
        equalization_scale: Current node's calculated equalization scale
        next_equalization_scale: Next node's calculated equalization scale if
           the following node needs to be equalized, 1 otherwise
    """
def scale_weight_functional(op_node: Node, model: GraphModule, modules: dict[str, nn.Module], equalization_scale: torch.Tensor, next_equalization_scale: torch.Tensor | None) -> None:
    """Scales the weight value for functional layers"""
def clear_weight_quant_obs_node(op_node: Node, modules: dict[str, nn.Module]) -> None:
    """Given the operation node, we want find the corresponding quantization
    observer and reset its min/max values
    """
def remove_node(model: GraphModule, node: Node, prev_node: Node):
    """Removes the given node from the model by replacing all of its users with
    the given previous node
    """
def update_obs_for_equalization(model: GraphModule, modules: dict[str, nn.Module]) -> dict[str, _WeightEqualizationObserver]:
    """Update all of the observer's equalization scale. For each
    InputEqualizationObserver, we will find the location of the next
    WeightEqualizationObserver, create it, and calculate the equalization scale
    based on the two observers.

    We will then return a dictionary mapping operation node names to
    the corresponding WeightEqualizationObservers for that operation.
    """
def convert_eq_obs(model: GraphModule, modules: dict[str, nn.Module], weight_eq_obs_dict: dict[str, _WeightEqualizationObserver]) -> None:
    """Converts the equalization operations and updates the other nodes in the
    following way:
        - Removes the input equalization observers and inserts a mul operator
          along with an equalization scale node wherever applicable (we do not
          want to insert a mul operator between connecting linear layers).
        - Updates the input quantization observers with the scaled input min/max
          values.
        - Scales the weights by the current and next equalization scales.
        - Removes the weight equalization observer node if it exists.

    Before (after prepare):
                                    weight values
                                          |
                                    WeightQuantObs
                                          |
                                      WeightEqObs
                                          |
        x -> InpQuantObs -> InpEqObs -> linear -> OutQuantObs

    After this function:
                                              scaled weight values
                                                      |
       equalization scale                       WeightQuantObs
              |                                       |
        x -> mul -> InpQuantObs (scaled min/max) -> linear -> OutQuantObs

    After convert:
       equalization scale                 scaled weight values
              |                                    |
        x -> mul -> quantize_per_tensor -> quantized::linear

    Note that although the equalization observer appeared after the quantization
    observer after prepare_fx, the mul node appears before the quantization node
    after convert_fx. This is because placing the equalization observer after
    the quantization observer in prepare_fx would allow us to keep the invariant
    that the graph before the current node inserts its observers is not
    modified.

    Having the equalization observer before the quantization observer would also
    cause some inconsistences between the ordering of the quantization and
    equalization observers.
    For example, a single linear layer would look like:
        x -> InpEqObs1 -> InpQuantObs1 -> linear1 -> OutQuantObs1
    But between two connected linear layers, it would look like:
        linear1 -> OutQuantObs1 -> InpEqObs2 -> linear2 -> OutQuantObs2
    """
def _convert_equalization_ref(model: GraphModule):
    """Reference function which applies changes needed for equalization, but
    does not quantize the nodes
    """
def get_layer_sqnr_dict(model_a: nn.Module, model_b: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """Runs the Numeric Suite on model_a and model_b and returns a dictionary
    containing the SQNR between layers in model_a and model_b.

    Note: In order to support equalized models, this function has a hacky fix in
    which we do not match any torch.mul operators. This is because equalized
    models contain extra mul operators to scale the input by the equalization
    scale, but this edge case has not been resolved yet within the numeric suite code.

    Args:
        model_a: A float model
        model_b: A quantized model
        x: Inputs to use during calibration
    """
def get_equalization_qconfig_dict(layer_sqnr_dict: dict[str, float], num_layers_to_equalize: int) -> Any:
    """Given the layer to SQNR dictionary, find the layers with the highest
    quantization errors, and return an equalization_qconfig_dict
    specifying to only equalize those top layers.

    Args:
        layer_sqnr_dict: Dictionary mapping layer names to SQNR values (found
            when comparing an equalized model against a float model)
        num_layers_to_equalize: Number of layers with the highest quantization
           errors to equalize
    """
