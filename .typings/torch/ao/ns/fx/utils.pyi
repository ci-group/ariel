import enum
import torch
from .ns_types import NSNodeTargetType as NSNodeTargetType, NSResultsType as NSResultsType
from _typeshed import Incomplete
from torch.ao.quantization import FakeQuantizeBase as FakeQuantizeBase, ObserverBase as ObserverBase
from torch.ao.quantization.observer import _is_activation_post_process as _is_activation_post_process
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node
from typing import Callable

toq: Incomplete

class NodeInputOrOutputType(enum.Enum):
    FP32 = ...
    INT8 = ...
    FP16 = ...
    UNKNOWN = ...
    FP32_OR_INT8 = ...

def get_node_first_input_and_output_type(node: Node, gm: GraphModule, logger_cls: Callable, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]]) -> tuple[NodeInputOrOutputType, NodeInputOrOutputType]: ...
def get_node_input_qparams(node: Node, gm: GraphModule, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]]) -> tuple[torch.Tensor | float, torch.Tensor | int] | None:
    """
    Returns the qparams (scale, zero_point) of the first input to `node`,
    if they can be inferred from the graph.
    """
def return_first_non_observer_node(node: Node, gm: GraphModule) -> Node:
    """
    If node is not an observer, returns it.  If node is an observer,
    navigates up the graph and returns the first parent which is not an
    observer.  For example,

    graph: (node_non_obs), node = node_non_obs : returns node_non_obs
    graph: (node_non_obs -> obs0), node = obs0 : returns node_non_obs
    graph: (node_non_obs -> obs0 -> fq0), node = fq0 : returns node_non_obs
    """
def get_number_of_non_param_args(node: Node, gm: GraphModule) -> int:
    """
    Assumes that all non-param args occur first. Returns the number of
    non-param args expected for a node.  For example, for

      F.linear(x, weight, bias)

    Returns 1, because x is a non-param arg and weight and bias are params.
    For

      lstm_mod(x, hid)

    Returns 2, because both x and hid are non-param args.
    """
def get_arg_indices_of_inputs_to_log(node: Node) -> list[int]:
    """
    Returns the indices of args of the node which we should attach
    loggers to, if input logging is enabled.

    For example,
    * for (x + y), returns [0, 1]
    * for (1 + y), returns [1]
    * for (x + 1), returns [0]
    * for (linear(x, w, b)) returns [0]
    * by default, returns [0]
    """
def get_target_type_str(node: Node, gm: GraphModule) -> str:
    """
    Returns a string representation of the type of the function or module
    pointed to by this node, or '' for other node types.
    """
def rekey_logger_info_on_node_name_of_model(results: NSResultsType, model_name: str) -> NSResultsType:
    """
    Rekeys the layer name of a results dictionary to use node names
    from `model_name`.

    For example, transforms

        {'base_op_1_0': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    into

        {'linear1': {'node_output': {'model_a':
          [{'ref_node_name': 'linear1', ...}]}}}

    Note: we cannot use these node names directly because they are not
    guaranteed to be consistent across models. This is why we extract
    the results first and rekey afterwards.
    """
def maybe_add_missing_fqns(results: NSResultsType) -> None:
    """
    If `fqn` entries are filled in for one of the models in `results`, copies
    them over to any models which do not have them filled out.

    A common use case benefitting from this is comparing a model prepared by
    quantization to a quantized model. In this case, the model prepared by
    quantization would have `fqn` entries, and the quantized model would not.
    """
def maybe_dequantize_first_two_tensor_args_and_handle_tuples(f): ...
@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_sqnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the SQNR between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_normalized_l2_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the normalized L2 error between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
@maybe_dequantize_first_two_tensor_args_and_handle_tuples
def compute_cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Computes the cosine similarity between `x` and `y`.

    Args:
        x: Tensor or tuple of tensors
        y: Tensor or tuple of tensors

    Return:
        float or tuple of floats
    """
def op_type_supports_shadowing(node: Node) -> bool: ...
def get_normalized_nth_input(node: Node, gm: GraphModule, idx: int) -> Node:
    """
    Given a node, gets the n'th input to that node, normalizing
    args and kwargs to the best of its ability.
    """
