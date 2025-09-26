import torch
import torch.nn as nn
from .custom_config import PrepareCustomConfig
from _typeshed import Incomplete
from dataclasses import dataclass
from torch.ao.quantization import QConfigAny, QuantType
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from typing import Any, Callable, NamedTuple

__all__ = ['all_node_args_except_first', 'all_node_args_have_no_tensors', 'assert_and_get_unique_device', 'collect_producer_nodes', 'create_getattr_from_value', 'create_node_from_old_node_preserve_meta', 'EMPTY_ARG_DICT', 'get_custom_module_class_keys', 'get_linear_prepack_op_for_dtype', 'get_new_attr_name_with_prefix', 'get_non_observable_arg_indexes_and_types', 'get_qconv_prepack_op', 'get_skipped_module_name_and_classes', 'graph_module_from_producer_nodes', 'maybe_get_next_module', 'NodeInfo', 'node_arg_is_bias', 'node_arg_is_weight', 'NON_OBSERVABLE_ARG_DICT', 'NON_QUANTIZABLE_WEIGHT_OPS', 'return_arg_list', 'ObservedGraphModuleAttrs']

NON_QUANTIZABLE_WEIGHT_OPS: Incomplete

@dataclass
class ObservedGraphModuleAttrs:
    node_name_to_qconfig: dict[str, QConfigAny]
    node_name_to_scope: dict[str, tuple[str, type]]
    prepare_custom_config: PrepareCustomConfig
    equalization_node_name_to_qconfig: dict[str, Any]
    qconfig_mapping: QConfigMapping
    is_qat: bool
    observed_node_names: set[str]
    is_observed_standalone_module: bool = ...
    standalone_module_input_quantized_idxs: list[int] | None = ...
    standalone_module_output_quantized_idxs: list[int] | None = ...

def node_arg_is_weight(node: Node, arg: Any) -> bool:
    """Returns if node arg is weight"""
def node_arg_is_bias(node: Node, arg: Any) -> bool:
    """Returns if node arg is bias"""
def get_custom_module_class_keys(custom_module_mapping: dict[QuantType, dict[type, type]]) -> list[Any]:
    """Get all the unique custom module keys in the custom config dict
    e.g.
    Input:
    {
        QuantType.STATIC: {
            CustomModule1: ObservedCustomModule
        },
        QuantType.DYNAMIC: {
            CustomModule2: DynamicObservedCustomModule
        },
        QuantType.WEIGHT_ONLY: {
            CustomModule3: WeightOnlyObservedCustomModule
        },
    }

    Output:
    # extract the keys across all inner STATIC, DYNAMIC, and WEIGHT_ONLY dicts
    [CustomModule1, CustomModule2, CustomModule3]
    """
def get_linear_prepack_op_for_dtype(dtype): ...
def get_qconv_prepack_op(conv_op: Callable) -> Callable: ...
def get_new_attr_name_with_prefix(prefix: str) -> Callable: ...
def collect_producer_nodes(node: Node) -> list[Node] | None:
    """Starting from a target node, trace back until we hit inpu or
    getattr node. This is used to extract the chain of operators
    starting from getattr to the target node, for example
    def forward(self, x):
      observed = self.observer(self.weight)
      return F.linear(x, observed)
    collect_producer_nodes(observed) will either return a list of nodes that
    produces the observed node or None if we can't extract a self contained
    graph without free variables(inputs of the forward function).
    """
def graph_module_from_producer_nodes(root: GraphModule, producer_nodes: list[Node]) -> GraphModule:
    """Construct a graph module from extracted producer nodes
    from `collect_producer_nodes` function
    Args:
      root: the root module for the original graph
      producer_nodes: a list of nodes we use to construct the graph
    Return:
      A graph module constructed from the producer nodes
    """
def assert_and_get_unique_device(module: torch.nn.Module) -> Any:
    """
    Returns the unique device for a module, or None if no device is found.
    Throws an error if multiple devices are detected.
    """
def create_getattr_from_value(module: torch.nn.Module, graph: Graph, prefix: str, value: Any) -> Node:
    """
    Given a value of any type, creates a getattr node corresponding to the value and
    registers the value as a buffer to the module.
    """
def all_node_args_have_no_tensors(node: Node, modules: dict[str, torch.nn.Module], cache: dict[Node, bool]) -> bool:
    """
    If we know for sure that all of this node's args have no
    tensors (are primitives), return True.  If we either
    find a tensor or are not sure, return False. Note: this
    function is not exact.
    """
def all_node_args_except_first(node: Node) -> list[int]:
    """
    Returns all node arg indices after first
    """
def return_arg_list(arg_indices: list[int]) -> Callable[[Node], list[int]]:
    """
    Constructs a function that takes a node as arg and returns the arg_indices
    that are valid for node.args
    """

class NodeInfo(NamedTuple):
    op: Incomplete
    target: Incomplete

NON_OBSERVABLE_ARG_DICT: dict[NodeInfo, dict[type | torch.dtype, Callable[[Node], list[int]]]]
EMPTY_ARG_DICT: dict[type | torch.dtype, Callable[[Node], list[int]]]

def get_non_observable_arg_indexes_and_types(node: Node) -> dict[type | torch.dtype, Callable[[Node], list[int]]]:
    """
    Returns a dict with of non float tensor types as keys and values which correspond to a
    function to retrieve the list (which takes the node as an argument)
    """
def maybe_get_next_module(node: Node, modules: dict[str, nn.Module], target_module_type: type[nn.Module] | None = None, target_functional_type: Any = None) -> Node | None:
    """Gets the next module that matches what is needed in
    is_target_module_type if it exists

    Args:
        node: The node whose users we want to look at
        target_module_type: Module type that we want to check
        target_functional_type: Functional type that we want to check
    """
def create_node_from_old_node_preserve_meta(quantized_graph: Graph, create_node_args: tuple[Any, ...], old_node: Node) -> Node:
    """
    Creates `new_node` and copies the necessary metadata to it from `old_node`.
    """
def get_skipped_module_name_and_classes(prepare_custom_config: PrepareCustomConfig, is_standalone_module: bool) -> tuple[list[str], list[type[Any]]]: ...

# Names in __all__ with no definition:
#   NodeInfo
