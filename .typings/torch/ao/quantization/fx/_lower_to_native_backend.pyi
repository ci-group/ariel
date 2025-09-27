import torch
import torch.nn as nn
from .utils import collect_producer_nodes as collect_producer_nodes, create_node_from_old_node_preserve_meta as create_node_from_old_node_preserve_meta, get_linear_prepack_op_for_dtype as get_linear_prepack_op_for_dtype, get_new_attr_name_with_prefix as get_new_attr_name_with_prefix, get_qconv_prepack_op as get_qconv_prepack_op, graph_module_from_producer_nodes as graph_module_from_producer_nodes
from _typeshed import Incomplete
from torch.ao.nn.quantized.modules.utils import WeightedQuantizedModule as WeightedQuantizedModule
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.ao.quantization.quantization_mappings import get_quantized_operator as get_quantized_operator
from torch.ao.quantization.utils import _parent_name as _parent_name
from torch.fx import GraphModule as GraphModule, Node as Node, map_arg as map_arg
from torch.fx.graph import Graph as Graph
from typing import Any, Callable

QOP_TO_ARG_NAMES_TO_SKIP: dict[Callable[..., Any], list[str]]

def _is_node_in_list(node, modules, func_list, method_list, module_type_list): ...
def is_fixed_qparams_node(node, modules): ...
def is_default_node(node, modules): ...
def is_copy_node(node, modules): ...
def is_general_tensor_shape_node(node, modules): ...
def is_other_node(node, modules): ...
def is_special_pattern_node(node, modules): ...
def is_dequantize_node(node): ...
def is_getattr_tensor_metadata_node(node): ...
def is_get_tensor_info_node(node): ...
def should_skip_lowering(op: torch.fx.node.Node, qconfig_map: dict[str, QConfigAny]):
    """
    Return True if the op is configured with a None qconfig, False otherwise.
    Note: maybe need to generalize this to also check for the dtype, and we
    only lower when dtype matches, but right now fbgemm/qnnpack only support
    a single dtype, so it is OK for now.
    """

STATIC_LOWER_MODULE_MAP: dict[type[nn.Module], type[WeightedQuantizedModule]]
DYNAMIC_LOWER_MODULE_MAP: dict[type[nn.Module], type[nn.Module]]
WEIGHT_ONLY_LOWER_MODULE_MAP: dict[type[nn.Module], type[nn.Module]]
SPECIAL_PATTERN_LOWER_MODULE_MAP: Incomplete
STATIC_LOWER_FUSED_MODULE_MAP: dict[type[nn.Module], tuple[type[nn.Module], type[WeightedQuantizedModule]]]
STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP: dict[type[nn.Module], tuple[type[nn.Module], type[WeightedQuantizedModule]]]
DYNAMIC_LOWER_FUSED_MODULE_MAP: dict[type[nn.Module], tuple[type[nn.Module], type[nn.Module]]]
STATIC_LOWER_FUNCTIONAL_MAP: dict[Callable, tuple[Callable, Callable | None]]
WEIGHT_PREPACK_OPS: set[Callable]
DYNAMIC_LOWER_FUNCTIONAL_MAP: dict[Callable, dict[tuple[torch.dtype, torch.dtype], tuple[Callable, Callable | None]]]
CONV_FUNCTIONAL_OPS: set[Callable]
CONV_TRANSPOSE_FUNCTIONAL_OPS: set[Callable]
QBIN_OP_MAPPING: dict[Callable | str, Callable]
QBIN_RELU_OP_MAPPING: dict[Callable | str, Callable]
ORIGINAL_WEIGHTS_LOOKUP: str

def _save_packed_weight(self, destination, prefix, keep_vars) -> None: ...
def _load_packed_weight(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None: ...
def fold_weight(quantized_model: GraphModule, node_name_to_scope: dict[str, tuple[str, type]], keep_original_weights: bool = False) -> GraphModule:
    """
    Trace back from the weight node util we hit getattr, reconstruct the
    graph module with the traced nodes and run the graph module to pack the
    weight. then replace the original chain of ops with the packed weight.
    """
def _get_module(node: Node, modules: dict[str, nn.Module]) -> nn.Module | None:
    """
    Return the `torch.nn.Module` that corresponds to the specified node's target.
    If no such node exists, return None.
    """
def _match_static_pattern(node: Node, modules: dict[str, nn.Module], qconfig_map: dict[str, QConfigAny], matching_modules_or_ops: list[Callable], dequantize_node_arg_indices: list[int]) -> tuple[Node, Node, Node] | tuple[None, None, None]:
    """
    Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 3-tuple of:
      1) q_node: the quantize node,
      2) relu_node: a relu node wrapping the ref_node, and
      3) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 3-tuple of (None, None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
      dequantize_node_arg_indices: A list of indices in the reference node args where dequantize
          nodes may be present. An empty list means skipping the check for dequantize nodes.
    """
def _match_static_pattern_with_two_inputs(node: Node, modules: dict[str, nn.Module], qconfig_map: dict[str, QConfigAny], matching_modules_or_ops: list[Callable]) -> tuple[Node, Node] | tuple[None, None]:
    """
                      (dequantize     Match the pattern (dequantize - ref node - quantize) against the node provided.

    If there is a match, return a 2-tuple of:
      1) q_node: the quantize node,
      2) ref_node: a reference module or functional node to replace with its quantized counterpart
    Otherwise, if there is no match, return a 2-tuple of (None, None).

    Parameters:
      node: The `torch.fx.Node` to match against.
      modules: A mapping from node names to modules in the model graph, used for module lookup.
      qconfig_map: A mapping from node names to the qconfigs associated with the nodes.
          If the corresponding qconfig for the reference node is None, then return no match.
      matching_modules_or_ops: Either a list of functions or a list of `torch.nn.Module`s.
          If the reference node is not in this list, then return no match.
    """
def _lower_static_weighted_ref_module(model: GraphModule, qconfig_map: dict[str, QConfigAny]):
    """
    Traverse the graph and find dequantize - ref module - quantize patterns
    and replace them with the quantized version of the ref module.
    """
def _lower_static_weighted_ref_module_with_two_inputs(model: GraphModule, qconfig_map: dict[str, QConfigAny]):
    """
    Traverse the graph and find patterns
    dequantize   dequantize
       \\         //
        ref module
            \\\n          quantize
    and replace them with the quantized version of the ref module.
    """
def _lower_dynamic_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find quantize_per_tensor_dynamic - dequantize - ref_module patterns
    and replace them with the dynamically quantized version of the ref module.
    """
def _lower_weight_only_weighted_ref_module(model: GraphModule):
    """
    Traverse the graph and find ref_module patterns
    and replace them with the weight only quantized version of the ref module.
    """
def _lower_static_weighted_ref_functional(model: GraphModule, qconfig_map: dict[str, QConfigAny]):
    """
    Traverse the graph and replace functional reference patterns with their quantized versions.
    """
def _lower_dynamic_weighted_ref_functional(model: GraphModule, qconfig_map: dict[str, QConfigAny]):
    """
    Traverse the graph and replace functional reference patterns with their dynamically
    quantized versions.
    Examples:
    quantize_per_tensor_dynamic - dequantize - functional linear --> linear_dynamic
    to(torch.float16) - dequantize - functional linear --> linear_dynamic_fp16
    """
def _lower_quantized_binary_op(model: GraphModule, qconfig_map: dict[str, QConfigAny]): ...
def special_pattern_replacement(model: GraphModule): ...
def _lower_getattr_tensor_metadta_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
def _lower_get_tensor_info_op(model: GraphModule):
    """Modified the graph of the model inplace, to skip extra dequantize op before
    the general tensor shape ops when possible
    """
def _lower_to_native_backend(model: GraphModule, qconfig_map: dict[str, QConfigAny], node_name_to_scope: dict[str, tuple[str, type]], keep_original_weights: bool = False) -> GraphModule:
    """Lower a quantized reference model (with reference quantized operator patterns)
    to the native backend in PyTorch (fbgemm/qnnpack), both backends shares the same
    operator signature so they can be lowered with the same function
    """
