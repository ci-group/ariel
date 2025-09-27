from .utils import _get_aten_graph_module_for_pattern as _get_aten_graph_module_for_pattern, _is_bn_node as _is_bn_node, _is_conv_or_conv_transpose_node as _is_conv_or_conv_transpose_node, _is_conv_transpose_fn as _is_conv_transpose_fn, fold_bn_weights_into_conv_node as fold_bn_weights_into_conv_node
from _typeshed import Incomplete
from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib as quantized_decomposed_lib
from torch.ao.quantization.pt2e.export_utils import _WrapperModule as _WrapperModule
from torch.ao.quantization.quantizer import DerivedQuantizationSpec as DerivedQuantizationSpec, EdgeOrNode as EdgeOrNode, QuantizationSpecBase as QuantizationSpecBase, SharedQuantizationSpec as SharedQuantizationSpec
from torch.fx import Graph as Graph, GraphModule as GraphModule, Node as Node
from torch.fx.passes.utils.matcher_with_name_node_map_utils import InternalMatch as InternalMatch
from torch.fx.subgraph_rewriter import ReplacedPatterns as ReplacedPatterns, replace_pattern_with_filters as replace_pattern_with_filters
from typing import Any, Callable

__all__: Incomplete

def _get_quantized_conv_bn_example_inputs_kwargs(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool, is_cuda: bool) -> dict[str, Any]:
    """
    Optional example inputs for quantized and folded conv-bn patterns
    used in convert, expressed as kwargs.
    """
def _get_conv_bn_pattern(conv_fn: Callable) -> Callable: ...
def _get_qat_conv_bn_pattern(conv_fn: Callable) -> Callable: ...
def _get_qat_conv_bn_pattern_no_conv_bias(conv_fn: Callable) -> Callable: ...
def _append_qdq(x, is_per_channel, is_bias, kwargs):
    """
    Helper function to append q-dq ops after `x`, using dummy values for the qparams
    and qmin/qmax. We use dummy values here because we match with `ignore_literals=True`
    and will manually replace these values after subgraph rewriting.

    Return the dq node.
    """
def _get_quantized_qat_conv_bn_pattern(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool, conv_fn: Callable, bn_is_training: bool) -> Callable:
    """
    Return the quantized version of QAT conv + BN pattern.
    This is based on `nniqat.ConvBn2d._forward_approximate`,
    used in QAT convert. We first match this pattern and replace
    it with the normal [conv - bn] pattern, then fold the BN
    weights into conv.
    """
def _get_folded_quantized_qat_conv_bn_pattern(is_per_channel: bool, has_bias: bool, bias_is_quantized: bool, conv_fn: Callable, bn_is_training: bool) -> Callable:
    """
    Quantized QAT conv - bn pattern with bn weights being folded into conv.
    """
def _has_conv_bias_filter(match: InternalMatch, original_graph: Graph, pattern_graph: Graph) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph has bias.
    """
def _no_conv_bias_filter(match: InternalMatch, original_graph: Graph, pattern_graph: Graph) -> bool:
    """
    Match filter for the subgraph rewriter that returns True if the conv node in
    the original graph does NOT have bias.
    """
def _is_quantize(n: Node) -> bool: ...
def _is_dequantize(n: Node) -> bool: ...
def _get_conv_bn_pattern_nodes(r: ReplacedPatterns) -> dict[str, tuple[Node, Node]]:
    '''
    Helper function to extract the nodes in the conv-bn fusion pattern after
    subgraph rewriting, in the form of a map:

        {name: (original_node, replacement_node)}

    The following names must exist in the map:

        "conv", "conv_weight", "conv_input", "bn", "getitem"

    The following names may exist in the map:

        "conv_weight_q", "conv_weight_dq", "conv_bias",
        "conv_bias_q", "conv_bias_dq"
    '''
def _filter_nodes_map(nodes_map: dict[Node, Node]) -> dict[Node, Node]:
    """
    Return a filtered `nodes_map` returned from the subgraph rewriter.
    The filtered `nodes_map` will contain only nodes that are actually
    matched in the pattern, excluding None or placeholder nodes.
    """
def _copy_over_literal_conv_args(original_node: Node, new_node: Node):
    """
    Copy over literal args in conv, such as stride and padding, from the matched node
    in the original graph to its replacement in the new graph.

    This is needed due to the following limitation in the subgraph rewriter when used
    with dynamo export: literal (non-tensor) args are not supported in the match and
    replacement patterns. This is because dynamo export automatically inlines these
    literal args, making them dead placeholder nodes. In the future, we should check
    if dynamo export can optionally disable this inlining, or if subgraph rewriter
    can do the copying for us. See https://github.com/pytorch/pytorch/issues/100419.

    Note: Unlike other tensor args like conv weights and biases, literal args are
    preserved in the original nodes after replacement, so we can access them here.
    """
def _update_conv_input_qspec_map_after_replacement(original_node: Node, replacement_node: Node):
    """
    Update the `input_qspec_map` in the annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the keys in the `input_qspec_map` will need to be updated to reflect
    the corresponding nodes in the replacement graph.
    """
def _update_special_qspecs_after_replacement(node: Node, original_to_replacement_node: dict[Node, Node]):
    """
    Update the `SharedQuantizationSpec`s and `DerivedQuantizationSpec`s
    used in `node`'s quantization annotation after subgraph rewriting.

    The original annotation referred to the nodes in the original graph,
    so the nodes used in these special quantization specs will need to
    be updated to the corresponding nodes in the replacement graph.
    """
def _fuse_conv_bn_qat(m: GraphModule) -> GraphModule: ...
def _fuse_conv_bn_qat_helper(m: GraphModule, conv_fn: Callable, example_inputs: tuple[Any, ...], is_cuda: bool) -> GraphModule:
    """
    Given a graph of decomposed aten ops, replace the (conv + bn) pattern with
    the fused QAT subgraph equivalent. The input graph should already be annotated.
    The annotations in the original nodes will be preserved in the corresponding
    nodes in the new subgraph.

    Note: This also handles the (conv + bn + relu) pattern.
    """
def _duplicate_dequantize_node(m: GraphModule):
    """
    Helper function to duplicate all dequantize nodes in the graph if the
    node has more than one user. For example:

    Before:
      quantize -> dequantize -> a
                          \\--> b
                          \\--> c

    After:
      quantize -> dequantize_1 -> a
            \\--> dequantize_2 -> b
            \\--> dequantize_3 -> c

    This is useful for subgraph rewriting. E.g. if we wish to match the
    pattern [dequantize - a] above, subgraph matching would fail because
    the dequantize node has users outside the matched portion of the graph.
    Instead, we match [dequantize_1 - a], which is safe.
    """
def _remove_extra_dequantize(m: GraphModule):
    '''
    Removes duplicate dequant nodes in the graph, for an operator that has
    multiple dequant nodes as a user, replace them with a single dequant node
    that can be shared across all the uses. This should be seen as the "reverse"
    of `_duplicate_dequantize_node`.
    '''
def _copy_over_q_dq_args(original_node: Node, replacement_node: Node):
    """
    Given a pair of quantize or dequantize nodes, copy over all literal args
    from the original node to the replacement node.
    """
def _fold_conv_bn_qat(m: GraphModule) -> GraphModule: ...
def _fold_conv_bn_qat_helper(m: GraphModule, conv_fn: Callable, example_inputs: tuple[Any, ...], is_cuda: bool) -> GraphModule:
    """
    Replace the quantized (conv + bn) pattern with conv with bn weights folded into the weights of conv.
    """
