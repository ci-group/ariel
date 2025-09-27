from torch.fx import GraphModule, Node

__all__ = ['fold_bn_weights_into_conv_node', 'remove_tensor_overload_for_qdq_ops']

def fold_bn_weights_into_conv_node(conv_node: Node, conv_weight_node: Node, conv_bias_node: Node | None, bn_node: Node, m: GraphModule) -> None: ...
def remove_tensor_overload_for_qdq_ops(match_pattern: GraphModule) -> None:
    """Remove .tensor overload for quantize/dequantize ops so that we can
    use the match_pattern that we get from torchdynamo export to match the output of convert_pt2e
    """
