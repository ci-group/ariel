from torch.ao.quantization.pt2e.utils import _is_sym_size_node as _is_sym_size_node
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation as QuantizationAnnotation
from torch.fx import Node as Node

def _annotate_input_qspec_map(node: Node, input_node: Node, qspec): ...
def _annotate_output_qspec(node: Node, qspec): ...
def _node_only_used_for_sym_size(node: Node, partition_nodes: list[Node]):
    """
    This utility is used to handle cases when dynami_shape=True tracing leads
    to symint nodes in the pattern of linear module. In those cases, we need to
    distinguish between the nodes that are in input for just extracting value of
    some dimentions (and symint nodes) vs. the one that is activation.
    For example:
    graph(x, y, weight):
       size_0 = torch.ops.aten.sym_size([x], [0])
       size_1 = torch.ops.aten.sym_size([y], [1])
       view_size = size_0 * size_1
       size_3 = torch.ops.aten.sym_size([x], [2])
       vie_out = torch.ops.aten.view(x, [view_size, size_3])
       return mm(view_out, weight)
    In the example above y node is not actual input. It exist only to extract size_1
    """
def _get_module_name_filter(module_name: str):
    '''Get the module_name_filter function for a given module name, the filter accepts
    a node and checks if the node comes from a module that has certain module name

    For example:
        node: linear_op = call_function[...](...)  # comes from a module with name blocks.sub.linear1


    >> module_name_filter = _get_module_name_filter("blocks.sub")
    >> print(module_name_filter(node))
    True  # the node is from "blocks.sub" based on the fully qualified name "blocks.sub.linear1"
    '''
