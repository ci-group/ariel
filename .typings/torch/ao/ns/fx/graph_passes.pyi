import torch
from .ns_types import NSNodeTargetType as NSNodeTargetType, NSSingleResultValuesType as NSSingleResultValuesType, NSSubgraph as NSSubgraph
from .utils import NodeInputOrOutputType as NodeInputOrOutputType, get_arg_indices_of_inputs_to_log as get_arg_indices_of_inputs_to_log, get_node_first_input_and_output_type as get_node_first_input_and_output_type, get_node_input_qparams as get_node_input_qparams, get_normalized_nth_input as get_normalized_nth_input, get_number_of_non_param_args as get_number_of_non_param_args, get_target_type_str as get_target_type_str, getattr_from_fqn as getattr_from_fqn, op_type_supports_shadowing as op_type_supports_shadowing, return_first_non_observer_node as return_first_non_observer_node
from torch.ao.ns.fx.mappings import get_node_type_to_io_type_map as get_node_type_to_io_type_map
from torch.ao.quantization.fx.utils import get_new_attr_name_with_prefix as get_new_attr_name_with_prefix
from torch.ao.quantization.observer import _is_activation_post_process as _is_activation_post_process
from torch.fx import GraphModule as GraphModule, map_arg as map_arg
from torch.fx.graph import Graph as Graph, Node as Node
from typing import Callable

def _maybe_get_fqn(node: Node, gm: GraphModule) -> str | None: ...
def _insert_logger_after_node(node: Node, gm: GraphModule, logger_cls: Callable, logger_node_name_suffix: str, ref_node_name: str, model_name: str, ref_name: str, ref_node_target_type: str, results_type: str, index_within_arg: int, index_of_arg: int, fqn: str | None) -> Node:
    """
    Given a starting graph of

    prev_node -> node -> next_node

    This function creates a new logger_cls obj and adds it
    after node, resulting in

    prev_node -> node -> logger_obj -> next_node
    """
def add_loggers_to_model(gm: GraphModule, node_to_instrument_inputs_to_ref_node_name: dict[Node, tuple[str, str]], node_to_instrument_outputs_to_ref_node_name: dict[Node, tuple[str, str]], logger_cls: Callable, model_name: str) -> GraphModule:
    """
    Takes the graph of gm, adds loggers to the output
    of each node in nodes_to_instrument. Returns a GraphModule with the new
    graph.
    """
def _insert_quantize_per_tensor_node(prev_node_c: Node, node_a: Node, gm_b: GraphModule, graph_c: Graph, scale: torch.Tensor | float, zero_point: torch.Tensor | int, dtype_cast_name: str) -> Node: ...
def _insert_dtype_cast_after_node(node_a: Node, node_c: Node, prev_node_c: Node | list[Node], gm_a: GraphModule, gm_b: GraphModule, graph_c: Graph, node_name_prefix: str, logger_cls: Callable, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]]) -> Node | list[Node]:
    """
    Given a starting graph C (derived from graph B) of

    ... -> prev_node_c -> node_c -> ...

    And a corresponding related node_a, inserts the correct dtype
    cast node after prev_node_c to cast into the dtype expected
    by node_a, resulting in:

                          dtype_cast
                        /
    ... -> prev_node_c -> node_c -> ...

    For example, if node_c is an int8 op and node_a is an fp32 op, this function
    will insert a dequant.
    """
def _copy_node_from_a_to_c(node_a: Node, gm_a: GraphModule, gm_b: GraphModule, graph_c: Graph) -> Node:
    """
    Simple copy of node_a to graph_c.
    """
def _can_insert_copy_of_subgraph_a(subgraph_a: NSSubgraph, gm_a: GraphModule, num_non_param_args_node_a: int) -> bool:
    """
    This function returns `False` if the input subgraph cannot be copied by
    `_insert_copy_of_subgraph_a_after_input_node_c`. This usually means
    that there is a corner case logic for which copy is not yet implemented.
    """
def _insert_copy_of_subgraph_a_after_input_node_c(input_node_c: Node | list[Node], input_node_c_2: Node | list[Node] | None, subgraph_a: NSSubgraph, gm_a: GraphModule, gm_b: GraphModule, node_name_prefix: str) -> Node:
    """
    TODO(before land): real docblock
    """
def _insert_copy_of_node_a_after_input_node_c(input_node_c: Node | list[Node], input_node_c_2: Node | list[Node] | None, node_a: Node, gm_a: GraphModule, gm_b: GraphModule, node_name_prefix: str) -> Node:
    """
    Assume that node_a from graph_a has
      args (input, (input2)?, arg1, ...), and
      kwargs {kw0: kwarg0, ...}

    Note: input2 is optional. If it equals to None, we assume that the op
    has a single non-param input.  If it is specified, we assume that the op
    has two non-param inputs.

    Copies the underlying values of arg1..argn and kwarg0..kwargn into gm_b,
    and creates the corresponding nodes in graph_c. Note: observers are ignored,
    so if an arg is an observer we navigate up until we find a non-observer parent.

    If node_a is a call_module, points the module pointed to by node_a to gm_b.

    Creates the copy of node_a in graph_c, with input as the first arg,
    and all other args and kwargs pointing to the copies of the objects
    in gm_b created above.

    An example in pictures:

    graph A:
    ========

    input -------------> node_a
                         / / /
    (input_2)?----------/ / /
                         / /
    weight -> weight_obs  /
                         /
    bias ----------------

    graph C (derived from B):
    =========================

    input_node_c --> node_a_copy
                     / / /
    (input_node_c_2)? / /
                     / /
    weight_copy ----/ /
                     /
    bias_copy ------/
    """
def create_a_shadows_b(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, matched_subgraph_pairs: dict[str, tuple[NSSubgraph, NSSubgraph]], logger_cls: Callable, should_log_inputs: bool, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None) -> GraphModule:
    """
    Creates a new GraphModule consisting of the graph of C, with the meaningful
    nodes of A shadowing the corresponding nodes of B.  For example,

    Graph A:
    a0 -> op0_fp32 -> a1 -> op1_fp32 -> a2

    Graph B:
    b0 -> op0_int8 -> b1 -> op1_int8 -> b2

    matched_node_pairs: {'op0': (op0_fp32, op0_int8), 'op1': (op1_fp32, op1_int8)}

    Graph C (A shadows B):

        / dequant0 -> op0_fp32 -> logger_a_0  / dequant_1 -> op1_fp32 -> logger_a_1
       /                                     /
    b0 -------------> op0_int8 -> logger_b_0 --------------> op1_int8 -> logger_b_1

    In a nutshell, this function does the following for each node pair:
    * copies the necessary attributes and modules from gm_a to gm_b,
      keeping names unique
    * adds a dtype cast op (dequant, quant, etc)
    * adds a copy of node_a in gm_b's graph
    * adds loggers to the outputs of node_a and node_b
    """
