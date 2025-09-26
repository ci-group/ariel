import torch
from _typeshed import Incomplete
from torch.ao.ns.fx.graph_passes import _maybe_get_fqn as _maybe_get_fqn
from torch.ao.ns.fx.ns_types import NSResultsType as NSResultsType, NSSingleResultValuesType as NSSingleResultValuesType
from torch.ao.ns.fx.utils import get_normalized_nth_input as get_normalized_nth_input, get_target_type_str as get_target_type_str
from torch.ao.quantization import QConfigMapping as QConfigMapping
from torch.ao.quantization.fx.match_utils import _MatchResult as _MatchResult
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.ao.quantization.utils import getattr_from_fqn as getattr_from_fqn
from torch.fx import Graph as Graph, GraphModule as GraphModule, Node as Node
from torch.utils._pytree import tree_map as tree_map
from typing import Any, Callable

SHADOW_NODE_NAME_PREFIX: str
SHADOW_WRAPPER_NODE_NAME_PREFIX: str
BINARY_FUNCTIONS: Incomplete

def _get_attr_name(subgraph_idx, subgraph_candidate_idx): ...
def _get_attr_wrapper_name(subgraph_idx, subgraph_candidate_idx): ...

class OutputProp:
    """
    Output propagation (modeled from shape propagation).

    Given a GraphModule and an example input, saves the output flowing
    through each node on `node.traced_result`.

    Code based on the example from
    https://pytorch.org/docs/stable/fx.html#the-interpreter-pattern
    """
    mod: Incomplete
    graph: Incomplete
    modules: Incomplete
    def __init__(self, mod) -> None: ...
    def propagate(self, *args): ...

def _get_dedup_subgraphs(matches: dict[str, _MatchResult]) -> dict[str, list[Node]]: ...
def _get_logger_for_subgraph(model: GraphModule, first_node: Node, last_node: Node, subgraph_idx: int, subgraph_candidate_idx: int, qconfig_str: str, logger_cls: Callable, fqn: str | None) -> torch.nn.Module:
    """
    Given a model and a linear subgraph starting from `first_node` and
    ending with `last_node`, creates a logger for the end of this
    subgraph.
    """
def create_submodule_from_subgraph(model: torch.nn.Module, first_node: Node, last_node: Node) -> GraphModule:
    """
    Input: a model, and a linear subgraph within the model from first_node to
      last_node.

    Output: a new submodule containing a copy of the subgraph, with the inputs
      to the first node becoming the inputs to the submodule, and all other
      nodes in the subgraph being copied.

    Example inputs:

    `model`: a module with graph

      x0 -> op1 -> x1 -> op2 -> x2
             |
            arg1

    `first_node`: op1
    `last_node`: op2

    Example output: a new module with graph

      input1 -> op1_copy -> x1 -> op2_copy -> output1
                   |
                  arg1
    """
def create_one_transformed_and_logged_copy_of_subgraph(mt: GraphModule, subgraph_idx: int, subgraph_candidate_idx: int, first_node: Node, last_node: Node, fqn: str | None, list_of_node_name_to_qconfig: list[dict[str, QConfigAny]], example_inputs: Any, last_added_shadow_node_list: list[Node | None], custom_prepare_fn: Callable | None = None, custom_prepare_kwargs: dict[str, Any] | None = None) -> None:
    """
    Given a subgraph in `mt` and a subgraph candidate idx, inserts the
    subgraph candidate copy and instruments it with loggers.

    If subgraph_candidate_idx is 0, this is the baseline fp32 subgraph and we just
    add a logger to the end.

    If subgraph_candidate_idx is not 0, we create a copy of the subgraph and
    prepare it with `prepare_fx`.
    """
def create_n_transformed_and_logged_copies_of_subgraph(mt: GraphModule, subgraph_idx: int, match_name: str, nodes_in_this_subgraph: list[Any], qconfig_mappings: list[QConfigMapping], list_of_node_name_to_qconfig: list[dict[str, QConfigAny]], custom_prepare_fn: Callable | None = None, custom_prepare_kwargs: dict[str, Any] | None = None) -> None:
    """
    Given a model `mt` and a subgraph_idx, creates the needed copies
    of the subgraph for all qconfigs, and instruments them with loggers.
    """
def create_add_loggers_graph(model: GraphModule, subgraphs_dedup: dict[str, list[Node]], qconfig_mapping: QConfigMapping, node_name_to_qconfig: dict[str, QConfigAny]) -> None:
    """
    Given a model, a model graph partition (currently a set of matched
    subgraphs) and instructions how to transform each subgraph
    (currently quantizing it according to qconfig_mapping), modifies
    the model graph to create an alternate path through the original graph,
    with each of the subgraphs quantized.  This is useful to compare
    propagation error of a transformation such as quantization.

    For example, given layer op0 and op1, there are four cases when handling op1:
    1. op0 and op1 quantized
    2. op0 and op1 unquantized
    3. op0 quantized, op1 unquantized
    4. op0 unquantized, op1 quantized

    Example input, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\          \\                 \\       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog    op1_1 -> x2_1 ----> clog

    Example output, case 1:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\                           \\        # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_1 -> x2_1 ----> clog

    """
def _get_weight_info_from_shadow_wrapper(shadow_wrapper: torch.nn.Module): ...
def extract_weight_comparison(m: GraphModule) -> NSResultsType: ...
def group_results_by_subgraph(results: NSResultsType) -> Any:
    """
    Creates a comparison of results

    Input:

    {
      'model': {
        'node_output': {
          'subgraph_0_0': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [], ...
            'comparison_fn_name': '',
            'fqn': '...',
          ],
          'subgraph_0_1': [
            'values': [torch.tensor(...), ...], ...
            'ref_node_name': ...,
            'ref_node_target_type': ...,
            'qconfig_str': ...,
            'comparisons': [torch.tensor(...), ...], ...
            'comparison_fn_name': '...',
            'fqn': '...',
          ],
          ...
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': None,
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...], ...
          'comparison_fn_name': '...',
          'fqn': '...',
        },
      },
    }

    """
def create_results_comparison(results_grouped) -> Any:
    """
    Input:

    {
      'subgraph_0': {
        '0': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '',
          'comparisons': [],
          'comparison_fn_name': '',
          'fqn': '...',
        },
        '1': {
          'ref_node_name': '...',
          'ref_node_target_type': ...,
          'values': [torch.tensor(...), ...],
          'qconfig_str': '...',
          'comparisons': [torch.tensor(...), ...],
          'comparison_fn_name': 'sqnr',
          'fqn': '...',
        },
      },
    }

    Output:
    {
      'subgraph_0': {
        'ref_node_name': '...',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': 'sqnr',
            'cmp_raw': [..., ...],
            'cmp_mean': ...,
          },
          ...,
        },
      },
    }
    """
def print_n_shadows_summary(results_comparison) -> None:
    """
    Input:

    {
      'subgraph_0': {
        'ref_node_name': 'linear1',
        'ref_node_target_type': '...',
        'fqn': '...',
        'candidates': {
          '1': {
            'qconfig_str': ...,
            'comparison_fn_name': ...,
            'cmp_raw': [45.0, 55.0],
            'cmp_mean': 50.0,
          },
          ...,
        },
      },
    }

    Prints:

    node_name | node_type | fqn | 0    | 1    | ...
    linear1   | ...       | ... | 45.0 | 50.0 | ...
    """
