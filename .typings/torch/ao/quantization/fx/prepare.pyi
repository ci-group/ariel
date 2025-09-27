from .custom_config import PrepareCustomConfig
from .match_utils import _MatchResultWithQConfig
from torch.ao.quantization.backend_config import BackendConfig
from torch.ao.quantization.qconfig import QConfigAny
from torch.ao.quantization.qconfig_mapping import QConfigMapping
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from typing import Any

__all__ = ['insert_observers_for_model', 'prepare', 'propagate_dtypes_for_known_nodes']

def propagate_dtypes_for_known_nodes(graph: Graph, node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig]) -> None:
    """
    Currently we assume that inputs to the graph are either `torch.float` or
    `torch.quint8`, which is not always correct. For ops such as
    `x.masked_fill(mask, value)`, we know that the dtype of  `mask` is a
    `BoolTensor`. Propagate this information throughout the graph.

    Note: not all dtypes in the graph will be correct after this pass, but a
    higher percentage of them will be correct. Hopefully in the future we can
    replace this with a better way to reason about dtypes of tensors.
    """
def insert_observers_for_model(model: GraphModule, node_name_to_match_result_with_qconfig: dict[str, _MatchResultWithQConfig], node_name_to_qconfig: dict[str, QConfigAny], prepare_custom_config: PrepareCustomConfig, equalization_config_map: dict[str, Any], backend_config: BackendConfig, observed_node_names: set[str], is_qat: bool) -> Node | None:
    """
    Inserts observers, using the following high level algorithm:

    For each node in the graph:
      1. determine the target dtype of this node in the quantized graph, and save
           it for future steps
      2. determine the target dtype or all args and kwargs of this node
      3. if any arg or kwarg's target dtype does not match the current node's
           dtype, insert an observer
      4. if the current node needs an output observer, insert it

    For example:

    - starting graph:
        x0 -> linear -> x1

    - observed graph after processing x0:
        x0(fp32)

    - observed graph after processing linear:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8)

    - observed graph after processing x1:
        x0(fp32) -> x0_obs0(int8) -> linear(int8) -> linear_obs0(int8) -> x1

    After a node is processed, the naive observer placement is guaranteed to be
    complete for that node and all of its predecessors. There can be future
    passes which optimize the graph by deduplicating observers, etc.
    """
def prepare(model: GraphModule, qconfig_mapping: QConfigMapping | dict[str, Any], is_qat: bool, node_name_to_scope: dict[str, tuple[str, type]], example_inputs: tuple[Any, ...], prepare_custom_config: PrepareCustomConfig | dict[str, Any] | None = None, _equalization_config: QConfigMapping | dict[str, Any] | None = None, backend_config: BackendConfig | dict[str, Any] | None = None, is_standalone_module: bool = False) -> GraphModule:
    '''standalone_module means it a submodule that is not inlined in
    parent module, and will be quantized separately as one unit.

    How the standalone module is observed is specified by `input_quantized_idxs` and
    `output_quantized_idxs` in the prepare_custom_config for the standalone module
    Args:
        node_name_to_scope: mapping from node name to the scope of the module which contains the node.
        The scope is a tuple of fully qualified path of the module and the type of the module
    Returns:
        model(GraphModule): prepared standalone module
        attributes related to standalone module
        in model.meta["_observed_graph_module_attrs"]:
            is_observed_standalone_module (bool): boolean value that shows whether the
            current model is a observed standalone module or not
            standalone_module_input_quantized_idxs(List[Int]): a list of
                indexes for the graph input that is expected to be quantized,
                same as input_quantized_idxs configuration provided
                for the standalone module
            standalone_module_output_quantized_idxs(List[Int]): a list of
                indexs for the graph output that is quantized
                same as input_quantized_idxs configuration provided
                for the standalone module
    '''
