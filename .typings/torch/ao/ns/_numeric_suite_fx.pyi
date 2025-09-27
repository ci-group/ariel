import torch
import torch.ao.quantization.quantize_fx as quantize_fx
import torch.nn as nn
from .fx.graph_passes import add_loggers_to_model as add_loggers_to_model, create_a_shadows_b as create_a_shadows_b
from .fx.ns_types import NSNodeTargetType as NSNodeTargetType, NSResultsType as NSResultsType, NSSingleResultValuesType as NSSingleResultValuesType
from .fx.utils import get_target_type_str as get_target_type_str, maybe_add_missing_fqns as maybe_add_missing_fqns, rekey_logger_info_on_node_name_of_model as rekey_logger_info_on_node_name_of_model
from .fx.weight_utils import extract_weight_from_node as extract_weight_from_node
from _typeshed import Incomplete
from torch.ao.ns.fx.graph_matcher import get_matching_subgraph_pairs as get_matching_subgraph_pairs
from torch.ao.ns.fx.mappings import get_base_name_to_sets_of_related_ops as get_base_name_to_sets_of_related_ops
from torch.ao.ns.fx.n_shadows_utils import OutputProp as OutputProp, SHADOW_WRAPPER_NODE_NAME_PREFIX as SHADOW_WRAPPER_NODE_NAME_PREFIX, _get_dedup_subgraphs as _get_dedup_subgraphs, create_add_loggers_graph as create_add_loggers_graph, create_n_transformed_and_logged_copies_of_subgraph as create_n_transformed_and_logged_copies_of_subgraph, create_results_comparison as create_results_comparison, extract_weight_comparison as extract_weight_comparison, group_results_by_subgraph as group_results_by_subgraph, print_n_shadows_summary as print_n_shadows_summary
from torch.ao.ns.fx.qconfig_multi_mapping import QConfigMultiMapping as QConfigMultiMapping
from torch.ao.quantization import QConfigMapping as QConfigMapping
from torch.ao.quantization.backend_config import BackendConfig as BackendConfig
from torch.ao.quantization.backend_config.utils import get_fusion_pattern_to_root_node_getter as get_fusion_pattern_to_root_node_getter
from torch.ao.quantization.fx.graph_module import _get_observed_graph_module_attr as _get_observed_graph_module_attr
from torch.ao.quantization.fx.match_utils import _find_matches as _find_matches
from torch.ao.quantization.fx.qconfig_mapping_utils import _generate_node_name_to_qconfig as _generate_node_name_to_qconfig
from torch.ao.quantization.fx.quantize_handler import _get_pattern_to_quantize_handlers as _get_pattern_to_quantize_handlers
from torch.ao.quantization.qconfig import QConfigAny as QConfigAny
from torch.fx import GraphModule as GraphModule
from torch.fx.graph import Node as Node
from typing import Any, Callable

RNNReturnType = tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]

class OutputLogger(nn.Module):
    """
    Base class for capturing intermediate values.
    """
    stats: list[torch.Tensor]
    stats_rnn: list[RNNReturnType]
    _is_impure: bool
    ref_node_name: Incomplete
    prev_node_name: Incomplete
    model_name: Incomplete
    ref_name: Incomplete
    prev_node_target_type: Incomplete
    ref_node_target_type: Incomplete
    results_type: Incomplete
    index_within_arg: Incomplete
    index_of_arg: Incomplete
    fqn: Incomplete
    enabled: bool
    qconfig_str: Incomplete
    save_activations: bool
    def __init__(self, ref_node_name: str, prev_node_name: str, model_name: str, ref_name: str, prev_node_target_type: str, ref_node_target_type: str, results_type: str, index_within_arg: int, index_of_arg: int, fqn: str | None, qconfig_str: str | None = '') -> None: ...
    def forward(self, x):
        """
        """
    def __repr__(self) -> str: ...

class OutputComparisonLogger(OutputLogger):
    """
    Same as OutputLogger, but also requires the original activation
    in order to calculate the comparison at calibration time
    """
    comparison_fn: Incomplete
    comparison_fn_name: str
    comparisons: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    def forward(self, x, x_ref):
        """
        """
    def __repr__(self) -> str: ...

class NSTracer(quantize_fx.QuantizationTracer):
    """
    Just like a regular FX quantization tracer, but treats observers and fake_quantize
    modules as leaf modules.
    """
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        """
        """

def _extract_weights_one_model(model_name: str, model: GraphModule, nodes_and_names_to_instrument: list[tuple[Node, str]], results: NSResultsType, op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]] | None = None) -> None: ...
def _extract_weights_impl(model_name_a: str, gm_a: GraphModule, model_name_b: str, gm_b: GraphModule, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None, op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]] | None = None) -> NSResultsType: ...
def extract_weights(model_name_a: str, model_a: nn.Module, model_name_b: str, model_b: nn.Module, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None, op_to_type_to_weight_extraction_fn: dict[str, dict[Callable, Callable]] | None = None) -> NSResultsType:
    """
    Extract weights from model A and model B, and return a comparison.

    Args:
        model_name_a: string name of model A to use in results
        model_a: model A
        model_name_b: string name of model B to use in results
        model_b: model B
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
        op_to_type_to_weight_extraction_fn: optional override of function which extracts weight
            from a type, subject to change

    Return:
        NSResultsType, containing the weight comparisons
    """
def _add_loggers_one_model(model_name: str, model: GraphModule, nodes_and_names_to_instrument_inputs: list[tuple[Node, str, str]], nodes_and_names_to_instrument_outputs: list[tuple[Node, str, str]], logger_cls: Callable) -> nn.Module: ...
def _add_loggers_impl(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, logger_cls: Callable, should_log_inputs: bool, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None) -> tuple[nn.Module, nn.Module]: ...
def add_loggers(name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module, logger_cls: Callable, should_log_inputs: bool = False, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None) -> tuple[nn.Module, nn.Module]:
    """
    Instrument model A and model B with loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change

    Return:
        Returns a tuple of (model_a_with_loggers, model_b_with_loggers).  Modifies both models inplace.
    """
def _extract_logger_info_one_model(model: nn.Module, results: NSResultsType, logger_cls: Callable) -> None: ...
def extract_logger_info(model_a: nn.Module, model_b: nn.Module, logger_cls: Callable, model_name_to_use_for_layer_names: str) -> NSResultsType:
    """
    Traverse all loggers in `model_a` and `model_b`, and extract the logged
    information.

    Args:
        model_a: model A
        model_b: model B
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
def _add_shadow_loggers_impl(name_a: str, gm_a: GraphModule, name_b: str, gm_b: GraphModule, logger_cls: Callable, should_log_inputs: bool, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None) -> nn.Module: ...
def add_shadow_loggers(name_a: str, model_a: nn.Module, name_b: str, model_b: nn.Module, logger_cls: Callable, should_log_inputs: bool = False, base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]] | None = None, node_type_to_io_type_map: dict[str, set[NSNodeTargetType]] | None = None, unmatchable_types_map: dict[str, set[NSNodeTargetType]] | None = None) -> nn.Module:
    """
    Instrument model A and model B with shadow loggers.

    Args:
        name_a: string name of model A to use in results
        model_a: model A
        name_b: string name of model B to use in results
        model_b: model B
        logger_cls: class of Logger to use
        should_log_inputs: whether to log inputs
        base_name_to_sets_of_related_ops: optional override of subgraph base nodes, subject to change
        unmatchable_types_map: optional override of unmatchable types, subject to change
    """
def extract_shadow_logger_info(model_a_shadows_b: nn.Module, logger_cls: Callable, model_name_to_use_for_layer_names: str) -> NSResultsType:
    """
    Traverse all loggers in a shadow model, and extract the logged
    information.

    Args:
        model_a_shadows_b: shadow model
        logger_cls: class of Logger to use
        model_name_to_use_for_layer_names: string name of model to use for
          layer names in the output

    Return:
        NSResultsType, containing the logged comparisons
    """
def extend_logger_results_with_comparison(results: NSResultsType, model_name_1: str, model_name_2: str, comparison_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], comparison_name: str) -> None:
    """
    Compares the logged values from `model_name_2` against the corresponding
    values in `model_name_1`, using `comparison_fn`. Records the result
    in `model_name_2`'s results under `comparison_name`. Modifies `results` inplace.

    Args:
        results: the result data structure from `extract_logger_info` or
          `extract_shadow_logger_info`.
        model_name_1: string name of model 1
        model_name_2: string name of model 2
        comparison_fn: function to compare two Tensors
        comparison_name: string name of model to use for
          layer names in the output
    """
def prepare_n_shadows_model(model: torch.nn.Module, example_inputs: Any, qconfig_multi_mapping: QConfigMultiMapping, backend_config: BackendConfig, custom_prepare_fn: Callable | None = None, custom_prepare_kwargs: dict[str, Any] | None = None, custom_tracer: Any = None) -> GraphModule:
    """
    Given a model with a graph with M ops such as


      args_kwargs_m -> op_m -> output_m


    And a set of N qconfigs for each op, creates a new model, with
    each of the subgraph of `op_m` transformed into

    .. code::

           |---------> op_m_n -> log_m_n
           |                     /
      args_kwargs_m ---------> op_m -> log_m_0

    Where op_m_n is op_m wrapped in a submodule and transformed with
    qconfig_n, and its inner graph looks like

    .. code::

      args_m -------- op_m_prepared_with_qconfig_n -> out_m_n
                  /
      kwargs_m ---

    This is useful for testing different quantization of multiple layers in
    a single pass through the model.

    High level TODOs for future PRs:
    * figure out a better way to name the output structure
    * return a results data structure instead of printing it out
    * add examples to docblocks
    """
def _prepare_n_shadows_add_loggers_model(model: torch.nn.Module, example_inputs: Any, qconfig_mapping: QConfigMapping, backend_config: BackendConfig) -> torch.nn.Module:
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.

    This creates a model which provides logging for the following
    problem: if we quantize `model` with `qconfig_mapping` and feed
    the same input through both models, log the comparisons of
    corresponding intermediate layers.

    The problem is solved with a single model.  Specifically, we
    partition `model` into N subgraphs, create a copy of each relevant
    subgraph, wrap it in a module, apply the quantization API to that
    module, and hook up loggers to measure the comparisons.

    Example starting graph:

      x0 -> op0 -> x1 -> op1 -> x2

    Example config: quantize op0 to int8, do nothing to op1.
    The following graph will be created:

    .. code::

      x0_0 -> op0_0 -> x1_0 -> log -----> op1_0 -> x2_0 -> log
       \\                        \\                           \\       # noqa: W605
         ---> op0_1 -> x1_1 ----> clog -> op1_0 -> x2_1 ----> clog

    Where op0_0 is op0, op0_1 is op0 wrapped in a submodule and quantized
    to int8, op1_0 is op1 (appearing in the graph twice), log is a logger,
    and clog is a comparison logger.
    """
def _n_shadows_compare_weights(model: torch.nn.Module, example_inputs: Any, qconfig_mapping: QConfigMapping, backend_config: BackendConfig) -> NSResultsType:
    """
    Note: this API is not recommended for wide usage, it is only
    provided for customers who need to migrate from the `add_loggers`
    API.
    """
def loggers_set_enabled(model: torch.nn.Module, enabled: bool) -> None:
    """
    Sets the `enabled` setting on a `model`'s loggers
    """
def loggers_set_save_activations(model: torch.nn.Module, save_activations: bool) -> None:
    """
    Sets the `save_activations` setting on a `model`'s loggers
    """
def convert_n_shadows_model(model: GraphModule, custom_convert_fn: Callable | None = None, custom_convert_kwargs: dict[str, Any] | None = None) -> GraphModule:
    """
    Given a model from `prepare_n_shadows_model`, runs `convert_fx`
    on each shadow submodule.
    """
def extract_results_n_shadows_model(model: torch.nn.Module) -> NSResultsType:
    """
    Extracts logger results from `model`.
    """
def print_comparisons_n_shadows_model(results: NSResultsType) -> None:
    """
    Prints a summary of extracted `results`.
    """
