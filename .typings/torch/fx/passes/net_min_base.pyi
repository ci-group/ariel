import torch.fx
from .tools_common import Names, NodeList, NodeSet, TensorOrTensors, Tensors
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ['FxNetMinimizerBadModuleError', 'FxNetMinimizerRunFuncError', 'FxNetMinimizerResultMismatchError']

class FxNetMinimizerBadModuleError(Exception):
    """
    Raised if failed to split out a minimize module
    """
class FxNetMinimizerRunFuncError(Exception):
    """
    Raised if error occurs during run_a or run_b functions
    """
class FxNetMinimizerResultMismatchError(Exception):
    """
    Raised if comparing function thinks the results are mismatching.
    """

@dataclass
class _MinimizerSettingBase:
    '''
    Args:
    `accumulate_error`: Instead of using a\'s input for both converted module to verify
    , use the previous outputs of each converted module as input to accumulate the
    errors.

    `traverse_method`: "sequential" or "binary" or "accumulate"
    Determine the way of traverse the nodes in FX module.

    `find_all`: Minimizer will go through the entire model and return all problematic nodes.

    `return_intermediate`: If true, when using `run_nodes()` function to run the
    model, intermediate results of all the ops will be returned as output.

    `all_outputs`: If true, when using `_run_and_compare()` function,
    all the output nodes in the subgraph will be used for comparison.
    '''
    accumulate_error: bool = ...
    traverse_method: str = ...
    find_all: bool = ...
    return_intermediate: bool = ...
    all_outputs: bool = ...
    def __str__(self) -> str: ...

class _MinimizerBase:
    """
    This class is used to automatically find problematic nodes in a model. It takes a FX
    graphmodule and generate some submodules while traverse the graph. Then two functions
    `run_a` and `run_b` will be used to run the same submodule and a function `compare_fn`
    will be used to compare the results.

    Currently we provides two ways to traverse the graph and generate submodules.
        1. Sequential traversal: this will traverse the graph node by node and generate
           one submodule with one sigle node.
        2. Binary searching: this will do a binary search style traversal on the graph.

    For internal Users, a guide can be found here https://fb.quip.com/HDtuAgiKGfkP.
    """
    module: Incomplete
    sample_input: Incomplete
    compare_fn: Incomplete
    module_exporter: Incomplete
    settings: Incomplete
    exclusion_fn: Incomplete
    a_outputs: dict[str, Any]
    b_outputs: dict[str, Any]
    results: dict[Any, Any]
    reports: list[list[str]]
    iteration: int
    fusions: Incomplete
    def __init__(self, module: torch.fx.GraphModule, sample_input: Tensors, compare_fn: Callable[[TensorOrTensors, TensorOrTensors, Names], tuple[float, bool]], settings: _MinimizerSettingBase, module_exporter: Callable[[Tensors, torch.fx.GraphModule, str], None] | None = None, exclusion_fn: Callable[[NodeList, int, int], None] | None = None) -> None: ...
    def run_shape_prop(self) -> None:
        """
        Helper function to run shape propagation on module. Can be overridden by
        subclasses for custom shape propagation logic.
        """
    def run_a(self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_b().
        """
    def run_b(self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_a().
        """
    def _store_outputs(self, a_result: TensorOrTensors, b_result: TensorOrTensors, submodule: torch.fx.GraphModule):
        """
        Store the outputs of self.run_a() and self.run_b() into self.a_outputs and
        self.b_outputs, so that we can use them when execute preceding nodes that
        use those outputs as inputs.

        Args:
            a_result: Output of self.run_a(). Could be a tensor or tensors.
            b_result: Output of self.run_b(). Could be a tensor or tensors.
            submodule: The module that generates a_result and b_result.
        """
    def _get_submod_inputs(self, main_module: torch.fx.GraphModule, submod_path: str) -> tuple[Tensors, Tensors]:
        """
        Try get submodule inputs from stored outputs. If not found then use
        torch_glow.get_submod_inputs to get the inputs.

        If accumulate_error is False, use a_input for run_a() and run_b()
        otherwise use a_input for run_a and b_input for run_b.

        Args:
            main_module: Top-levlel fx module.
            submod_path: Path to the submodule we want to run and compare results.

        Returns:
            a_input: List of tensor(s) that will be used by run_a() as submodule inputs.
            b_input: List of tensor(s) that will be used by run_b() as submodule inputs.
        """
    def _tag_nodes(self, selected_nodes: NodeSet):
        '''
        Tag selected nodes with tag "minimize". Nodes with the same tags will
        be split to the same submodule afterwards.

        Args:
            selected_nodes: Nodes that we want to minimize. We will tag those nodes
                with "minimize", all preceding nodes with "main_0" and all following
                nodes with "main_1".
        '''
    def _build_submodule(self, nodes: NodeSet) -> tuple[torch.fx.GraphModule, str]:
        """
        Split self.module so that one submodule consists of `nodes` and only `nodes`.

        Args:
            nodes: Nodes that we want to include in the minimize submodule.

        Returns:
            split_module (torch.fx.GraphModule): the module after split.
            submodule_name (str): the name of the submodule that consists of `nodes`.
        """
    def _run_and_compare(self, split_module: torch.fx.GraphModule, submod_name: str, output_names: Names, report_idx: int = -1):
        """
        Run the submodule in `split_module` that has name `submod_name`
        using `self.run_a` and `self.run_b` and compare their results.

        Args:
            split_module: Main module that contains the minimize submodule.
            submod_name: Name of the minimize submodule.
            output_names: Names of the node we want to output. If None, we
                will use the original output.
        """
    def _binary_search_impl(self, all_nodes: NodeList, start_idx: int, end_idx: int) -> NodeSet:
        """
        Recursive binary search implementation.
        """
    def _binary_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Binary search on `nodes` for culprit.
        """
    def _sequential_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Traverse `nodes` one by one and determine if any of them is a culprit.
        """
    def _block_traverse_impl(self, nodes: NodeList, start_idx: int, end_idx: int, find_last_node: bool) -> int | None:
        """
        Recursive block search implementation.
        find_last_node: If True, search for the last node which result in numerics difference
        if False: find first node in sorted node list
        """
    def _block_traverse(self, nodes: NodeList, find_last_node: bool | None) -> NodeSet:
        """
        Traverse topologically sorted node list
        Find minimium block (start_idx, end_idx) which contains the culprit
        1st pass: search for end_idx by finding the last node in culprit block
        where Numerical accuracy (0, end_idx) > threshold
        2nd pass: search for start_idx by finding the first node in culprit block
        where Numerical accuracy (start_idx, end_idx) < threshold
        Form minimum block by (start_idx - 1, end_idx)
        """
    def _defined_traverse(self, nodes: NodeList) -> NodeSet:
        """
        run user defined `nodes` and determine if it is a culprit.
        """
    def _accumulate_traverse(self, nodes: NodeList) -> NodeSet: ...
    def _skip_traverse_impl(self, all_nodes: NodeList, start_idx: int, end_idx: int) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
    def _skip_traverse(self, all_nodes: NodeList, skip_nodes: list) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
    def _collect_nodes(self, start: str | None, end: str | None) -> NodeList:
        """
        Collect nodes in the model that between nodes with name of `start` and `end`.
        These two nodes are also included.
        """
    def run_nodes(self, start: str | None = None, end: str | None = None):
        """
        Run part of the model from `start` node to `end` node. If `start` is None
        then we start from the beginning of the model. If `end` is None then we
        stop at the end of the model.

        Args:
            start: The name of the node which is the first node of the submodule
                we want to run. If set to None, then we'll start with the first
                node of the model.
            end: The name of the node which is the last node of the submodule we
                want to run. If set to None, we'll end with the last node of the
                model.
        """
    def print_report(self, report: list[str]): ...
    def print_reports(self) -> None: ...
    def minimize(self, start: str | None = None, end: str | None = None, skip_nodes: list | None = None, find_last_node: bool | None = None) -> NodeSet:
        '''
        Minimizing the model from node with name `start` to node with name `end` base
        on self.settings. Find culprits that causes FxNetMinimizerRunFuncError or
        FxNetMinimizerResultMismatchError errors.

        Args:
            start: The name of the node where we want to start minimizing. If set
                to None, then we\'ll start with the first node of the model.
            end: The name of the node where we want to terminate minimizing. If
                set to None, we\'ll end with the last node of the model.
            skip_nodes: The names of nodes where we want to skip during minimizing.
                It\'ll create subgraphs without these skip nodes under the hood.
                Only applicable in mode "skip".
            find_last_node: True if only last_node of a culprits is needed in mode "block".
                False if only the first_node of a culprits is needed.
                Only applicable in mode "block".

        Returns:
            nodes: A list of nodes that causes FxNetMinimizerRunFuncError or
                FxNetMinimizerResultMismatchError errors during minimizing.
        '''
