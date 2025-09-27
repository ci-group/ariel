import torch
from .operator_support import OperatorSupportBase
from .tools_common import NodeList, NodeSet, Tensors
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

__all__ = ['FxNetAccNodesFinder', 'FxNetSplitterInternalError', 'Subgraph', 'SplitResult', 'generate_inputs_for_submodules']

class _SplitterSettingBase:
    min_acc_module_size: int
    skip_fusion: bool
    allow_non_tensor: bool
    max_acc_splits: int
    def __init__(self, min_acc_module_size=..., skip_fusion=..., allow_non_tensor=..., max_acc_splits: int = -1) -> None: ...

class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """
    module: Incomplete
    operator_support: Incomplete
    allow_non_tensor: Incomplete
    acc_nodes: NodeSet
    def __init__(self, module: torch.fx.GraphModule, operator_support: OperatorSupportBase, allow_non_tensor: bool) -> None: ...
    def reduce_acc_nodes_non_tensor_input_helper(self, cpu_worklist: NodeList):
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
    def reduce_acc_nodes_non_tensor_input(self) -> None:
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
    def reduce_acc_nodes_non_tensor_output(self) -> None:
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
    def __call__(self) -> NodeSet: ...

class FxNetSplitterInternalError(Exception): ...

@dataclass
class Subgraph:
    is_acc: bool
    nodes: NodeList
    device_ordinal: int | None = ...

class SplitResult(NamedTuple):
    '''
    Stores the results of the splitter.

    Attributes:
        split_module: root module after splitting.
        submodule_inputs: a dict that maps submodule name to its inputs.
        non_acc_submodule_prefix: the prefix for non acc submodules. For
            acc submodule the prefix is alwasy "_run_on_acc_".
    '''
    split_module: torch.fx.GraphModule
    submodule_inputs: dict[str, Any]
    non_acc_submodule_prefix: str

def generate_inputs_for_submodules(model: torch.nn.Module, inputs: Sequence[Any], target_submodules: Iterable[str], deepcopy: bool = False) -> dict[str, Any]:
    """
    Generate inputs for targeting submdoules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """

class _SplitterBase:
    '''
    Splits a GraphModule into sub-GraphModules for execution on CPU or the accelerator.
    Output is a GraphModule with supported and unsupported operators grouped into as few sub-GraphModules as possible.
    Assumes that only "call_module", "call_function" and "call_method" from FX IR can potentially be executed on the accelerator.

    Given the following graph:
          ==> b ==>
        //         \\\n       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    and providing "operator_support" that indicates that \'b\' and \'c\' can be executed on the accelerator,
    we will get the following split result:

    main:
    def forward(self, a):
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        getitem = run_on_acc_0_0[0]
        getitem_1 = run_on_acc_0_0[1]
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        sin_1 = torch.sin(a)
        cos_1 = torch.cos(a)
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        add_1 = sin_1 + cos_1
        return add_1
    '''
    PCIe_BW: Incomplete
    module: Incomplete
    settings: Incomplete
    operator_support: Incomplete
    sample_input: Incomplete
    acc_nodes: Incomplete
    fusions: Incomplete
    deps: Incomplete
    non_acc_submodule_name: Incomplete
    _node_submodule_map: dict[str, str]
    _return_tuple: Incomplete
    tags: list[str]
    def __init__(self, module: torch.fx.GraphModule, sample_input: Sequence[Any], operator_support: OperatorSupportBase, settings: _SplitterSettingBase, non_acc_submodule_name: str = '_run_on_cpu_', return_tuple: bool = False, nodes_finder: FxNetAccNodesFinder | None = None) -> None:
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
    def get_node_submodule_map(self) -> dict[str, str]:
        """Returns a map from node name to submodule name, e.g.
        node: main_module_impl_impl_over_arch_unary_multiple_embedding
          _pooling_embedding_pooling_sparse_entity_equivalence_key
          _proxy_embedding_bag
        maps to submodule name of: _run_on_acc_1
        """
    def find_deps(self) -> dict[torch.fx.Node, NodeSet]:
        '''
        Builds a graph of node dependencies. Leaf nodes don\'t have any
        dependencies and the "output" node doesn\'t have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        '''
    def update_deps_for_fusions(self) -> None:
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
    def _lower_model_to_backend(self, mod: torch.fx.GraphModule, inputs: Tensors) -> torch.nn.Module:
        """
        Lower the model to a backend.
        """
    def _find_culprit(self, mod: torch.fx.GraphModule, inputs: Tensors) -> str:
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """
    def _draw_graph_based_on_node_support(self, mod: torch.fx.GraphModule, supported_nodes: NodeList): ...
    def node_support_preview(self, dump_graph: bool = False): ...
    def split_preview(self, dump_graph: bool = False): ...
    def find_reverse_deps(self, tag_id: int | None = None) -> dict[torch.fx.Node, NodeSet]:
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
    def update_reverse_deps_for_fusions(self, deps: dict[torch.fx.Node, NodeSet]): ...
    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
    def extend_acc_subgraph(self, tag: str):
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
    def starter_nodes(self) -> tuple[NodeSet, NodeSet]:
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
    def put_nodes_into_subgraphs(self) -> list[Subgraph]: ...
    def remove_small_acc_subgraphs(self, subgraphs: list[Subgraph]) -> list[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
    def tag(self, subgraphs: list[Subgraph]): ...
    def split(self, remove_tag: bool = False) -> torch.fx.GraphModule: ...
    def __call__(self) -> torch.fx.GraphModule: ...
    def generate_split_results(self) -> SplitResult: ...
