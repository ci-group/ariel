import pydot
import torch
import torch.fx
from _typeshed import Incomplete
from torch.fx.passes.shape_prop import TensorMetadata
from typing import Any

__all__ = ['FxGraphDrawer']

class FxGraphDrawer:
    '''
        Visualize a torch.fx.Graph with graphviz
        Basic usage:
            g = FxGraphDrawer(symbolic_traced, "resnet18")
            g.get_dot_graph().write_svg("a.svg")
        '''
    _name: Incomplete
    dot_graph_shape: Incomplete
    normalize_args: Incomplete
    _dot_graphs: Incomplete
    def __init__(self, graph_module: torch.fx.GraphModule, name: str, ignore_getattr: bool = False, ignore_parameters_and_buffers: bool = False, skip_node_names_in_args: bool = True, parse_stack_trace: bool = False, dot_graph_shape: str | None = None, normalize_args: bool = False) -> None: ...
    def get_dot_graph(self, submod_name=None) -> pydot.Dot:
        '''
            Visualize a torch.fx.Graph with graphviz
            Example:
                >>> # xdoctest: +REQUIRES(module:pydot)
                >>> # xdoctest: +REQUIRES(module:ubelt)
                >>> # define module
                >>> class MyModule(torch.nn.Module):
                >>>     def __init__(self) -> None:
                >>>         super().__init__()
                >>>         self.linear = torch.nn.Linear(4, 5)
                >>>     def forward(self, x):
                >>>         return self.linear(x).clamp(min=0.0, max=1.0)
                >>> module = MyModule()
                >>> # trace the module
                >>> symbolic_traced = torch.fx.symbolic_trace(module)
                >>> # setup output file
                >>> import ubelt as ub
                >>> dpath = ub.Path.appdir("torch/tests/FxGraphDrawer").ensuredir()
                >>> fpath = dpath / "linear.svg"
                >>> # draw the graph
                >>> g = FxGraphDrawer(symbolic_traced, "linear")
                >>> g.get_dot_graph().write_svg(fpath)
            '''
    def get_main_dot_graph(self) -> pydot.Dot: ...
    def get_submod_dot_graph(self, submod_name) -> pydot.Dot: ...
    def get_all_dot_graphs(self) -> dict[str, pydot.Dot]: ...
    def _get_node_style(self, node: torch.fx.Node) -> dict[str, str]: ...
    def _get_leaf_node(self, module: torch.nn.Module, node: torch.fx.Node) -> torch.nn.Module: ...
    def _typename(self, target: Any) -> str: ...
    def _shorten_file_name(self, full_file_name: str, truncate_to_last_n: int = 2): ...
    def _get_node_label(self, module: torch.fx.GraphModule, node: torch.fx.Node, skip_node_names_in_args: bool, parse_stack_trace: bool) -> str: ...
    def _tensor_meta_to_label(self, tm) -> str: ...
    def _stringify_tensor_meta(self, tm: TensorMetadata) -> str: ...
    def _get_tensor_label(self, t: torch.Tensor) -> str: ...
    def _to_dot(self, graph_module: torch.fx.GraphModule, name: str, ignore_getattr: bool, ignore_parameters_and_buffers: bool, skip_node_names_in_args: bool, parse_stack_trace: bool) -> pydot.Dot:
        """
            Actual interface to visualize a fx.Graph. Note that it takes in the GraphModule instead of the Graph.
            If ignore_parameters_and_buffers is True, the parameters and buffers
            created with the module will not be added as nodes and edges.
            """
