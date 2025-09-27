import types
from _typeshed import Incomplete
from torch.fx import Graph, Node
from torch.fx.graph_module import GraphModule
from typing import Callable, TypeVar

__all__ = ['GraphTransformObserver']

T = TypeVar('T')

class GraphTransformObserver:
    __pass_count: int
    gm: Incomplete
    passname: Incomplete
    subsystem: Incomplete
    log_url: Incomplete
    active: Incomplete
    erased_nodes: set[str]
    created_nodes: set[str]
    name_to_node: dict[str, Node]
    copied_gms: list[GraphModule]
    _node_creation_hook: Incomplete
    _node_erase_hook: Incomplete
    _node_replace_hook: Incomplete
    _deepcopy_hook: Incomplete
    input_dot_graph: Incomplete
    def __init__(self, gm: GraphModule, passname: str, subsystem: str | None = None, log_url: str | None = None) -> None:
        """
        log_url is inferred to be torch._inductor.config.trace.log_url_for_graph_xform unless otherwise specified
        """
    @classmethod
    def get_current_pass_count(cls): ...
    def apply_gm_pass(self, pass_fn: Callable[[GraphModule], T]) -> T | None: ...
    def apply_graph_pass(self, pass_fn: Callable[[Graph], T]) -> T | None: ...
    def _check_disable_pass(self): ...
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, tb: types.TracebackType | None) -> None: ...
    def get_node_creation_hook(self): ...
    def get_node_erase_hook(self): ...
    def get_node_replace_hook(self): ...
    def get_deepcopy_hook(self): ...
