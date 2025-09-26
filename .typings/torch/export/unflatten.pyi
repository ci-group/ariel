import abc
import torch
import torch.utils._pytree as pytree
from _typeshed import Incomplete
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from torch.export.exported_program import ExportedProgram, ModuleCallSignature
from typing import Any

__all__ = ['FlatArgsAdapter', 'InterpreterModule', 'InterpreterModuleDispatcher', 'UnflattenedModule', 'unflatten']

class _AttrKind(Enum):
    PARAMETER = 'parameter'
    BUFFER = 'buffer'
    CONSTANT = 'constant'
    MODULE = 'module'

class _SubmoduleBase:
    _ty: str | None
    def type_name(self) -> str | None: ...

class InterpreterModule(_SubmoduleBase, torch.nn.Module):
    """A module that uses torch.fx.Interpreter to execute instead of the usual
    codegen that GraphModule uses. This provides better stack trace information
    and makes it easier to debug execution.
    """
    graph_module: torch.fx.GraphModule | None
    graph: Incomplete
    _ty: Incomplete
    _run_with_interpreter: Incomplete
    def __init__(self, graph: torch.fx.Graph, ty: str | None = None) -> None: ...
    def forward(self, *args, **kwargs): ...
    arg_names: Incomplete
    def finalize(self) -> None: ...
    def print_readable(self, print_output: bool = True, include_stride: bool = False, include_device: bool = False, colored: bool = False): ...

class InterpreterModuleDispatcher(_SubmoduleBase, torch.nn.Module):
    """
    A module that carries a sequence of InterpreterModules corresponding to
    a sequence of calls of that module. Each call to the module dispatches
    to the next InterpreterModule, and wraps back around after the last.
    """
    _modules: Incomplete
    _ty: Incomplete
    _call_modules: Incomplete
    _num_calls: int
    def __init__(self, attrs: set[str], call_modules: list[InterpreterModule]) -> None: ...
    def forward(self, *args, **kwargs): ...
    def call_modules(self): ...
    def print_readable(self, print_output: bool = True, include_stride: bool = False, include_device: bool = False, colored: bool = False): ...

class FlatArgsAdapter(abc.ABC, metaclass=abc.ABCMeta):
    """
    Adapts input arguments with ``input_spec`` to align ``target_spec``.
    """
    @abc.abstractmethod
    def adapt(self, target_spec: pytree.TreeSpec, input_spec: pytree.TreeSpec, input_args: list[Any], metadata: dict[str, Any] | None = None, obj: Any | None = None) -> list[Any]:
        """NOTE: This adapter may mutate given ``input_args_with_path``."""

class UnflattenedModule(torch.nn.Module):
    graph_signature: Incomplete
    graph: Incomplete
    module_call_graph: Incomplete
    flat_args_adapter: Incomplete
    meta: Incomplete
    adapted: bool
    _run_with_interpreter: Incomplete
    ivals: Incomplete
    range_constraints: Incomplete
    equality_constraints: list
    input_placeholders: Incomplete
    check_input_constraints: bool
    def __init__(self, export_module: ExportedProgram, flat_args_adapter: FlatArgsAdapter | None = None) -> None: ...
    def _print_graph(self) -> None: ...
    def _adapt_flat_args(self, flat_args, in_spec, input): ...
    def process_forward_inputs(self, *args, **kwargs): ...
    def forward(self, *args, **kwargs): ...
    def finalize(self) -> None: ...
    def _dispatch_modules(self, redirected_call_indices, consts_targets) -> None:
        """For a module whose call signatures are preserved, replace
        multiple modules corresponding to multiple calls to that module
        with a single dispatcher module that tracks which module to call.
        """
    def print_readable(self, print_output: bool = True, include_stride: bool = False, include_device: bool = False, colored: bool = False): ...

def unflatten(module: ExportedProgram, flat_args_adapter: FlatArgsAdapter | None = None) -> UnflattenedModule:
    """Unflatten an ExportedProgram, producing a module with the same module
    hierarchy as the original eager module. This can be useful if you are trying
    to use :mod:`torch.export` with another system that expects a module
    hierachy instead of the flat graph that :mod:`torch.export` usually produces.

    .. note:: The args/kwargs of unflattened modules will not necessarily match
        the eager module, so doing a module swap (e.g. :code:`self.submod =
        new_mod`) will not necessarily work. If you need to swap a module out, you
        need to set the :code:`preserve_module_call_signature` parameter of
        :func:`torch.export.export`.

    Args:
        module (ExportedProgram): The ExportedProgram to unflatten.
        flat_args_adapter (Optional[FlatArgsAdapter]): Adapt flat args if input TreeSpec does not match with exported module's.

    Returns:
        An instance of :class:`UnflattenedModule`, which has the same module
        hierarchy as the original eager module pre-export.
    """

class _ModuleFrame:
    flat_graph: Incomplete
    nodes: Incomplete
    seen_nodes: Incomplete
    seen_modules: Incomplete
    seen_attrs: Incomplete
    created_modules: Incomplete
    parent: Incomplete
    module_stack: Incomplete
    module_id: Incomplete
    module_call_graph: Incomplete
    verbose: bool
    child_fqn: Incomplete
    module: torch.fx.GraphModule | UnflattenedModule | InterpreterModule
    ivals: Incomplete
    graph: Incomplete
    node_map: dict[torch.fx.Node, torch.fx.Node]
    node_to_placeholder: Incomplete
    parent_call_module: torch.fx.Node | None
    def __init__(self, flat_graph: torch.fx.Graph, nodes: tuple[torch.fx.Node, ...], seen_nodes, seen_modules, seen_attrs, created_modules, parent, module_stack: list[tuple[str, str | None, int]], module_id, module_call_graph: dict[str, ModuleCallSignature], module: torch.fx.GraphModule | UnflattenedModule | None = None) -> None: ...
    def add_placeholder(self, x) -> None: ...
    def copy_sym_call_function(self, x): ...
    def remap_input(self, x): ...
    def finalize_outputs(self): ...
    def copy_node(self, node) -> None: ...
    def run_outer(self) -> None: ...
    def print(self, *args, **kwargs) -> None: ...
    def run_from(self, node_idx): ...

@dataclass
class _SubmoduleEntry:
    parent_fqn: str
    parent_module: torch.nn.Module
    parent_call_module: torch.fx.Node
    fqn: str
    call_idx: int
    module: torch.nn.Module

class _IVals:
    """
    Collect the intermediate values of mutations in a graph.

    Example: in the following graph, suppose that buf_in and buf_out
    are the input and output values of a buffer.

        buf_in = placeholder()
        ...
        ival1 = f0(buf_in, ...)  # inside self.n0(...)
        ...
        ival2 = f1(ival1, ...)  # inside self.n1(...)
        ...
        buf_out = f2(ival2, ...)  # inside self.n2(...)
        return buf_out, ...

    Here ival1 and ival2 are intermediate values created inside
    calls to n0 and n1 respectively, and used inside calls to
    n1 and n2 respectively.
    """
    node_names_by_fqn: Incomplete
    def __init__(self) -> None: ...
    def _is_mutable(self, target): ...
    def read(self, mf, node):
        """
        Read state corresponding to a given intermediate value.
        """
    def update(self, partitions) -> None:
        """
        Update states corresponding to intermediate values that were read.
        """
