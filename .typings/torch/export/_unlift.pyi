import torch
import torch.utils._pytree as pytree
from ._remove_effect_tokens_pass import _remove_effect_tokens as _remove_effect_tokens
from ._tree_utils import reorder_kwargs as reorder_kwargs
from .exported_program import ExportGraphSignature as ExportGraphSignature, ExportedProgram as ExportedProgram, InputKind as InputKind, OutputKind as OutputKind
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._export.non_strict_utils import _enter_enable_graph_inputs_of_type_nn_module as _enter_enable_graph_inputs_of_type_nn_module, _exit_enable_graph_inputs_of_type_nn_module as _exit_enable_graph_inputs_of_type_nn_module, _get_graph_inputs_of_type_nn_module as _get_graph_inputs_of_type_nn_module
from torch._export.utils import _check_input_constraints_for_graph as _check_input_constraints_for_graph
from torch.export.unflatten import _AttrKind as _AttrKind, _assign_attr as _assign_attr
from torch.fx.experimental.proxy_tensor import _pytree_subclasses_that_lose_info as _pytree_subclasses_that_lose_info
from torch.fx.graph import _PyTreeCodeGen as _PyTreeCodeGen, _PyTreeInfo as _PyTreeInfo
from torch.fx.traceback import NodeSource as NodeSource, NodeSourceAction as NodeSourceAction
from typing import Any

def eq_spec(self, other: pytree.TreeSpec) -> bool:
    """
    Refinement of TreeSpec.__eq__ where, e.g., torch.Size(...) matches tuple(...).
    See _pytree_subclasses_that_lose_info in proxy_tensor.py for more details.
    """
def _check_inputs_match(args, kwargs, in_spec: pytree.TreeSpec) -> list: ...
@torch._dynamo.disable
def _check_input_constraints_pre_hook(self, args, kwargs) -> None: ...
def _unlift_inputs_as_getattr(gm: torch.fx.GraphModule, lifted_inputs: Sequence[str | None]) -> tuple[dict[str, torch.fx.Node], dict[str, torch.fx.Node]]:
    """
    Unlift inputs referring to params/buffers/constants as getattr nodes in the
    graph
    """
def _insert_copy_for_mutations(gm: torch.fx.GraphModule, mutated_outputs: Sequence[str | None], unlifted_name_to_node: dict[str, torch.fx.Node], input_name_to_node: dict[str, torch.fx.Node]) -> None:
    """
    Find the all the buffers and inputs that were mutated and insert copy_
    operators to reflect mutations.
    """
def _get_codegen(in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec | None, forward_arg_names: list[str] | None = None) -> _PyTreeCodeGen:
    """
    Create the codegen for the graph module based on the in/out specs
    """
def _unlift(gm: torch.fx.GraphModule, lifted_inputs: Sequence[str | None], mutated_outputs: Sequence[str | None], in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec | None, state_dict: dict[str, Any], constants: dict[str, Any], forward_arg_names: list[str] | None = None):
    """
    Args:
        lifted_inputs: A list matching the graph module's input nodes. For
        an input node that is referring to a lifted parameter/buffer, this
        list will contain the fqn the corresponding attribute. Otherwise, this
        list will contain None. This is used to unlift the lifted parameters as
        get_attr nodes.

        mutated_outputs: A list matching the graph module's output nodes. For
        an output node that is referring to a mutated buffer or user input, this
        list will contain the name of the corresponding buffer or user input
        that needs to be mutated. Otherwise, this list will contain None. This
        is used to re-insert an inplace copy_ operator to copy the mutated
        values back to the original node.
    """
def _register_attrs_to_new_gm(new_gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature, state_dict: dict[str, Any], constants: dict[str, Any]) -> None: ...

class _StatefulGraphModuleFactory(type):
    """
    Metaclass that ensures a private constructor for _StatefulGraphModule
    """
    def __call__(cls, *args, **kwargs) -> None: ...
    def _create(cls, root, graph, range_constraints=None): ...

class _StatefulGraphModule(torch.fx.GraphModule, metaclass=_StatefulGraphModuleFactory):
    range_constraints: Incomplete
    validate_inputs: bool
    def __init__(self, root, graph, range_constraints=None) -> None: ...

def _create_stateful_graph_module(plain_graph_module: torch.fx.GraphModule, range_constraints, ep: ExportedProgram) -> _StatefulGraphModule: ...
def _unlift_exported_program_lifted_states(ep: ExportedProgram) -> torch.nn.Module: ...
