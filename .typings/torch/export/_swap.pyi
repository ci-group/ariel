import torch
from _typeshed import Incomplete
from torch.export.exported_program import ConstantArgument as ConstantArgument, ExportedProgram as ExportedProgram, ModuleCallSignature as ModuleCallSignature
from torch.fx.passes.tools_common import NodeList as NodeList, legalize_graph as legalize_graph
from torch.fx.passes.utils.fuser_utils import erase_nodes as erase_nodes, fuse_as_graphmodule as fuse_as_graphmodule

log: Incomplete

def _get_getitem_users(node: torch.fx.Node) -> set[torch.fx.Node]: ...
def _try_remove_connecting_pytrees(curr_module_node: torch.fx.Node) -> None:
    """
    We want to try to remove extraneous pytree flatten/unflatten calls between modules
    calls. Instead of having the following:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %tree_flatten_spec : [num_users=1] = call_function[target=torch.fx._pytree.tree_flatten_spec](args = (%foo, %_spec_1), kwargs = {})
        %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_flatten_spec, 0), kwargs = {})
        %tree_unflatten_1 : [num_users=2] = call_function[target=torch.utils._pytree.tree_unflatten](args = ([%getitem_4], %_spec_2), kwargs = {})
        %getitem_5 : [num_users=1] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 0), kwargs = {})
        %getitem_7 : [num_users=0] = call_function[target=operator.getitem](args = (%tree_unflatten_1, 1), kwargs = {})
        %getitem_6 : [num_users=1] = call_function[target=operator.getitem](args = (%getitem_5, 0), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    We could do the following, if we know that all the outputs of `foo` feed into `bar`:
    graph():
        ...
        %foo : [num_users=1] = call_module[target=foo](args = (%getitem_1, %getitem_2), kwargs = {})
        %bar : [num_users=1] = call_module[target=bar](args = (%getitem_6,), kwargs = {})
        ...

    Currently this optimization only works for the case where all of the outputs
    of `foo` go directly into `bar`, and `bar` has no other inputs.
    """
def _remove_extraneous_pytrees(gm: torch.fx.GraphModule) -> None:
    """
    Remove extraneous pytree flatten/unflatten calls.

    We try a couple of optimizations here:
        1. Remove pytree flatten/unflatten calls between modules
        2. TODO: Remove module's in_spec + initial unflatten call
        3. TODO: Remove module's out_spec + final flatten call
    """
def _construct_inputs(gm: torch.fx.GraphModule, signature: ModuleCallSignature, node_name_map: dict[str, torch.fx.Node]) -> tuple[list[torch.fx.Node], dict[str, torch.fx.Node]]: ...
def _insert_call_module(gm: torch.fx.GraphModule, args_nodes: list[torch.fx.Node], kwargs_nodes: dict[str, torch.fx.Node], module_to_swap: torch.nn.Module, name: str) -> torch.fx.Node: ...
def _deconstruct_outputs(gm: torch.fx.GraphModule, signature: ModuleCallSignature, module_node: torch.fx.Node, node_name_map: dict[str, torch.fx.Node], orig_outputs: tuple[torch.fx.Node, ...]) -> None: ...
def _swap_module_helper(gm: torch.fx.GraphModule, modules_to_swap: dict[str, torch.nn.Module], module_call_graph: dict[str, ModuleCallSignature]) -> torch.fx.GraphModule: ...
def _fix_input_output_signature(gm: torch.fx.GraphModule, signature: ModuleCallSignature) -> None:
    """
    Given the unlifted module from calling ep.module(), we want to remove the
    pytree processing from the graph module's PyTreeCodeGen and instead make it
    nodes inside of the graph. This allows us to do some optimizations, like
    remove these pytree calls if it is unnecessary, and makes the PyTree part
    more obvious to graph passes.
    """
def _swap_modules(ep: ExportedProgram, modules_to_swap: dict[str, torch.nn.Module]) -> torch.fx.GraphModule:
    """
    Unlifts the given ExportedProgram into a fx.GraphModule, and then swaps
    previously traced modules with new eager modules specified. Returns a
    fx.GraphModule with a custom forward function.

    Args:
        ep (ExportedProgram): Exported program to modify
        modules_to_swap (Dict[str, torch.nn.Module]): Mapping from module fqn to
            eager module to swap with. The specified module fqn should have also
            been specified in the `preserve_module_call_signature` argument to
            torch.export so that we know how to restore the calling convention
            to this argument.
        run_with_interpreter: Whether or not to run the graph using
            fx.Interpreter. Setting to true will help result in better error
            messages and easier debugging, but it has found to result in a QPS
            drop.
    """
