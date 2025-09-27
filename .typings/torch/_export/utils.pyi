import inspect
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch._export.passes.lift_constants_pass import ConstantAttrMap as ConstantAttrMap
from torch._guards import detect_fake_mode as detect_fake_mode
from torch._ops import OperatorBase as OperatorBase
from torch._subclasses.fake_tensor import FakeTensor as FakeTensor, FakeTensorMode as FakeTensorMode
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.export import ExportedProgram as ExportedProgram
from torch.export.graph_signature import CustomObjArgument as CustomObjArgument, ExportGraphSignature as ExportGraphSignature, InputKind as InputKind, OutputKind as OutputKind
from torch.fx._pytree import _deregister_pytree_flatten_spec as _deregister_pytree_flatten_spec, register_pytree_flatten_spec as register_pytree_flatten_spec
from torch.fx._utils import first_call_function_nn_module_stack as first_call_function_nn_module_stack
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts as insert_deferred_runtime_asserts
from torch.utils._pytree import Context as Context, FlattenFunc as FlattenFunc, FromDumpableContextFn as FromDumpableContextFn, GetAttrKey as GetAttrKey, KeyPath as KeyPath, MappingKey as MappingKey, SequenceKey as SequenceKey, ToDumpableContextFn as ToDumpableContextFn, UnflattenFunc as UnflattenFunc, _deregister_pytree_node as _deregister_pytree_node, _register_pytree_node as _register_pytree_node, keystr as keystr, tree_flatten_with_path as tree_flatten_with_path
from typing import Any, Callable

placeholder_prefixes: Incomplete
_DISABLE_ATEN_TO_ASSERTION_PASS: bool

def _collect_and_set_constant_attrs(graph_signature, constants, mod) -> ConstantAttrMap: ...
def _register_constants_as_buffers(mod: torch.fx.GraphModule, state_dict, non_persistent_buffers): ...
def _override_graph_signature_for_temp_registered_constants(sig: ExportGraphSignature, temp_registered_constants): ...
def _overwrite_signature_for_non_persistent_buffers(old_sig: ExportGraphSignature, new_sig: ExportGraphSignature): ...
def _collect_param_buffer_metadata(mod: torch.fx.GraphModule) -> dict[str, Any]:
    """
    Param/buffer metadata needs to be saved before lowering to aten IR
    because aten IR lifts them, as a result, automatic preservation doesn't work.
    This is intended to be called on the strict mode tracing right before lowering to
    aten IR OR run_decomposition pass.
    """
def _populate_param_buffer_metadata_to_new_gm(params_buffers_to_node_meta: dict[str, Any], gm: torch.fx.GraphModule, new_sig: ExportGraphSignature) -> None:
    """
    Given that we collected param'buffer metadata before, we put them back in
    newly traced graph module
    """
def _get_shape_env_from_gm(gm: torch.fx.GraphModule): ...
def _rename_without_collisions(name_map: dict[str, str], orig_name: str, name: str, is_placeholder: bool = False):
    """
    Renames nodes to avoid name collisions, with suffixing.
    name_map: map from original name to new name
    orig_name: mapping key
    name: candidate name (potentially suffixed, e.g. mul_2)
    is_placeholder: if the node is a placeholder, avoid detecting suffix
    """
def get_keystr(key_path: KeyPath) -> str:
    '''For a given index into the flat_args, return a human readable string
    describing how to access it, e.g. "*args["foo"][0].bar"
    '''
def _check_symint(symint: int | torch.SymInt, arg: int, range_constraints, unification_map, keypath: KeyPath, i: int | None = None) -> None: ...
def _check_input_constraints_for_graph(input_placeholders: list[torch.fx.Node], flat_args_with_path, range_constraints) -> None: ...
def register_dataclass_as_pytree_node(cls, flatten_fn: FlattenFunc | None = None, unflatten_fn: UnflattenFunc | None = None, *, serialized_type_name: str | None = None, to_dumpable_context: ToDumpableContextFn | None = None, from_dumpable_context: FromDumpableContextFn | None = None, return_none_fields: bool = False) -> None: ...
def is_param(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a parameter within the exported program
    """
def get_param(program: ExportedProgram, node: torch.fx.Node) -> torch.nn.Parameter | None:
    """
    Returns the parameter associated with the given node in the exported program.
    Returns None if the node is not a parameter within the exported program
    """
def is_buffer(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a buffer within the exported program
    """
def get_buffer(program: ExportedProgram, node: torch.fx.Node) -> torch.Tensor | None:
    """
    Returns the buffer associated with the given node in the exported program.
    Returns None if the node is not a buffer within the exported program
    """
def is_lifted_tensor_constant(program: ExportedProgram, node: torch.fx.Node) -> bool:
    """
    Checks if the given node is a lifted tensor constant within the exported program
    """
def get_lifted_tensor_constant(program: ExportedProgram, node: torch.fx.Node) -> torch.Tensor | None:
    """
    Returns the lifted tensor constant associated with the given node in the exported program.
    Returns None if the node is not a lifted tensor constant within the exported program
    """
def sequential_split(gm: torch.fx.GraphModule, node_call_back: Callable[[torch.fx.Node], torch.fx.Node | bool]) -> torch.fx.GraphModule:
    """
    sequential_split creates a new graph module that splits the input graph module into multiple submodules
    based on the node_call_back. It doesn't mutate the input graph module. The node_call_back should return
    True if the node is a delimiter.  Delimiter will be the first node in the next submodule.
    """
def nodes_filter(nodes: list[torch.fx.Node], node_call_back) -> list[torch.fx.Node]:
    """Returns the nodes that match the node_call_back as a list."""
@contextmanager
def _disable_aten_to_metadata_assertions() -> Generator[None]: ...
def _insert_aten_to_metadata_assert_pass(gm: torch.fx.GraphModule) -> None: ...
def apply_runtime_assertion_pass(gm: torch.fx.GraphModule, graph_signature): ...
def nodes_first(nodes: list[torch.fx.Node], node_call_back=None) -> torch.fx.Node | None:
    """
    Returns the first node that matches the node_call_back. If no node matches, returns None.
    When node_call_back is None, returns the first node in the node list.
    """
def nodes_count(nodes: list[torch.fx.Node], node_call_back) -> int:
    """Returns the number of nodes that match the node_call_back."""
def nodes_map(nodes: list[torch.fx.Node], node_call_back) -> list[torch.fx.Node]:
    """
    Sequentially visit the nodes list and invoke node_call_back on each element.
    Returns the nodes list after the node_call_back is invoked on each element.
    """
def node_replace_(old_node: torch.fx.Node, new_node: torch.fx.Node) -> None:
    """
    Replace all uses of old_node with new_node.
    """
def _update_gm_meta_if_possible(gm: torch.fx.GraphModule, mod: torch.nn.Module) -> None: ...
def node_inline_(call_mod_node: torch.fx.Node) -> torch.fx.GraphModule | None:
    """
    Inline the submodule of the given node into the parent module.
    Note: we only support the case where submodule takes tensors inputs.
    """
def _get_torch_jit_trace_forward_signature(mod: torch.nn.Module) -> inspect.Signature:
    """
    Get source code and parse argument names using AST. The function returns
    a signature of the forward() function.

    # TODO: Directly provide inspect.signature compatible TS-d module.
    """
def _bind_signature_to_inputs(mod, fake_args, fake_kwargs): ...
def _name_hoo_subgraph_placeholders(gm: torch.fx.GraphModule) -> None:
    """
    Propagate placeholder names from the top-level graph into HigherOrderOp subgraphs,
    and handle collisions with non-placeholders by count suffixing.
    Different HOO subgraph types have different input schemas, so we first enumerate them
    and gather the top-level named placeholder nodes.
    """
def placeholder_naming_pass(gm: torch.fx.GraphModule, export_graph_signature: ExportGraphSignature, mod: torch.nn.Module, fake_args, fake_kwargs, fake_params_buffers, constants: dict[str, Any]) -> None:
    '''
    This pass is run at the end of _export_non_strict() to assign better placeholder node names:
        - User inputs:
            These follow the signature of mod.forward(), e.g. forward(x, y) produces nodes x, y.
            For nested inputs from dictionaries, lists, tuples, or dataclasses,
            the names are a concatenation of the path to the tensor.
                e.g. x = {
                    \'a\': torch.randn(),
                    \'b\': [torch.randn(), torch.randn()]
                }
            produces nodes x_a, x_b_0, x_b_1.
        - Parameters/buffers/constants/custom objects:
            These follow the FQN of the object, prefixed by "p", "b", "c", "obj" respectively.
                e.g. self.bar.l0.weight produces "p_bar_l0_weight".
        - Effect tokens:
            These are named token, token_1, ...
    '''
def remove_proxy_from_state_dict(state_dict: dict, in_place: bool) -> dict:
    '''
    If `in_place` is false, return a new copy of `state_dict` with "proxy" removed from `v.__dict__`.
    `v` is the values in the dictionary.
    If `in_place` is true, modify `state_dict` in place.
    '''
def _detect_fake_mode_from_gm(gm: torch.fx.GraphModule) -> torch._subclasses.fake_tensor.FakeTensorMode:
    '''
    For a given graph module, we look at the "val" of placeholder nodes to find the fake inputs.
    Additionally, if gm doesn\'t have placeholders, we further look at the "example_value" or "val" of other nodes.
    If no fake mode is found, we return None for fake_mode.
    '''
@contextmanager
def _disable_load_state_dict_hooks(mod: torch.nn.Module): ...
def _is_cia_op(op: OperatorBase) -> bool: ...
def _is_preservable_cia_op(op: OperatorBase) -> bool: ...
def _is_aten_op(op: OperatorBase) -> bool: ...
def _is_custom_op(op: OperatorBase) -> bool: ...
def _materialize_cpp_cia_ops() -> None:
    """
    Utility function to query C++ dispatcher to get the all
    possible CIA ops and populate them into torch.ops namespace
    """
def _special_op_to_preserve_cia(*args, **kwargs):
    """
    This is an special marker that tells our infra that we shouldn't decompose this op.
    """
def _check_valid_to_preserve(op_overload: OperatorBase): ...
def _collect_all_valid_cia_ops_for_aten_namespace() -> set['OperatorBase']: ...
def _collect_all_valid_cia_ops_for_namespace(op_namespace: torch._ops._OpNamespace) -> set['OperatorBase']: ...
def _collect_all_valid_cia_ops() -> set['OperatorBase']:
    """
    This is an util function that gets the all CIA functional ops.

    The algorithm is in 2 steps:
      1. We first query C++ dispatcher to get the list of CIA ops
         and then we call getattr on torch.ops.aten to lazily populate
         them.

      2. Sometimes, handful of ops have CIA registered in python dispatcher
         but not on the C++ side, these can't be caught at the first step.
         So we walk again to get the final list.

    Note that the output of this function should never be modified
    """
def _get_decomp_for_cia(op: OperatorBase): ...
@contextmanager
def _compiling_state_context() -> Generator[None]: ...
def _fakify_params_buffers(fake_mode: FakeTensorMode, mod: torch.nn.Module) -> dict[str, torch.Tensor | torch.nn.Parameter]: ...
def register_module_as_pytree_input_node(cls) -> None:
    """
    Registers a module as a valid input type for :func:`torch.export.export`.

    Args:
        mod: the module instance
        serialized_type_name: The serialized name for the module. This is
        required if you want to serialize the pytree TreeSpec containing this
        module.

    Example::

        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)

            def forward(self, x):
                return self.linear(x)

        torch._export.utils.register_module_as_pytree_node(InputDataClass)

        class Mod(torch.nn.Module):
            def forward(self, x, m):
                return m(x) + x

        ep = torch.export.export(Mod(), (torch.randn(3), Module()))
        print(ep)

    """
def deregister_module_as_pytree_input_node(cls) -> None: ...
def _sync_state(src, dst) -> None: ...
def sync_state(*wrapped_method_modules) -> None:
    """
    Sync state between exported modules corresponding to wrapped methods.
    This might be necessary after serializing/deserializing due to copying.
    """

class _WrappedMethod(torch.nn.Module):
    forward: Incomplete
    def __init__(self, method) -> None: ...

def wrap_method(method):
    """
    Wrap a method as a module so that it can be exported.
    The wrapped module's forward points to the method, and
    the method's original module state is shared.
    """
