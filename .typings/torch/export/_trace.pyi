import dataclasses
import torch
import torch.fx
import torch.utils._pytree as pytree
from ._safeguard import AutogradStateOpsFailSafeguard as AutogradStateOpsFailSafeguard
from ._wrapper_utils import _WrapperModule as _WrapperModule
from .exported_program import ExportedProgram as ExportedProgram, InputKind as InputKind, ModuleCallEntry as ModuleCallEntry, ModuleCallSignature as ModuleCallSignature, _disable_prexisiting_fake_mode as _disable_prexisiting_fake_mode
from .graph_signature import ExportGraphSignature as ExportGraphSignature, _convert_to_export_graph_signature as _convert_to_export_graph_signature
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.exc import UserError as UserError, UserErrorType as UserErrorType
from torch._export.db.logging import exportdb_error_message as exportdb_error_message, get_class_if_classified_error as get_class_if_classified_error
from torch._export.non_strict_utils import _NonStrictTorchFunctionHandler as _NonStrictTorchFunctionHandler, _fakify_module_inputs as _fakify_module_inputs, _fakify_script_objects as _fakify_script_objects, _gather_constant_attrs as _gather_constant_attrs, _override_builtin_ops as _override_builtin_ops, make_constraints as make_constraints, make_fake_inputs as make_fake_inputs, produce_guards_and_solve_constraints as produce_guards_and_solve_constraints
from torch._export.passes.collect_tracepoints_pass import CollectTracepointsPass as CollectTracepointsPass
from torch._export.passes.lift_constants_pass import ConstantAttrMap as ConstantAttrMap, _materialize_and_lift_constants as _materialize_and_lift_constants
from torch._export.utils import _collect_param_buffer_metadata as _collect_param_buffer_metadata, _compiling_state_context as _compiling_state_context, _fakify_params_buffers as _fakify_params_buffers, _populate_param_buffer_metadata_to_new_gm as _populate_param_buffer_metadata_to_new_gm, _update_gm_meta_if_possible as _update_gm_meta_if_possible, apply_runtime_assertion_pass as apply_runtime_assertion_pass, placeholder_naming_pass as placeholder_naming_pass, placeholder_prefixes as placeholder_prefixes
from torch._export.verifier import SpecViolationError as SpecViolationError
from torch._export.wrappers import _wrap_submodules as _wrap_submodules
from torch._functorch._aot_autograd.input_output_analysis import _graph_input_names as _graph_input_names, _graph_output_names as _graph_output_names
from torch._functorch._aot_autograd.schemas import GraphSignature as GraphSignature
from torch._functorch._aot_autograd.subclass_utils import get_subclass_typing_container as get_subclass_typing_container
from torch._functorch._aot_autograd.traced_function_transforms import create_functional_call as create_functional_call
from torch._functorch._aot_autograd.utils import create_tree_flattened_fn as create_tree_flattened_fn, register_buffer_assignment_hook as register_buffer_assignment_hook
from torch._functorch.aot_autograd import _detect_attribute_assignment as _detect_attribute_assignment, aot_export_module as aot_export_module
from torch._guards import TracingContext as TracingContext, detect_fake_mode as detect_fake_mode, tracing as tracing
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._logging import dtrace_structured as dtrace_structured
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch._utils_internal import log_export_usage as log_export_usage
from torch.export._unlift import _check_input_constraints_pre_hook as _check_input_constraints_pre_hook
from torch.export.dynamic_shapes import _DimHintType as _DimHintType, _IntWrapper as _IntWrapper, _check_dynamic_shapes as _check_dynamic_shapes, _combine_args as _combine_args, _process_dynamic_shapes as _process_dynamic_shapes
from torch.export.exported_program import OutputKind as OutputKind
from torch.fx._symbolic_trace import _ConstantAttributeType as _ConstantAttributeType
from torch.fx.experimental.proxy_tensor import PreDispatchTorchFunctionMode as PreDispatchTorchFunctionMode, get_proxy_slot as get_proxy_slot, make_fx as make_fx, track_tensor_tree as track_tensor_tree
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError as ConstraintViolationError, GuardOnDataDependentSymNode as GuardOnDataDependentSymNode, ShapeEnv as ShapeEnv, free_unbacked_symbols as free_unbacked_symbols
from torch.fx.graph import _PyTreeCodeGen as _PyTreeCodeGen, _PyTreeInfo as _PyTreeInfo
from torch.utils._pytree import TreeSpec as TreeSpec
from torch.utils._sympy.value_ranges import ValueRangeError as ValueRangeError
from typing import Any, Callable

log: Incomplete

@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    """
    allow_rnn: bool = ...
    reorderable_logging_functions: set[Callable] = dataclasses.field(default_factory=set)
    do_not_emit_runtime_asserts: bool = ...
    specialize_int: bool = ...
    specialize_float: bool = ...
    assume_static_by_default: bool = ...
    automatic_dynamic_shapes: bool = ...
    capture_dynamic_output_shape_ops: bool = ...
    capture_scalar_outputs: bool = ...
    prefer_deferred_runtime_asserts_over_guards: bool = ...

@dataclasses.dataclass
class ATenExportArtifact:
    gm: torch.fx.GraphModule
    sig: ExportGraphSignature
    constants: dict[str, _ConstantAttributeType]

@dataclasses.dataclass(frozen=True)
class ExportArtifact:
    aten: ATenExportArtifact
    in_spec: TreeSpec
    out_spec: TreeSpec
    fake_mode: FakeTensorMode
    module_call_specs: dict[str, dict[str, pytree.TreeSpec]]

DEFAULT_EXPORT_DYNAMO_CONFIG: Incomplete

@contextmanager
def _ignore_backend_decomps() -> Generator[None]: ...
@contextmanager
def _disable_custom_triton_op_functional_decomposition() -> Generator[Incomplete]: ...
def custom_triton_ops_decomposition_disabled(): ...
def _fixup_key(x): ...
def _strip_root(x): ...
def _rewrite_tracepoint_node(gm: torch.fx.GraphModule):
    """
    In-place modifiy input graph module by replacing the export tracepoint with a new node
    that has the same target and args, but with the _export_root stripped from path.
    """
def detect_shape_env(inputs: Any = None): ...
def _extract_fake_inputs(gm, args, kwargs):
    """
    Given a graph module, extract fakified input tensors from the metadata of
    its placeholders, and map them to the structure of given args and kwargs.
    Also return the fake mode used to fakify those inputs.
    """
def _replace_param_buffer_names(param_buffer_table, sig) -> None: ...
def _convert_to_positional_args(orig_arg_names, args, kwargs): ...
def _normalize_nn_module_stack(gm_torch_level, root_cls): ...
def _get_param_buffer_mapping(original_module: torch.nn.Module, traced_module: torch.nn.Module) -> dict[str, str]:
    """
    Returns a mapping of parameter/buffer names from the new module to the
    original model. This is to help with restoring the FQN for parameter/buffers
    of a traced module to what the original module contains.
    """
def _preserve_requires_grad_pass(gm: torch.fx.GraphModule, sig: ExportGraphSignature, fake_params_buffers: dict[str, torch.Tensor], constants: dict[str, _ConstantAttributeType], flat_fake_args: list[Any]): ...
def _remap_constants(orig_constant_attrs: ConstantAttrMap, graph_signature: ExportGraphSignature, constants: dict[str, _ConstantAttributeType]) -> None:
    """Rewrite the graph signature and constants table to use the FQN from the original module."""
def _replace_unbacked_bindings(gm: torch.fx.GraphModule) -> None:
    """
    When we run an interpreter-based pass over a GraphModule, execution of data-dependent operators
    will produce example values with new unbacked symbols. To track that the new/old symbols are equivalent,
    we used to rely on the unbacked_renamings mapping. This led to problematic metadata where the unbacked_bindings
    keys mapped new symbols (u2) to paths containing old symbols (u0) in the example values, or worse, backed symbols
    or constants (e.g. if the original unbacked was replaced/specialized). Additionally this created problems with
    de/serialized programs, since we didn't comprehensively serialize ShapeEnv/unbacked renamings/node bindings.

    This pass attempts a simpler way of handling these for export, by throwing away the previously computed bindings, and re-running
    the pattern match used in compute_unbacked_bindings. This ensures we keep the original symbols contained in the example values,
    or delete bindings if they've been replaced/specialized.
    """
def _produce_aten_artifact(*, gm: torch.fx.GraphModule, mod, constant_attrs, graph_signature, pre_dispatch, fake_args, fake_kwargs, fake_params_buffers, _prettify_placeholder_names: bool = True) -> ATenExportArtifact:
    """
    This is a helper function that is shared between export_to_aten_ir and export_to_aten_ir_make_fx
    to produce the aten artifact. (export compatible graph module + signature)

    It does:
    1. Applies runtime assertion pass
    2. Recompute unbacked_bindings pass
    3. Populate meta val when missing
    4. Lift constants as placeholders
    5. Replace raw autograd and autocast ops with HOPs
    6. Prettify names for placeholders
    7. Preserve requires_grad value on node meta val
    """
def _rename_constants_nodes(gm: torch.fx.GraphModule, graph_signature: ExportGraphSignature) -> None:
    """
    For strict mode, rename constants nodes that were previously annotated as buffers.
    """
def _restore_state_dict(original_module: torch.nn.Module, traced_module: torch.fx.GraphModule) -> None:
    """
    Restores the state dict of the traced module to that of the original module.
    """
def _get_module_hierarchy(mod: torch.nn.Module) -> dict[str, str]: ...
def _make_module_call_graph(in_spec: TreeSpec, out_spec: TreeSpec, module_call_signatures: dict[str, ModuleCallSignature], forward_arg_names: list[str] | None = None) -> list[ModuleCallEntry]: ...
def _export_to_torch_ir(f: Callable, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None, *, preserve_module_call_signature: tuple[str, ...] = (), disable_constraint_solver: bool = False, allow_complex_guards_as_runtime_asserts: bool = False, restore_fqn: bool = True, _log_export_usage: bool = True, same_signature: bool = True) -> torch.fx.GraphModule:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a torch.fx.GraphModule in torch IR.
    """
def _export_to_aten_ir(mod: torch.nn.Module, fake_args, fake_kwargs, fake_params_buffers, constant_attrs: ConstantAttrMap, produce_guards_callback=None, *, transform=..., pre_dispatch: bool = False, decomp_table=None, _check_autograd_state: bool = True, _is_torch_jit_trace: bool = False, _prettify_placeholder_names: bool = True, decompose_custom_triton_ops: bool = False) -> ATenExportArtifact: ...
def _get_forward_arg_names(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> list[str]:
    """
    Gets the argument names to forward that are used, for restoring the
    original signature when unlifting the exported program module.
    - Positional args: retain the original argument names, and enumerate
        *args as args_0, args_1, ...
    - Keyword args: retain the original kwarg names in the order specified
        by the user. This order seems to matter for the current state of
        export lifted modules.
    """
def _get_non_persistent_buffers(mod: torch.nn.Module) -> set[str]:
    """
    Returns set of non-persistent buffers in a module and its submodules.
    """
def _rewrite_dynamo_tensor_constants(orig_mod_buffers: set[torch.Tensor], traced_mod_buffers: dict[str, torch.Tensor], graph_signature: ExportGraphSignature, constants: dict[str, _ConstantAttributeType]) -> None:
    """
    Dynamo erroneously marks tensor attributes on modules as buffers.
    Rewrite them to be tensor constants.
    """
def _move_non_persistent_buffers_to_tensor_constants(orig_mod: torch.nn.Module, graph_signature: ExportGraphSignature, constants: dict[str, _ConstantAttributeType]) -> None:
    """
    Moves non-persistent buffers to tensor constants.
    """
def _verify_nn_module_stack(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform nn_module_stack checks on the graph.
    Current constraints:
        For the top level graph:
        - populated for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
        For submodule graphs:
        - None for 'placeholder', output'

    TODO(pianpwk): make this a consistent node-level check once nn_module_stack is populated for cond submodules.
    """
def _verify_stack_trace(graph_module: torch.fx.GraphModule) -> None:
    """
    Perform stack trace checks on the graph.
    Constraints:
        - None or non-empty str for 'call_function', 'get_attr'
        - None for 'placeholder', 'output'
    """
def _verify_placeholder_names(gm: torch.fx.GraphModule, sig: ExportGraphSignature) -> None:
    """
    Performs a sanity check on the placeholder node names.
    - User input nodes: no restrictions, should match the original forward() signature
    - Params/buffers/constants/custom_obj/token nodes: should start with prefixes defined in <placeholder_prefixes>
    """
def get_ep_stats(ep: ExportedProgram) -> dict[str, Any]: ...

_EXPORT_FLAGS: set[str] | None
_EXPORT_MODULE_HIERARCHY: dict[str, str] | None

def _log_export_wrapper(fn): ...
def _process_jit_trace_inputs_for_export(example_inputs, example_kwarg_inputs): ...
def _get_original_state_dict(mod: torch.nn.Module) -> dict[str, Any]: ...
def _process_export_inputs(mod, args, kwargs, dynamic_shapes): ...
def _get_module_call_graph(export_artifact: ExportArtifact, preserve_module_call_signature: tuple[str, ...], strict_mode_export: bool, forward_arg_names: list[str] | None = None) -> tuple[torch.fx.GraphModule, list[ModuleCallEntry]]:
    """
    In-place modify the graph module in export_artifact, remove _export_tracepoint nodes and
    return module_call_graph.
    """
def _get_range_constraints(mod: torch.nn.Module, export_artifact: ExportArtifact, args, kwargs, dynamic_shapes, _is_torch_jit_trace: bool = False): ...
def _get_inline_constraints(fake_mode: FakeTensorMode): ...
@contextmanager
def patch_forward(obj: torch.nn.Module, new_method):
    """Helper method to make it easier to cleanly torch.export() a method on a
    module that is not `forward`.
    """
@contextmanager
def _temp_disable_texpr_fuser() -> Generator[None]: ...
def _convert_ts_to_export_experimental(traced_callable, args, kwargs=None): ...
def _strict_export(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, preserve_module_call_signature: tuple[str, ...], orig_in_spec: TreeSpec, allow_complex_guards_as_runtime_asserts: bool, _is_torch_jit_trace: bool, _to_aten_func: Callable) -> ExportArtifact:
    """
    _to_aten_func can either be `_export_to_aten_ir_make_fx` or `_export_to_aten_ir`
    """
def _export_to_aten_ir_make_fx(mod: torch.nn.Module, fake_args, fake_kwargs, fake_params_buffers, constant_attrs: ConstantAttrMap, produce_guards_callback=None, transform=...) -> ATenExportArtifact: ...
def set_missing_meta_vals(gm, flat_args, num_params_buffers) -> None: ...
def _find_node(gm: torch.fx.GraphModule, name: str) -> torch.fx.Node: ...
def _non_strict_export(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, preserve_module_call_signature: tuple[str, ...], orig_in_spec: TreeSpec, allow_complex_guards_as_runtime_asserts: bool, _is_torch_jit_trace: bool, _to_aten_func: Callable) -> ExportArtifact:
    """
    _to_aten_func can either be `_export_to_aten_ir_make_fx` or `_export_to_aten_ir`
    """
@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export_for_training(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None, *, strict: bool = True, preserve_module_call_signature: tuple[str, ...] = ()) -> ExportedProgram: ...
@_log_export_wrapper
@_disable_prexisiting_fake_mode
def _export(mod: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None, *, strict: bool = True, preserve_module_call_signature: tuple[str, ...] = (), pre_dispatch: bool = False, allow_complex_guards_as_runtime_asserts: bool = False, _is_torch_jit_trace: bool = False) -> ExportedProgram:
    """
    Traces either an nn.Module's forward function or just a callable with PyTorch
    operations inside and produce a ExportedProgram.

    Args:
        mod: the `nn.Module` to trace.

        args: example positional inputs.

        kwargs: optional example keyword inputs.

        dynamic_shapes:
         An optional argument where the type should either be:
         1) a dict from argument names of ``f`` to their dynamic shape specifications,
         2) a tuple that specifies dynamic shape specifications for each input in original order.
         If you are specifying dynamism on keyword args, you will need to pass them in the order that
         is defined in the original function signature.

         The dynamic shape of a tensor argument can be specified as either
         (1) a dict from dynamic dimension indices to :func:`Dim` types, where it is
         not required to include static dimension indices in this dict, but when they are,
         they should be mapped to None; or (2) a tuple / list of :func:`Dim` types or None,
         where the :func:`Dim` types correspond to dynamic dimensions, and static dimensions
         are denoted by None. Arguments that are dicts or tuples / lists of tensors are
         recursively specified by using mappings or sequences of contained specifications.

        preserve_module_call_signature: A list of submodule paths for which the original
            calling conventions are preserved as metadata.

        allow_complex_guards_as_runtime_asserts:
         With the current dynamic shapes language for dims and derived dims, we can run into constraints
         that are not expressible with the language. For example, flattening a matrix and adding to a vector,
         both fully dynamic (i.e. x.reshape([-1]) + y) emits a guard s0 * s1 = s2, which is not expressible.
         By default, we either raise a constraint violation error or specialize to static values.
         If this flag is set to True, we avoid erroring out and instead allow complex constraints to exist as runtime
         assertions in the graph. The sympy interpreter (torch/utils/_sympy/interp.py) will produce the math ops
         required to compute and assert the value of the guard (e.g. sym_size_int, eq, _assert_scalar).
         Additionally, if TORCH_DYNAMO_DO_NOT_EMIT_RUNTIME_ASSERTS=1 is specified, we will allow complex constraints
         while not emitting runtime asserts, returning a cleaner graph with lesser guarantees around dynamic shapes.

    Returns:
        An ExportedProgram containing the traced module.
    """
