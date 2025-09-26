import contextlib
import inspect
import torch
from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from sympy import Symbol as Symbol
from torch._dynamo.source import AttrSource as AttrSource, GetItemSource as GetItemSource, LocalSource as LocalSource, TensorProperty as TensorProperty, TensorPropertySource as TensorPropertySource
from torch._dynamo.variables.builder import TrackedFake as TrackedFake
from torch._export.passes.lift_constants_pass import ConstantAttrMap as ConstantAttrMap
from torch._export.utils import _fakify_params_buffers as _fakify_params_buffers
from torch._guards import Source as Source
from torch._library.fake_class_registry import FakeScriptObject as FakeScriptObject
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from torch.export import Constraint as Constraint
from torch.export.dynamic_shapes import _DimHint as _DimHint, _DimHintType as _DimHintType, _IntWrapper as _IntWrapper, _RelaxedConstraint as _RelaxedConstraint, _check_dynamic_shapes as _check_dynamic_shapes, _combine_args as _combine_args, _process_dynamic_shapes as _process_dynamic_shapes, _tree_map_with_path as _tree_map_with_path
from torch.export.graph_signature import CustomObjArgument as CustomObjArgument
from torch.fx.experimental.symbolic_shapes import ConstraintViolationError as ConstraintViolationError, DimDynamic as DimDynamic, EqualityConstraint as EqualityConstraint, GuardOnDataDependentSymNode as GuardOnDataDependentSymNode, RelaxedUnspecConstraint as RelaxedUnspecConstraint, ShapeEnv as ShapeEnv, StatelessSymbolicContext as StatelessSymbolicContext, SymIntSymbolicContext as SymIntSymbolicContext, ValueRanges as ValueRanges, _find_user_code_frame as _find_user_code_frame, _suggest_fixes_for_data_dependent_error_non_strict as _suggest_fixes_for_data_dependent_error_non_strict
from torch.utils._pytree import GetAttrKey as GetAttrKey, KeyPath as KeyPath, MappingKey as MappingKey, SequenceKey as SequenceKey, tree_map_with_path as tree_map_with_path
from torch.utils._sympy.numbers import int_oo as int_oo
from typing import Any

log: Incomplete

class _KeyPath:
    """
    Wraps `KeyPath` to aid `isinstance` checks.
    """
    kp: Incomplete
    def __init__(self, kp: KeyPath) -> None: ...

class _KeyPathTrie:
    """
    Builds a trie of `KeyPath` prefixes mapping to `Source` leaves.
    """
    root: Incomplete
    def __init__(self) -> None: ...
    def add(self, kp: KeyPath, src: Source): ...
    def get(self, kp: KeyPath) -> tuple[Source, KeyPath]: ...

def make_sourced_prefixes(nn_module, args, kwargs) -> _KeyPathTrie: ...
def key_path_to_source(kp: KeyPath, sourced_prefixes: _KeyPathTrie | None = None) -> Source:
    """
    Given a key path, return the source for the key path.
    """
def _is_constant_argument(t): ...
def fakify(mode: FakeTensorMode, kp: KeyPath, t: Any, t_constraints: dict[int, dict[int, Constraint]], sources: dict[tuple[int, int], list[Source]], sourced_prefixes: _KeyPathTrie | None = None): ...
def _is_unbacked_symint(symbol): ...
def _tensor_min_max(*args, real_callable, tensor_callable, **kwargs):
    """
    This logic is replicated from dynamo/variables/builtin.py
    """
@contextmanager
def _override_builtin_ops() -> Generator[None, None, Incomplete]: ...
def make_fake_inputs(nn_module, args, kwargs, dynamic_shapes, _is_torch_jit_trace: bool = False, allow_complex_guards_as_runtime_asserts: bool = False):
    """
    Given an nn module, example inputs, and constraints, return a new fake mode,
    fake inputs created in that mode whose dynamic shape dimensions are constrained
    by the given ranges, and sources for pairs of dynamic shape dimensions that are
    constrained to be equal.
    """
def _flatten_dynamic_shapes(combined_args: dict[str, Any], dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any]) -> list[Any]: ...
def _clean_dynamic_markers(tensor: torch.Tensor) -> None: ...
def produce_guards_and_solve_constraints(fake_mode: FakeTensorMode, gm: torch.fx.GraphModule, dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, equalities_inputs: EqualityConstraint, original_signature: inspect.Signature, _is_torch_jit_trace: bool = False):
    """
    Given a fake mode, sources pairs corresponding to equal dynamic shape dimensions,
    and a graph module, produce guards on the fake mode's shape env (raising constraint
    violations if any), solve (to suggest simplifications or fixes).
    Dynamo already performs this, so this is for non-strict mode.

    Additional inputs:
        equalities_inputs: the equality constraints to use for guards
        original_signature: the signature of the forward method
    """
def is_int(x: object) -> bool: ...
def _constrain_user_specified_dimhint_range(symint: torch.SymInt, hint: int, dim: _DimHint, range_constraints, shape_env, keypath: KeyPath, i: int | None = None) -> str | None: ...
def make_constraints(fake_mode: FakeTensorMode, gm: torch.fx.GraphModule, combined_args: dict[str, Any], dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None, num_lifted_inputs: int):
    """
    Given a fake mode's shape env and user-specified dynamic shapes,
    return the resulting range constraints and equality constraints.

    Additional args:
        num_lifted_inputs: the number of non-user-input placeholder nodes in the graph
        (used only to enumerate the user-input nodes)
    """
def _gather_constant_attrs(m: torch.nn.Module) -> ConstantAttrMap:
    """Search the module hierarchy, gathering up all tensor and ScriptObject constants.

    Returns a dictionary mapping hash(value) to the name of the constant. We
    have to abuse `hash` here unfortunately, see: [ScriptObject hash].
    """
def _get_graph_inputs_of_type_nn_module(args: tuple[tuple[Any], dict[Any, Any]] | None) -> set[type[torch.nn.Module]]: ...
def _enter_enable_graph_inputs_of_type_nn_module(module_types: set[type[torch.nn.Module]]) -> None: ...
def _exit_enable_graph_inputs_of_type_nn_module(module_types: set[type[torch.nn.Module]]) -> None: ...
@contextlib.contextmanager
def _enable_graph_inputs_of_type_nn_module(args: tuple[tuple[Any], dict[Any, Any]] | None): ...
@contextlib.contextmanager
def _fakify_module_inputs(args: tuple[Any], kwargs: dict[Any, Any], fake_mode: torch._subclasses.fake_tensor.FakeTensorMode): ...
@contextlib.contextmanager
def _fakify_script_objects(mod: torch.nn.Module, args: Sequence[Any], kwargs: dict[Any, Any], fake_mode: torch._subclasses.fake_tensor.FakeTensorMode): ...

class _NonStrictTorchFunctionHandler(torch.overrides.TorchFunctionMode):
    '''
    1. Handles data-dependent errors raised by torch function calls in non-strict.

    Any data-dependent error is due to some condition on unbacked symints
    that cannot be resolved. A mechanical way of fixing the error is to use
    a torch._check() call to assert either that condition or its negation.
    The handler suggests these options as code and points to the location
    of the torch function call that raised the error as part of the error
    message shown to the user, who can then simply select and copy-paste
    a suggested fix at that location.

    NOTE: Not all data-dependent errors are raised by torch function calls.
    In particular, conditions on unbacked symints can appear outside such
    calls, and as such are not handled here.

    2. Overrides torch functions that are known to cause problems in non-strict.

    Certain Python features, such as indexing/slicing, cannot be intercepted
    in non-strict. Likewise, certain legacy ops, such as distributed collectives,
    may need to be mapped to other ops. When there is special handling in Dynamo
    for such things, tracing can fail in non-strict (while succeeding in strict).
    Fortunately, redirecting to other torch functions can often fix such issues.

    3. Handles line-of-code logging for each torch function call in non-strict.

    Usage: TORCHEXPORT_EXTENDED_DEBUG_CURRENT_LOC=1 TORCH_LOGS="+export" ...
    '''
    def _override(self, func, args, kwargs): ...
    def __torch_function__(self, func, types, args=(), kwargs=None): ...
