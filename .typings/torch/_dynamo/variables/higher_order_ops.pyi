import contextlib
from .. import graph_break_hints as graph_break_hints, variables as variables
from ..exc import IncorrectUsage as IncorrectUsage, ObservedException as ObservedException, UncapturedHigherOrderOpError as UncapturedHigherOrderOpError, Unsupported as Unsupported, unimplemented as unimplemented, unimplemented_v2 as unimplemented_v2
from ..source import AttrSource as AttrSource, DictGetItemSource as DictGetItemSource
from ..utils import proxy_args_kwargs as proxy_args_kwargs, set_example_value as set_example_value
from .base import VariableTracker as VariableTracker
from .dicts import ConstDictVariable as ConstDictVariable
from .lazy import LazyVariableTracker as LazyVariableTracker
from .lists import ListVariable as ListVariable, TupleVariable as TupleVariable
from _typeshed import Incomplete
from collections.abc import Generator
from torch._dispatch.python import enable_python_dispatcher as enable_python_dispatcher
from torch._dynamo.symbolic_convert import InstructionTranslator as InstructionTranslator
from torch._dynamo.utils import get_fake_value as get_fake_value
from torch._dynamo.variables.builtin import BuiltinVariable as BuiltinVariable
from torch._dynamo.variables.constant import ConstantVariable as ConstantVariable
from torch._dynamo.variables.functions import UserFunctionVariable as UserFunctionVariable
from torch._dynamo.variables.nn_module import UnspecializedNNModuleVariable as UnspecializedNNModuleVariable
from torch._dynamo.variables.tensor import SymNodeVariable as SymNodeVariable
from torch._guards import Source as Source
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch.fx.passes.shape_prop import _extract_tensor_metadata as _extract_tensor_metadata

log: Incomplete
hc_log: Incomplete

def raise_hard_error_if_graph_break(reason): ...
@contextlib.contextmanager
def discard_graph_changes(tx) -> Generator[None]: ...
def check_meta_consistency_vt(vars1: list[VariableTracker], vars2: list[VariableTracker], lhs_name: str, rhs_name: str, include_contiguity: bool = True) -> None: ...
@contextlib.contextmanager
def dynamo_enable_grad(tx: InstructionTranslator, enable: bool = True): ...
@contextlib.contextmanager
def dynamo_under_activation_checkpoint(tx: InstructionTranslator): ...
def find_mismatched_vars(var, types, allow_none: bool = False):
    """
    Recursively finds variables whose type is not an instance of the specified types.
    Args:
        var: The variable to check.
        types: A tuple of allowed types.
        allow_none (bool): Whether to allow None values. Defaults to False.
    Returns:
        A set of variables whose type is not an instance of the specified types.
    """
def only_consist_of(var, types, allow_none: bool = False): ...
def _make_inlined(tx: InstructionTranslator, f): ...
def _call_function_and_unflatten_output(tx, fn, args, kwargs, flat_example_value, ret_treespec): ...
def _assert_tensors_nonaliasing(inputs, outputs) -> None: ...
def _check_all_tensorvariable(args) -> None: ...
def _check_supported_callable_arg(tx: InstructionTranslator, func_var: VariableTracker, arg_name): ...
def are_same_graph_modules(fn_name, a_mod, b_mod, fake_mode): ...
def validate_args_and_maybe_create_graph_inputs(sub_args, tracer, tx, set_subgraph_inputs, description, sub_args_names=None): ...
def _merge_graph_inputs(l_graph, l_lifted_freevars, l_name, r_graph, r_lifted_freevars, r_name): ...
def speculate_subgraph(tx, f, sub_args, sub_kwargs, description, *, source_target=None, always_restore: bool = False, enable_grad=None, set_subgraph_inputs: str = 'automatic', restore_side_effects: bool = True, should_flatten_outputs: bool = False, under_activation_checkpoint: bool = False, supports_input_mutation: bool = True, supports_aliasing: bool = True, tracer=None): ...
def make_attr(tx: InstructionTranslator, name): ...

class TorchHigherOrderOperatorVariable(VariableTracker):
    value: Incomplete
    source: Incomplete
    def __init__(self, value: HigherOrderOperator, source: Source | None = None, **kwargs) -> None: ...
    @staticmethod
    def make(value, source=None, **kwargs): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
    def as_python_constant(self): ...

class CustomFunctionHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Wraps torch._functorch.autograd_function.custom_function_call
    """
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class CallTorchbindHigherOrderVariable(TorchHigherOrderOperatorVariable):
    script_obj_var: Incomplete
    method_name: Incomplete
    def __init__(self, hop, source, script_obj_var, method_name) -> None: ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

def validate_subgraph_output_types(output: VariableTracker):
    """Verify that that the output of the subgraph is a tensor,
    int, bool, SymBool, or SymInt.
    """

class WhileLoopHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class AssociativeScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class ScanHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

def non_single_tensor_return_unsupported(api, ret) -> None: ...

class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class FunctorchHigherOrderVariable(UserFunctionVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class FunctionalCallVariable(FunctorchHigherOrderVariable):
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def install_subgraph_in_output_graph(self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name: str = 'wrap_body'): ...
    def create_wrapped_node(self, tx: InstructionTranslator, fn_vt, fn_args_vt, kwargs, description, under_activation_checkpoint: bool = False, *, subgraph_name: str = 'wrap_body'): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class WrapWithSetGradEnabledHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class WrapWithAutocastHigherOrderVariable(TorchHigherOrderOperatorVariable):
    """
    This hop is not exposed to users but is inserted into the graph
    after export as a post-processing step.
    """
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class HintsWrapperHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class StrictModeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class CheckpointHigherOrderVariable(WrapHigherOrderVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class DynamoBypassingWrapperHigherOrderVariable(WrapHigherOrderVariable):
    def __init__(self, hop, source) -> None: ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class RunWithRNGStateHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class AutoFunctionalizeHigherOrderVariable(TorchHigherOrderOperatorVariable):
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class FlexAttentionBackwardHighOrderVariable(TorchHigherOrderOperatorVariable):
    def proxy_submod(self, tx, arg): ...
    def to_proxy(self, tx, arg): ...
    def call_function(self, tx, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):
    """
    Handles torch._dynamo._trace_wrapped_higher_order_op.inner_trace
    by unwrapping the higher order op and inlining through it.  This op
    is created by dynamo to survive through AotAutograd, then unwrapped
    here in the call to dynamo from compiled autograd.
    """
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class FlexAttentionHigherOrderVariable(TorchHigherOrderOperatorVariable):
    @staticmethod
    def normalize_to_args(args, kwargs): ...
    def create_wrapped_node(self, tx: InstructionTranslator, query: VariableTracker, fn: VariableTracker, fn_name: str): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class AutogradFunctionApplyVariable(VariableTracker):
    fwd_graph: Incomplete
    bwd_graph: Incomplete
    parent_source: Incomplete
    def __init__(self, fwd_graph, bwd_graph, parent_source, **kwargs) -> None: ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

def _get_fake_value(x): ...
def maybe_positional_arg_names(func): ...

class BaseHOPVariable(WrapHigherOrderVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def python_type(self): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...

class InvokeSubgraphHigherOrderVariable(WrapHigherOrderVariable):
    supports_input_mutation: bool
    supports_aliasing: bool
    def install_subgraph_in_output_graph(self, tx, fn_vt, fn_args_vt, kwargs, body_gmod, attr_name): ...
    def call_function(self, tx: InstructionTranslator, args: list[VariableTracker], kwargs: dict[str, VariableTracker]) -> VariableTracker: ...
