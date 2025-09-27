import functools
import torch
import torch.utils._pytree as pytree
from .. import config as config
from .functional_utils import FunctionalTensorMetadataEq as FunctionalTensorMetadataEq, _check_if_mutation_can_be_in_graph as _check_if_mutation_can_be_in_graph
from .utils import strict_zip as strict_zip
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from torch._guards import Source as Source
from torch._ops import OpOverload as OpOverload
from torch._subclasses import FakeTensor as FakeTensor
from torch._subclasses.fake_tensor import is_fake as is_fake
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any, Callable, NamedTuple

zip = strict_zip
OutputType: Incomplete

@dataclass(frozen=True)
class OutputAliasInfo:
    output_type: OutputType
    raw_type: type
    base_idx: int | None
    dynamic_dims: set[int] | None
    requires_grad: bool
    functional_tensor: FunctionalTensorMetadataEq | None = ...

class MutationType(Enum):
    NOT_MUTATED = 1
    MUTATED_IN_GRAPH = 2
    MUTATED_OUT_GRAPH = 3

@dataclass(frozen=True)
class InputAliasInfo:
    is_leaf: bool
    mutates_data: bool
    mutates_metadata: bool
    mutations_hidden_from_autograd: bool
    mutations_under_no_grad_or_inference_mode: bool
    mutation_inductor_storage_resize: bool
    mutates_storage_metadata: bool
    requires_grad: bool
    keep_input_mutations: bool
    def __post_init__(self) -> None: ...
    @functools.cached_property
    def mutation_type(self) -> MutationType: ...

@dataclass
class MemoryFormatMeta:
    size: Sequence[int] | None = ...
    stride: Sequence[int] | None = ...
    memory_format: torch.memory_format | None = ...
    @staticmethod
    def from_tensor(t: torch.Tensor) -> MemoryFormatMeta | None: ...

@dataclass
class PlainTensorMeta:
    unwrapped_idx: int
    memory_format: MemoryFormatMeta | None = ...

@dataclass
class SubclassCreationMeta:
    '''
    Used for AOTDispatch.
    This dataclass gives us the information we need to reconstruct a tensor subclass
    from our flat inputs.
    Why is this important? The graph that we\'d like to trace out contains flat tensor inputs,
    But the user\'s original model may have subclass inputs and outputs.
    So we need to wrap/unwrap subclasses as necessary to translate between the user\'s
    view (subclass inps/outs), and the backend compiler\'s view (graph with no subclass args).

    Complications arise mostly from the fact that a subclass can hold more than one inner tensor;
    So for a given subclass input/output, we need to carefully track which indices map
    to the subclass tensor in the corresponding "dense-tensor-only" graph.
    '''
    flat_tensor_start_idx: int
    arg_count: int
    included_subclass_symints: bool
    attrs: dict[str, SubclassCreationMeta | PlainTensorMeta]
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]
    meta: Any
    original_subclass: torch.Tensor | None
    original_subclass_type: type | None = ...
    memory_format: MemoryFormatMeta | None = ...
    def compute_outer_size_and_stride(self, all_args, *, curr_start_idx: int): ...
    def creation_fn(self, all_args, *, is_runtime: bool): ...
    def make_runtime_safe(self): ...
    def __post_init__(self) -> None: ...

@dataclass(eq=False)
class ViewAndMutationMeta:
    input_info: list[InputAliasInfo]
    output_info: list[OutputAliasInfo]
    num_intermediate_bases: int
    keep_input_mutations: bool
    traced_tangents: list[Any]
    subclass_inp_meta: list[PlainTensorMeta | SubclassCreationMeta]
    subclass_fw_graph_out_meta: list[PlainTensorMeta | SubclassCreationMeta]
    subclass_tangent_meta: list[PlainTensorMeta | SubclassCreationMeta]
    is_train: bool = ...
    traced_tangent_metas: list[Any] | None = ...
    num_symints_saved_for_bw: int | None = ...
    grad_enabled_mutation: bool | None = ...
    deterministic: bool | None = ...
    static_input_indices: list[int] = field(default_factory=list)
    tokens: dict[Any, torch.Tensor] = field(default_factory=dict)
    indices_of_inputs_that_requires_grad_with_mutations_in_bw: list[int] = field(default_factory=list)
    bw_donated_idxs: list[int] | None = ...
    num_backward_tokens: int = ...
    num_graphsafe_rng_states: int = ...
    graphsafe_rng_state_index: int | None = ...
    mutated_graph_handled_indices = ...
    num_mutated_graph_handled_indices = ...
    mutated_graph_handled_indices_seen_by_autograd = ...
    num_mutated_graph_handled_indices_seen_by_autograd = ...
    mutated_inp_runtime_indices = ...
    num_mutated_inp_runtime_indices = ...
    aliased_out_indices = ...
    unsafe_view_out_indices = ...
    num_outputs = ...
    num_outputs_non_aliased = ...
    num_outputs_aliased_to_inputs = ...
    num_unsafe_view_outputs = ...
    num_outputs_aliased_to_intermediates = ...
    num_outputs_aliased = ...
    dynamic_outputs = ...
    dynamic_saved_tensors_idxs: dict[int, set[int]] = ...
    output_types = ...
    is_rng_op_functionalized = ...
    num_outputs_rng_offset = ...
    num_forward_returns = ...
    num_forward = ...
    def __post_init__(self) -> None: ...
    def make_runtime_safe(self):
        """
        There are various fields in ViewAndMutationMeta that aren't serializable. This function is called after all tracing
        is completed to simplify certain fields in the metadata so that they can be safely cached.

        Doing so may lose information (in the case of traced_tangents), but none of the information is needed at runtime.
        """
    @property
    def tensors_saved_for_backwards_slice(self): ...
    @property
    def symints_saved_for_backwards_slice(self): ...
    def __eq__(self, other): ...

@dataclass(eq=False)
class SubclassMeta:
    fw_metadata: ViewAndMutationMeta
    grad_input_metas: list[PlainTensorMeta | SubclassCreationMeta] | None = ...
    def __init__(self) -> None: ...

@dataclass(frozen=True)
class TensorAlias:
    alias: torch.Tensor

@dataclass
class BackwardSignature:
    """
    Provides information about the backward section of an exported
    joint forward-backward graph.
    For a particular fx GraphModule, this class contains information on:
    (1) A mapping from each gradient (backwards output) to the parameter
        it corresponds to (forward input)
    (2) A mapping from each gradient (backwards output) to the user input
        it corresponds to (forward input)
    (3) Which of the forward outputs corresponds to the loss, that we backprop on.

    Each string name is the `node.name` of the corresponding node in the fx graph.
    """
    gradients_to_parameters: dict[str, str]
    gradients_to_user_inputs: dict[str, str]
    loss_output: str

GraphOutputName: Incomplete
GraphInputName: Incomplete
FQN: Incomplete

@dataclass
class GraphSignature:
    """
    Provides information about an exported module.
    For a particular fx GraphModule, this class contains information on:
    (1) Which graph inputs are parameters, buffers, or user inputs
    (2) (for params/buffers) a mapping from the name of each graph argument
        to its parameter/buffer FQN in the original nn.Module.
    (3) If there are input mutations, these are represented as extra outputs
        in the fx GraphModule. We provide a mapping from these
        extra output names to the names of the actual inputs.
    (4) The pytree metadata on how to flatten/unflatten inputs and outputs.
        The corresponding FX GraphModule only accepts and returns
        pytree-flattened inputs/outputs.
    (5) (Optionally) if the FX is a joint forward-backward graph, we provide
        a signature on the backward section of the joint graph.
    """
    parameters: list[FQN]
    buffers: list[FQN]
    user_inputs: list[GraphInputName]
    user_outputs: list[GraphOutputName]
    inputs_to_parameters: dict[GraphInputName, FQN]
    inputs_to_buffers: dict[GraphInputName, FQN]
    buffers_to_mutate: dict[GraphOutputName, FQN]
    user_inputs_to_mutate: dict[GraphOutputName, GraphInputName]
    in_spec: pytree.TreeSpec
    out_spec: pytree.TreeSpec
    backward_signature: BackwardSignature | None
    input_tokens: list[GraphInputName]
    output_tokens: list[GraphOutputName]
    @classmethod
    def from_tracing_metadata(cls, *, in_spec: pytree.TreeSpec, out_spec: pytree.TreeSpec, graph_input_names: list[str], graph_output_names: list[str], view_mutation_metadata: ViewAndMutationMeta, named_parameters: list[str], named_buffers: list[str], num_user_inputs: int, num_user_outputs: int, loss_index: int | None, backward_signature: BackwardSignature | None) -> GraphSignature: ...

@dataclass
class AOTAutogradCacheInfo:
    cache_key: str
    start_time_ns: int
    forward_symints: list[torch.SymInt]

@dataclass
class AOTConfig:
    """
    Configuration for AOTDispatcher
    """
    fw_compiler: Callable
    bw_compiler: Callable
    partition_fn: Callable
    decompositions: dict[OpOverload, Callable]
    num_params_buffers: int
    aot_id: int
    keep_inference_input_mutations: bool
    is_export: bool = ...
    no_tangents: bool = ...
    dynamic_shapes: bool = ...
    aot_autograd_arg_pos_to_source: list[Source] | None = ...
    static_input_indices: list[int] | None = ...
    inference_compiler: Callable | None = ...
    enable_log: bool = ...
    pre_dispatch: bool = ...
    cache_info: AOTAutogradCacheInfo | None = ...
    ignore_shape_env: bool = ...
    precompile_backend_id: str | None = ...
    def __post_init__(self) -> None: ...

class SubclassTracingInfo(NamedTuple):
    plain_tensor_trace_fn: Incomplete
    plain_tensor_args: Incomplete
    maybe_subclass_meta: Incomplete
