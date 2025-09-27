import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from enum import Enum
from torch._dynamo.utils import counters as counters, get_metrics_context as get_metrics_context
from torch._inductor.utils import GraphPartitionMap as GraphPartitionMap, InputType as InputType
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Callable

perf_hint_log: Incomplete
static_inputs_log: Incomplete
OutputType = list[int | torch.Tensor | None]
ModelType = Callable[[list[InputType]], OutputType]

@dataclasses.dataclass(frozen=True)
class FunctionID:
    """Unique counter of a function wrapped in cudagraphify_impl"""
    id: int

@dataclasses.dataclass(frozen=True)
class PlaceholderInfo:
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to cudagraphs, and will cache these results.
    """
    name: str
    stack_trace: str | None
    users: list[PlaceholderInfo]
    mutating_use_stack_trace: str | None

@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for CUDA graph replay,
    with a little more metadata so we can identify if we have an applicable
    CUDA graph in our CUDA graph tree for it.
    """
    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]

def get_mutating_use_stack_trace_from_node(placeholder_node: torch.fx.Node) -> str | None: ...
def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> str | None: ...
def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo: ...
def get_placeholder_info(graph: torch.fx.Graph) -> list[PlaceholderInfo]: ...
def format_default_skip_message(reason: str) -> str: ...
def get_mutation_stack_trace(placeholders: Sequence[PlaceholderInfo], mutation_indices: Sequence[int]) -> str: ...
def check_for_mutation(func: WrappedFunction, inputs: list[InputType], is_cuda_graph_recorded_tensor: Callable[[torch.Tensor], bool]) -> str | None: ...
def _get_use_stack_trace(node: torch.fx.Node) -> str | None: ...
def check_multiple_devices_or_any_cpu_nodes(device_node_mapping: dict[torch.device, torch.fx.Node]) -> str | None: ...
def check_lowering_disable_cudagraph(device_node_mapping: dict[torch.device, torch.fx.Node]) -> str | None: ...
def log_cudagraph_skip_and_bump_counter(msg: str) -> None: ...

@dataclasses.dataclass
class BoxedDeviceIndex:
    value: int | None
    def set(self, device_idx: int | None) -> None: ...

def check_for_mutation_ignore_cuda_graph_managed_tensor(gm: torch.fx.GraphModule, mutated_inputs: OrderedSet[str], mutated_input_idxs: OrderedSet[int], static_input_idxs: Sequence[int]) -> str | None: ...
def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> str | None:
    """
    Gets the first non-empty stack trace of a placeholder or its users.
    """

class CheckInvariantStatus(Enum):
    SUCCESS = 1
    CudagraphManagedIdxMismatch = 2
    StaticInputIdxMismatch = 3
    ExpectedDeadIndicesBeforeGraphMismatch = 4
    def __str__(self) -> str: ...

def log_data_ptr_mismatch(placeholders: Sequence[PlaceholderInfo], inputs: list[InputType], recorded_data_ptr: Sequence[int | None], target_idxs: Sequence[int], mismatch: CheckInvariantStatus) -> str:
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """
def maybe_warning_due_to_dynamic_shape(fn_cache: dict[tuple[int, ...], Callable[..., Any]], new_int_key: Any) -> bool: ...

@dataclasses.dataclass(frozen=True)
class CudagraphCachedInfo:
    """
    Info needed to realign inputs
    """
    placeholders: Sequence[PlaceholderInfo]
    stack_traces: list[str | None]
    cudagraph_fail_reasons: list[str]

@dataclasses.dataclass(frozen=True)
class CudagraphMetadata:
    """
    Metadata for recording a CUDA graph.
    """
    placeholders: Sequence[PlaceholderInfo]
    static_input_idxs: OrderedSet[int]
    mutated_input_idxs: OrderedSet[int]
    stack_traces: list[str | None]
    constants: dict[str, torch.Tensor]

def get_partition_cudagraph_metadata(partition_map: GraphPartitionMap, metadata: CudagraphMetadata) -> CudagraphMetadata:
    """
    Convert the cudagraph metadata at the graph level to the graph partition level,
    given the graph partition info (i.e., mapping from partition input/output index
    to graph input/output index).
    """
