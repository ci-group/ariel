import contextlib
import dataclasses
import torch
from . import config as config
from _typeshed import Incomplete
from collections.abc import Generator, Iterator, Sequence
from contextlib import AbstractContextManager
from enum import Enum
from torch import Tensor as Tensor
from torch._C import _cuda_CUDAAllocator_AllocatorState as AllocatorState
from torch._dynamo.callback import CallbackTrigger as CallbackTrigger
from torch._dynamo.mutation_guard import GenerationTracker as GenerationTracker
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed, preserve_rng_state as preserve_rng_state
from torch._guards import CompileId as CompileId
from torch._inductor.compile_fx import align_inputs_from_check_idxs as align_inputs_from_check_idxs, copy_misaligned_inputs as copy_misaligned_inputs, get_expanded_dims as get_expanded_dims, get_input_idxs_to_check as get_input_idxs_to_check, index_expanded_dims as index_expanded_dims, remove_unaligned_input_idxs as remove_unaligned_input_idxs, static_input as static_input
from torch._inductor.cudagraph_utils import CheckInvariantStatus as CheckInvariantStatus, FunctionID as FunctionID, ModelType as ModelType, OutputType as OutputType, PlaceholderInfo as PlaceholderInfo, WrappedFunction as WrappedFunction, check_for_mutation as check_for_mutation, log_cudagraph_skip_and_bump_counter as log_cudagraph_skip_and_bump_counter, log_data_ptr_mismatch as log_data_ptr_mismatch, maybe_warning_due_to_dynamic_shape as maybe_warning_due_to_dynamic_shape
from torch._inductor.utils import InputType as InputType
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.storage import UntypedStorage as UntypedStorage
from torch.types import _bool as _bool
from torch.utils._ordered_set import OrderedSet as OrderedSet
from torch.utils.weak import TensorWeakRef as TensorWeakRef
from typing import Any, Callable, TypeVar

StorageWeakRefPointer = int
StorageDataPtr = int
NBytes = int
S = TypeVar('S', bound='StorageWeakRefWrapper')

class AllocatorState: ...

log: Incomplete

@dataclasses.dataclass(frozen=True)
class GraphID:
    """Unique counter of a cuda graph recording"""
    id: int

def clear_cublass_cache() -> None:
    """
    Cublas keeps a persistent workspace allocation for running matmuls. This poses a problem for
    doing warmup within a CUDAGraph private pool because we do not want persistent allocations from
    one one run to the next. When we begin a new run of a cudagraphs path (generation), all tensors
    from the previous generation are freed. This frees them the memory pool, but not elsewhere.
    A tensor in the cublas workspace would continue to be in use the workspace but would also get allocated
    in the next run. The memory would be in use in two places.

    To solve this, we clear cublas caches before and after warming up or recording. If a workspace is required
    it will be allocated to the cudagraph private pool and accounted for in the allocator for the duration of the
    program. There is no overhead to this on replay since cudagraphs removes allocation overhead.
    """
@contextlib.contextmanager
def clear_cublas_manager() -> Generator[None, None, None]:
    """Context manager around clearing cublas caches that will clear on enter and exit"""
@contextlib.contextmanager
def disable_conv_cache_emptying() -> Generator[None, None, None]: ...
@contextlib.contextmanager
def enable_history_recording() -> Generator[None, None, None]:
    """Turns on history recording in the CUDA Caching Allocator"""
def get_history_recording() -> AbstractContextManager[None]: ...

class TreeManagerContainer:
    """
    Manages the lifetime of the tree manager. Like `PrivatePool` in cuda caching allocator,
    the tree and its corresponding memory pool should be kept alive as long as any outstanding
    graph or tensor which is an output of a graph remains alive.

    There is a single tree manager container per device.

    The lifecycle of a tree_manager is:
    -  Is constructed, no graph, no fns, no tensors
    -  Tree manager is fetched, resulting in tree manager being allocated
    -  We generate a bunch of functions, calling add_strong_reference
    -  These functions die, calling finalize_reference
    -  When all the functions die, we finalize_tree_manager.

    TODO: in the future, we would like to do the following once storage weak refs land
    -  We look for all the live storages and add references to THOSE
    -  We count as storages die
    -  All the storages are dead, we deallocate the tree manager
    """
    tree_manager: CUDAGraphTreeManager | None
    live_cudagraphify_fns: int
    device_index: Incomplete
    live_storages_count: int
    graph: torch.cuda.CUDAGraph | None
    lock: Incomplete
    def __init__(self, device_index: int) -> None: ...
    def _finalize_tensor(self) -> None: ...
    def finalize_cudagraphify_fn(self) -> None: ...
    def _finalize_tree_manager(self) -> None: ...
    def add_strong_reference(self, fn: Callable[..., Any]) -> None: ...
    def get_tree_manager(self) -> CUDAGraphTreeManager: ...

local: Incomplete

class MarkStepBox:
    mark_step_counter: int

def mark_step_begin() -> None:
    """Indicates that a new iteration of inference or training is about to begin."""
def reset_cudagraph_trees() -> None:
    """Clear all cudagraph trees"""
def get_obj(local: Any, attr_name: str) -> Any: ...
def get_container(device_index: int) -> TreeManagerContainer: ...
def get_manager(device_index: int, create_if_none_exists: bool = True) -> CUDAGraphTreeManager | None: ...
def is_cudagraph_capture_sizes(int_key: int | tuple[int, ...]) -> bool:
    """
    Returns true if all dynamic shapes should be captured or the dynamic shape
    int_key should be captured.
    """
def cudagraphify_impl(model: ModelType, inputs: list[InputType], static_input_idxs: Sequence[int], *args: Any, **kwargs: Any) -> ModelType: ...
@contextlib.contextmanager
def dynamo_timed_cudagraph(name: str, compile_id: CompileId | None, mode: CompilationMode | None) -> Generator[Any, None, None]:
    """
    Makes usages of dynamo_timed in this file less verbose. NOTE: This CM sums
    all durations into a single column in the dynamo_compile table. Use only if
    you consider the timed region to be part of the runtime overhead associated
    with the compiler.
    """
def cudagraphify(model: ModelType, inputs: list[InputType], static_input_idxs: Sequence[int] = (), *, device_index: int, is_backward: bool, is_inference: bool, stack_traces: StackTraces | None = None, constants: tuple[torch.Tensor, ...] = (), placeholders: tuple[PlaceholderInfo, ...] = (), mutated_input_idxs: tuple[int, ...] = (), compile_id: CompileId | None = None) -> tuple[ModelType, OutputType]: ...

class StorageWeakRefWrapper:
    """
    Wrapper around a storage weak ref. Will deallocate it upon expiration if invoked.
    """
    __slots__: Incomplete
    storage_ref: StorageWeakRef | None
    ref: Incomplete
    _data_ptr: Incomplete
    extra_ref_check: Incomplete
    def __init__(self, inp: Tensor | UntypedStorage, extra_ref_check: Callable[[], bool] | None = None) -> None:
        """
        extra_ref_check is an additional check we need to run to check if the
        weak ref has expired. in checking storage use count we assume extra_ref_check
        will hold an additional reference to the storage.
        """
    @classmethod
    def from_weakref_and_data_ptr(cls, cdata: Any, data_ptr: int, extra_ref_check: Callable[[], bool] | None = None) -> StorageWeakRefWrapper: ...
    def __call__(self) -> StorageWeakRefPointer | None: ...
    def swap_weakref(self, cdata: Any) -> None: ...
    def data_ptr(self) -> int:
        """NB: returns the data ptr even if the storage has expired"""
    def remove_extra_reference(self) -> None: ...
    def expired(self) -> bool: ...
    def __repr__(self) -> str: ...

def is_live(weak_ref: StorageWeakRefWrapper | None) -> bool: ...
def maybe_deref(weak_ref: StorageWeakRefWrapper | None) -> tuple[StorageWeakRefPointer, int] | None: ...
@contextlib.contextmanager
def _use_cuda_memory_pool_manager(device: int, mem_pool: tuple[int, int], stream: torch.cuda.Stream) -> Generator[None, None, None]:
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
def map_to_ref(t: Tensor | None) -> StorageWeakRefWrapper | None: ...
PathOutputIndex = tuple[int, int]
PathLiveness = list[list[bool]]
StackTraces = list[str | None]

class CUDAWarmupNode:
    """
    Simplified Wrapper around A CUDA Model that wraps outputs in storage refs and exposes
    apis to get the live storages in the current chain of warmup.

    A CUDAWarmupNode may have either CUDAGraphNode or CUDAWarmupNode as a parent, but may only have
    CUDAWarmupNode as children, because we cannot record or execute with tensors which do not have stable
    memory addresses.

    CUDAWarmupNode and CUDAGraphNode have a number of differences that make it easier to use separate classes.
    - Much of the CUDAGraphNode logic & initialization is based on the tensor properties of first recording. In the
    first instance of warmup, these are not finalized yet.
    - All Inputs to the RecordedFunction must be copied over to the cuda graph memory pool, this is unnecessary in warmup.
    - CUDAWarmup is only used once and so does not need to optimize as much bookkeeping. It is much simpler.

    NB: this class and CUDAGraphNode need to expose `path_live_weakrefs`, `all_outputs_are_dead`, and
    `self.outputs_weakrefs`, `stack_traces`, and `tensor_weakrefs` for compatibility.
    """
    wrapped_function: Incomplete
    parent: CUDAGraphNode | CUDAWarmupNode | None
    cuda_graphs_pool: Incomplete
    outputs_weakrefs: list[StorageWeakRefWrapper | None]
    tensor_weakrefs: list[TensorWeakRef | None]
    existing_cuda_graph: Incomplete
    has_run: bool
    device_index: Incomplete
    stack_traces: Incomplete
    stream: Incomplete
    already_warm: Incomplete
    id: Incomplete
    def __init__(self, wrapped_function: WrappedFunction, parent: CUDAGraphNode | CUDAWarmupNode | None, cuda_graphs_pool: tuple[int, int], existing_cuda_graph: torch.cuda.CUDAGraph | None, device_index: int, stack_traces: StackTraces | None, stream: torch.cuda.Stream, already_warm: bool, id: GraphID) -> None: ...
    def run(self, new_inputs: Any) -> OutputType: ...
    @property
    def _path_from_root(self) -> Generator[CUDAGraphNode | CUDAWarmupNode, None, None]: ...
    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        """Returns all live storages weakrefs that created by nodes in this path"""
    def all_outputs_are_dead(self) -> bool: ...
    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor) -> bool: ...
InputList = list
OutputList = list
LevelList = list

class OutputAliasInfo: ...
class _UnaliasedStorage(OutputAliasInfo):
    """Singleton to mark that the graph output constructs a new alias or is None"""

UnaliasedStorage: Incomplete

class AliasesPriorGraphOutput(OutputAliasInfo):
    """Marks that the graph output aliases an output of a prior graph"""
    __slots__: Incomplete
    index: PathOutputIndex
    def __init__(self, index: PathOutputIndex) -> None: ...

class AliasesNewOutput(OutputAliasInfo):
    """Marks that the graph output aliases an index in the new, returned outputs"""
    __slots__: Incomplete
    index: int
    def __init__(self, index: int) -> None: ...

class CUDAGraphNode:
    """
    A single recording of a function into a CUDA Graph. Recordings of CUDA Graphs share a single memory pool
    and are structured into a tree, where there is a single recording that can precede it (parent) and multiple
    subsequent recordings that may follow (children). A node will have no parent if it is the first recording
    in a tree; i.e., when it is first recorded, there are no live tensors from a previous recording which
    would force a dependency.

    On first recording, all of the live tensors in the current CUDA Graph Node path will be
    reflected in the corresponding private pool. On subsequent executions, the caching allocator
    is unaffected when the graph is replayed.

    In order to support recording a subsequent cuda graph recording after execution of this graph,
    we checkpoint the state of the memory pool so that it may later be resumed.

    WrappedFunction should have already been warmed up prior to invocation.

    See [setCheckpointPoolState] for further explanation, as well as
    https://user-images.githubusercontent.com/13564/222815509-374f3400-f83d-4f7d-8fa6-4a092b3250bb.png
    """
    wrapped_function: Incomplete
    id: Incomplete
    device: Incomplete
    stack_traces: Incomplete
    stream: Incomplete
    rerecord_if_static_inputs_change: Incomplete
    _parent: Incomplete
    cuda_graphs_pool: Incomplete
    children: dict[FunctionID, list[CUDAGraphNode]]
    outputs_weakrefs: OutputList[StorageWeakRefWrapper | None]
    path_weakrefs: LevelList[OutputList[StorageWeakRefWrapper | None]]
    path_stacktraces: LevelList[StackTraces | None]
    tensor_weakrefs: OutputList[TensorWeakRef | None]
    cudagraph_managed_idxs: list[int]
    live_cudagraph_managed_path_refs: InputList[PathOutputIndex | None]
    preserved_aliased_inputs: InputList[bool]
    static_input_idxs: list[int]
    non_static_input_idx: LevelList[int]
    non_managed_static_input_idxs: LevelList[int]
    static_input_data_ptrs: InputList[int | None]
    expanded_dims: list[list[int]]
    recorded_liveness_before_graph: LevelList[OutputList[bool]]
    recorded_liveness_after_graph: LevelList[OutputList[bool]]
    expected_dead_indices_before_graph: list[PathOutputIndex]
    expected_dead_indices_after_graph: list[PathOutputIndex]
    live_indices_after_graph: list[PathOutputIndex]
    graph: torch.cuda.CUDAGraph | None
    reconstructed_inputs: list[InputType]
    checkpointed_caching_state: AllocatorState | None
    output_storage_alias: OutputList[OutputAliasInfo | None]
    unaliased_in_all_paths: OutputList[bool]
    cached_tensor_outputs: OutputList[Tensor | None]
    static_output_tensors: OutputList[Tensor | None]
    recording_outputs: OutputType | None
    outputs_metadata: OutputList[dict[str, Any] | int | None]
    def __init__(self, wrapped_function: WrappedFunction, id: GraphID, parent: CUDAGraphNode | None, inputs: list[InputType], cuda_graphs_pool: tuple[int, int], device_index: int, stack_traces: StackTraces | None, stream: torch.cuda.Stream, mode: CompilationMode | None, compile_id: CompileId | None) -> None: ...
    def _copy_inputs_and_remove_from_src(self, dsts: list[InputType], srcs: list[InputType]) -> None: ...
    def check_static_inputs_are_stable(self, new_inputs: list[InputType]) -> None: ...
    def run_first_inputs(self, new_inputs: list[InputType]) -> OutputType: ...
    static_inputs_stable: bool
    def run(self, new_inputs: list[InputType]) -> OutputType: ...
    def reconstruct_outputs(self) -> OutputType:
        """Reconstruct output tensors according to their saved metadata and alias information"""
    def prepare_alias_info_for_tensor_construction(self, out_alias_info: OutputAliasInfo | None, metadata: dict[str, Any] | int | None) -> UntypedStorage | None | int: ...
    def prepare_storages_for_construction(self) -> list[UntypedStorage | None | int]: ...
    def run_graph(self) -> None: ...
    def all_outputs_are_dead(self) -> bool:
        """All outputs of the path from this node to its root are dead"""
    def _record(self, model: ModelType, inputs: list[InputType]) -> OutputType:
        """Record the model"""
    def _add_first_outputs(self, outputs: OutputType, static_input_persistent_storage_ptrs: dict[int, StorageWeakRefWrapper]) -> None:
        """Add the outputs from the first invocation of the node and set up metadata"""
    def _mark_prior_graph_output_as_aliased(self, index: PathOutputIndex) -> None:
        """Remove a graph output from the unaliased, cached tensors in an ancestor node"""
    def _initialize_cached_tensors(self) -> None: ...
    def get_output_refcount(self, index: int) -> int: ...
    @property
    def parent(self) -> CUDAGraphNode | None:
        """unwraps the weakref to _parent"""
    @property
    def _path_to_root(self) -> Generator[CUDAGraphNode, None, None]:
        """Returns all nodes in the path starting at self and ending at root"""
    @property
    def _path_from_root(self) -> Generator[CUDAGraphNode, None, None]:
        """Returns all nodes in the path starting at the root and ending at self"""
    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor) -> bool:
        """Is this tensor an output of a node in this path"""
    def _is_alias_of_live_recorded_tensor(self, t: torch.Tensor) -> PathOutputIndex | None: ...
    @staticmethod
    def _check_liveness(indices: list[PathOutputIndex], output_refs: list[list[StorageWeakRefWrapper | None]]) -> bool:
        """Check that all of the indices specified are dead references"""
    def add_child(self, function_id: FunctionID, node: CUDAGraphNode) -> None:
        """Adds node as a a child of self"""
    @staticmethod
    def _get_different_indices(prev: list[list[bool]], curr: list[list[bool]]) -> list[PathOutputIndex]:
        """Find indices where the two lists differ."""
    @staticmethod
    def _get_liveness(weakrefs: list[list[StorageWeakRefWrapper | None]]) -> list[list[bool]]:
        """Maps weakrefs to true if the reference is alive and false otherwise"""
    def debug_assert_invariants(self, expected_liveness: list[list[bool]], newly_dead: list[PathOutputIndex]) -> None: ...
    def debug_check_invariants_before_invocation(self) -> None: ...
    def debug_check_invariants_after_invocation(self) -> None: ...
    def data_ptrs_dead_since_invocation(self) -> list[int]:
        """
        Since this node was invoked, return data ptrs of all tensor outputs that have died
        in the current executing tree path.
        """
    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]: ...
    def remove_node_cached_tensors(self) -> None: ...
    def remove_path_cached_tensors(self) -> None: ...
    def clear_path_state(self) -> None:
        """Clear the path state in this current executing node"""
    @staticmethod
    def _tensor_metadata(x: torch.Tensor, ignore_storage_offset: bool = True) -> dict[str, Any]: ...
    def _reconstruct_from_tensor_metadata(self, metadata: dict[str, Any], storage: UntypedStorage | None = None) -> Tensor: ...
    def create_storage(self, metadata: dict[str, Any]) -> torch.types.Storage: ...
    def _allocate_and_copy_recording_inputs(self, inputs: list[InputType]) -> list[InputType]:
        """
        Allocate inputs for non static, non cudagraph managed tensors in the memory pool
        and copy over the tensor values.
        """
    def check_invariants(self, inputs: list[InputType]) -> tuple[CheckInvariantStatus, Callable[..., str]]:
        """
        Checks if this node can be run. The same pattern of tensor liveness, static inputs,
        and tensors managed in the cudagraph private pool must remain stable.
        """
    def num_descendants(self) -> int:
        """Total number of descendents of this node"""

def get_cudagraph_segments(pool_id: tuple[int, int]) -> Any: ...
def get_block_addrs(pool_id: tuple[int, int], live_only: bool = True) -> list[int]: ...
def format_tb(frames: list[Any]) -> str: ...
def check_memory_pool(device: int, pool_id: tuple[int, int], live_storages_ptrs: list[StorageWeakRefWrapper]) -> None: ...

class ExecutionState(Enum):
    """
    Represents the state of the CUDAGraph Tree. Will be None if there is no live current memory allocated
    in the cuda graph pool. Otherwise will reflect the state of the most recently executed node.
    """
    NONE = ...
    WARMUP = ...
    RECORDING = ...
    EXECUTION = ...

class CompilationMode(Enum):
    FORWARD = ...
    BACKWARD = ...
    INFERENCE = ...

class CUDAGraphTreeManager:
    """
    Groups individual recordings or executions of cuda graphs into a tree of recordings,
    and checks required invariants, and manages warmups of graphs.

    When graphs are recorded in the same tree, it enforces subsequent execution
    to follow the same order and have the same output tensor livespans. To remove
    unnecessary coupling of cuda graphs (and additional imposed invariants),
    the tree manager will end a currently recording tree whenever it is valid - when
    the memory pool no longer has any live allocations.

    We ignore outputs from a previous generation that correspond to prior model outputs.
    Currently this is hardcoded `GenerationTracker.generation` tracked in torch dynamo.
    # TODO: make generation increment configurable, warn on overwrite.

    We run graph warmups in the cudagraph memory pool and return the result on the first invocation
    of a function. For many models it is important to reclaim activations as you run the backward.
    If we were to warm up the model and keep an extra copy of the inputs around to subsequently
    use for recording, we would incur a memory penalty. Additionally, if we are part way through training
    your model and need to recompile, memory will be allocated to the cuda graph pool, so we run this
    warmup run in the cuda graph memory pool. As for recording, warm up needs the state of live tensors
    to be accurately reflected so we checkpoint the allocator state if we need to warm up following graph
    replay.
    """
    roots: dict[FunctionID, list[CUDAGraphNode]]
    ids_to_funcs: dict[FunctionID, WrappedFunction]
    ids_to_stack_traces: dict[FunctionID, StackTraces | None]
    warmed_up_functions: OrderedSet[FunctionID]
    warned_functions: OrderedSet[FunctionID]
    warned_mutation: OrderedSet[FunctionID]
    stream: Incomplete
    graph: torch.cuda.CUDAGraph | None
    cuda_graphs_thread_pool: Incomplete
    graph_counter: Incomplete
    func_counter: Incomplete
    non_cudagraph_managed_mutation_hint: dict[GraphID | None, dict[FunctionID, bool]]
    warmup_node_counter: Incomplete
    num_rerecord: dict[GraphID | None, dict[FunctionID, int]]
    path_state: Incomplete
    device_index: Incomplete
    current_gen: int
    debug_fail_counter: int
    debug_checkpointing_counter: int
    id_to_mode: dict[FunctionID, CompilationMode]
    id_to_compile_id: dict[FunctionID, CompileId | None]
    running_forwards_with_pending_backwards: bool
    mode: CompilationMode | None
    disable_invalidate_aliases: Incomplete
    def __init__(self, device_index: int) -> None: ...
    compile_id: Incomplete
    def run(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType: ...
    def set_to_running_backward(self) -> None: ...
    def _get_cuda_graph_recorded_tensor_checker(self) -> Callable[[Tensor], bool]: ...
    def new_warmup_node_id(self) -> GraphID: ...
    def _update_non_cudagraph_managed_mutation(self, function_id: FunctionID, inputs: list[InputType]) -> None: ...
    def _get_node_id(self) -> GraphID | None: ...
    def exceed_rerecord_limit(self, node_id: GraphID | None, function_id: FunctionID) -> bool: ...
    def _run(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType: ...
    def shutdown(self) -> None:
        """
        Remove all cached tensors in all nodes. Because cached tensors can hold gradients which in turn
        might reference a backward which invokes a CUDA Graph Node, we have to manually clear them on shutdown
        to avoid a reference cycle.
        """
    def record_function(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType: ...
    def execute_node(self, node: CUDAGraphNode, new_inputs: list[InputType]) -> OutputType: ...
    def run_eager(self, new_inputs: list[InputType], function_id: FunctionID) -> OutputType: ...
    def new_graph_id(self) -> GraphID: ...
    def new_func_id(self) -> FunctionID: ...
    def add_function(self, model: ModelType, inputs: list[InputType], static_input_idxs: Sequence[int], stack_traces: StackTraces | None, mode: CompilationMode, constants: tuple[torch.Tensor, ...], placeholders: tuple[PlaceholderInfo, ...], mutated_input_idxs: tuple[int, ...], compile_id: CompileId | None) -> tuple[ModelType, OutputType]: ...
    @property
    def in_recording(self) -> bool: ...
    @property
    def in_warmup(self) -> bool: ...
    def get_roots(self) -> Iterator[CUDAGraphNode]: ...
    @property
    def current_node(self) -> CUDAGraphNode | CUDAWarmupNode | None: ...
    _current_node: Incomplete
    @current_node.setter
    def current_node(self, value: CUDAGraphNode | CUDAWarmupNode | None) -> None: ...
    def update_generation(self) -> None: ...
    @staticmethod
    def get_curr_generation() -> int: ...
    @staticmethod
    def user_invoked_mark_step() -> bool: ...
    def can_start_new_generation(self) -> bool: ...
    def in_new_torch_compile_invocation(self) -> bool: ...
    def try_end_curr_recording(self, function_id: FunctionID) -> None:
        """
        Check if the current recording can be terminated, either because all outputs of the
        previously recorded node are dead or because it was executed in a different
        generation. Will set current_node to None and in_recording to False if successful.
        """
    def try_end_curr_execution(self) -> None:
        """
        Check if the current executing node can be terminated, either because all outputs of the
        previously executed node are dead or because it was executed in a different generation.
        Will set current_node to None if successful.
        """
    def try_end_curr_warmup(self, function_id: FunctionID) -> None: ...
    def check_warn_on_unable_to_start_executing(self, function_id: FunctionID) -> None:
        """Warn if we in a potential loop where we are unable to hit fast path"""
    @staticmethod
    def format_dealloc_msg(stack_trace: str | None) -> str: ...
    def dealloc_current_path_weakrefs(self) -> None: ...
    def clear_current_path_state_and_set_to_none(self) -> None: ...
    def apply_checkpoint_execution_state_in_allocator(self) -> None:
        """
        Checkpoint the current execution state in the caching allocator so that
        additional cudagraph recordings can be made respecting existent live storages.
        """
    def live_cudagraph_pool_storages_in_curr_execution(self) -> list[StorageWeakRefPointer]: ...
