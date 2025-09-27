import collections
import dataclasses
import enum
import torch
from _typeshed import Incomplete
from collections.abc import Iterator
from torch._C import FunctionSchema as FunctionSchema
from torch._C._autograd import _ProfilerResult as _ProfilerResult
from torch._C._profiler import RecordScope as RecordScope, _EventType as _EventType, _ExtraFields_Allocation as _ExtraFields_Allocation, _ExtraFields_TorchOp as _ExtraFields_TorchOp, _ProfilerEvent as _ProfilerEvent, _TensorMetadata as _TensorMetadata
from torch._utils import _element_size as _element_size
from torch.profiler import _utils as _utils

KeyAndID: Incomplete
TensorAndID: Incomplete
log: Incomplete

class Category(enum.Enum):
    INPUT = ...
    TEMPORARY = ...
    ACTIVATION = ...
    GRADIENT = ...
    AUTOGRAD_DETAIL = ...
    PARAMETER = ...
    OPTIMIZER_STATE = ...

_CATEGORY_TO_COLORS: Incomplete
_CATEGORY_TO_INDEX: Incomplete

class Action(enum.Enum):
    PREEXISTING = ...
    CREATE = ...
    INCREMENT_VERSION = ...
    DESTROY = ...

_ACTION_TO_INDEX: Incomplete

@dataclasses.dataclass(eq=True, unsafe_hash=False, frozen=True)
class Key:
    device: torch.device

@dataclasses.dataclass
class _Storage:
    """Bundle storage pointer and id.

    All profiling logic should use `allocation_id`, however it is useful to
    print storage pointers for debugging and unit tests sometimes look up
    values using the storage data pointer of a live Tensor."""
    ptr: int
    allocation_id: int
    def __repr__(self) -> str: ...
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

@dataclasses.dataclass(eq=True, unsafe_hash=True, frozen=True)
class TensorKey(Key):
    """Hashable identifier for a storage which has been assigned an ID.

    A detailed description of Tensor IDs and why they are needed is given in
    `torch/csrc/profiler/collection.h` when `TensorID` is declared. To
    summarize, multiple Storage buffers can map to the same logical Tensor.
    This dataclass is used to refer to a concrete in-memory StorageImpl of
    a Tensor.
    """
    id: int
    storage: _Storage
    def __repr__(self) -> str: ...
    def __lt__(self, other: TensorKey) -> bool: ...
    @staticmethod
    def _make(tensor_id: int | None, storage_ptr: int | None, allocation_id: int | None, device: torch.device) -> TensorKey | None: ...
    @classmethod
    def from_allocation(cls, alloc: _ExtraFields_Allocation) -> TensorKey | None: ...
    @classmethod
    def from_tensor(cls, t: _TensorMetadata | None) -> TensorKey | None: ...
    @property
    def _as_sortable(self) -> tuple[int, int, str, int]: ...

def _extract_parameters_and_gradients(node: _ProfilerEvent) -> Iterator[tuple[TensorKey | None, TensorKey | None]]: ...
def extract_parameters(node: _ProfilerEvent) -> Iterator[TensorKey]: ...
def extract_gradients(node: _ProfilerEvent) -> Iterator[tuple[TensorKey | None, TensorKey]]: ...
def get_scopes(event: _ProfilerEvent | None) -> tuple[RecordScope, ...]: ...

class SchemaMatcher:
    """Lookup operator schema based on profiled name.

    When profiling we record the operator's name but not the schema. However
    some analysis requires that information. Fortunately we can look up
    registered schema from the recorded name. We do not, however, record the
    overload and so we must compare the profiled arguments with all overloads
    to determine viable matches.

    Note: Once https://github.com/pytorch/pytorch/issues/78871 is completed
    this code will be obsolete.
    """
    @classmethod
    def inputs_are_mutable(cls, t: _ExtraFields_TorchOp) -> tuple[bool | None, ...]:
        """Determine which inputs may have mutated based on function schema.

        Note that we don't need to resolve down to a single schema to perform
        this analysis. An input is mutable if it is mutable in any overload. In
        practice, however, it is overwhelmingly common to match a single
        overload. If we cannot find any valid schema then we must be
        conservative and assume all inputs are mutable.
        """
    @classmethod
    def match_schemas(cls, t: _ExtraFields_TorchOp) -> tuple[FunctionSchema, ...]: ...
    @classmethod
    def _types_match(cls, observed, schema_type) -> bool: ...
    @staticmethod
    def lookup_schemas(name: str) -> tuple[FunctionSchema, ...] | None: ...

class OpTree:
    _root_nodes: Incomplete
    _sorted_nodes: Incomplete
    def __init__(self, result: _ProfilerResult) -> None: ...
    def dfs(self, *args, **kwargs) -> Iterator[_ProfilerEvent]: ...
    @property
    def sorted_nodes(self) -> tuple[_ProfilerEvent, ...]: ...

class SizeMap:
    _values: dict[TensorKey, int]
    def __init__(self, op_tree: OpTree) -> None: ...
    def _update_values(self, t: _TensorMetadata | None) -> None: ...
    @staticmethod
    def _flat_tensor_inputs(op: _ExtraFields_TorchOp) -> Iterator[_TensorMetadata]: ...
    def __getitem__(self, key: TensorKey): ...

@dataclasses.dataclass()
class DataFlowEdge:
    input_version: int | None = ...
    mutated: bool | None = ...
    @property
    def is_allocation(self) -> bool: ...
    @property
    def is_deletion(self) -> bool: ...

class DataFlowNode:
    _event: Incomplete
    _graph: Incomplete
    _edges: dict[TensorKey, DataFlowEdge]
    def __init__(self, event: _ProfilerEvent, graph: DataFlowGraph) -> None: ...
    def _determine_edges(self) -> dict[TensorKey, DataFlowEdge]: ...
    @property
    def inputs(self) -> dict[TensorKey, tuple[bool, int]]: ...
    @property
    def outputs(self) -> dict[TensorKey, int]: ...
    @property
    def intermediates(self) -> tuple[TensorKey, ...]: ...
    @property
    def start_time(self) -> int: ...

class DataFlowGraph:
    _op_tree: Incomplete
    _leaf_events: Incomplete
    _active_version: dict[TensorKey, int | None]
    _flow_nodes: Incomplete
    def __init__(self, op_tree: OpTree) -> None: ...
    @property
    def flow_nodes(self) -> tuple[DataFlowNode, ...]: ...
    def validate(self) -> None: ...
    @property
    def leaf_events(self) -> tuple[_ProfilerEvent, ...]: ...
    @staticmethod
    def _extract_leaf_events(op_tree: OpTree) -> tuple[_ProfilerEvent, ...]:
        '''Partially traverse the op tree and extract top level ops.

        Consider the following code:
        ```
        with record_function("My annotation"):
            x.zero_()
            y.zero_()
        ```

        The op tree (assuming no Autograd) will look like:
          <Python context>
            TorchOp: "My annotation"
              TorchOp: zero_
                TorchOp: fill_
              TorchOp: zero_
                TorchOp: fill_

        The recursive structure of operator calls makes data flow unwieldy.
        In order to simplify analysis we would like to select the highest level
        ops to represent in the graph. In this case those are the `zero_` ops;
        the fact that `fill_` is called is an implementation detail. We also
        do not want to group everything under "My annotation" as this could
        create overly coarse bundles and lose critical semantics.

        To address this issue we walk over the graph and select the topmost
        torch ops ** which match at least one operator schema **. These form
        the leaves of the first pass through the op tree. (As well as any
        allocations or frees which do are not part of a kernel.) These events
        form the logical nodes in our data flow graph.
        '''
    def lookup(self, key: TensorKey) -> int: ...
    def bump(self, key: TensorKey) -> None: ...
    def delete(self, key: TensorKey) -> None: ...

@dataclasses.dataclass
class CategoryElement:
    by_id: Category | None = ...
    by_key: dict[TensorKey, Category] = dataclasses.field(default_factory=dict)
    by_version: dict[TensorAndID, Category] = dataclasses.field(default_factory=dict)
    _by_id_keyset: set[TensorKey] = dataclasses.field(default_factory=set)

@dataclasses.dataclass
class CategoryDict:
    _values: collections.defaultdict[int, CategoryElement] = dataclasses.field(default_factory=Incomplete)
    def set_by_id(self, key: TensorKey, category: Category) -> None: ...
    def set_by_key(self, key: TensorKey, category: Category) -> None: ...
    def set_by_version(self, key: TensorKey, version: int, category: Category) -> None: ...
    def setdefault_by_version(self, key: TensorKey, version: int, category: Category) -> None: ...
    def get(self, key: Key, version: int) -> Category | None: ...

class MemoryProfile:
    _op_tree: Incomplete
    _data_flow_graph: Incomplete
    _size_map: Incomplete
    _categories: Incomplete
    def __init__(self, result: _ProfilerResult) -> None: ...
    @property
    def timeline(self) -> tuple[tuple[int, Action, KeyAndID, int], ...]: ...
    def _is_gradient(self, *args, **kwargs) -> bool: ...
    def _category_snapshot(self) -> dict[TensorAndID, Category | None]: ...
    def _any_version_depends_on_gradient(self) -> set[int]:
        '''Extract IDs of Tensors which depend or will depend on a gradient.

        Note that this weakened definition of "depends" requires us to loop
        over the data flow graph multiple times because it allows dependency
        information to flow backward through edges and removes the guarantee
        that nodes are topologically sorted. (Or indeed, even that a valid
        topological order exists.) Put another way, we have converted an
        acyclic data flow graph into a cyclic graph and we are attempting to
        partition cycles involving a gradient from the rest of the graph.
        '''
    def _set_gradients_and_temporaries(self) -> None:
        """Mark Tensors which are unambiguous and simple to reason about."""
    def _set_parameters_using_python_tracer(self) -> None: ...
    def _set_inputs(self) -> None:
        '''Mark inputs based on which Tensors are updated using gradients.

        The process for differentiating between inputs and activations is more
        involved. Most Tensors in a training loop depend on at least one
        gradient: parameters depend on them through updates, and activations
        and optimizer state depend on them transitively through parameters.
        Critically, we do not need to know which Tensors are parameters to
        apply this method; we can simply walk the data flow graph to build the
        set of all values which depend on a gradient and then obtain the set
        of inputs from the conjugate set.

        There is, however, one hiccup. The first time we see a parameter is
        generally on the forward pass of the first step. We know from
        inspection of the data flow graph that v1 of that Tensor depends on
        a gradient (provided we profile an optimizer step), but not v0. To
        address this problem we weaken the definition of "depends on a
        gradient" to "any version of this Tensor depends on a gradient",
        which in turn strengthens the criteria for the input set enough to
        filter the activations in the forward pass of the first step.'''
    def _set_parameters_using_data_flow(self) -> None:
        """Deduce which Tensors are parameters.

        Consider the following code for the step of SGD with momentum
        (nesterov=False), where `d_p` is the gradient of `param` and `buf` is
        the momentum buffer.
        ```
          buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
          d_p = buf
          param.add_(d_p, alpha=-lr)
        ```
        Both `param` and `buf` take a gradient and perform an in-place update.

        The python tracer will inspect calls to `nn.Module.forward` and
        `optim.Optimizer.step` to extract parameter and optimizer state
        respectively (including parameters), so this is generally a non-issue.

        However as a fallback we can also exploit several properties of
        parameters to distinguish them from other model state.

        First, they are directly used in the forward pass. (At this point we
        haven't established which parts of the graph correspond to the forward
        pass but we can deduce enough to suffice.) Some mutable state such as
        batch norm moving averages also contribute to the forward pass, but
        optimizer state does not.

        Second, a parameter is by definition used to compute at least one
        gradient and depends on at least one gradient.
        """
    def _set_activations(self) -> None:
        """Flood the graph to identify activations."""
    def _set_optimizer_state(self) -> None: ...
    def _set_autograd_detail(self) -> None: ...

class MemoryProfileTimeline:
    timeline: Incomplete
    categories: Incomplete
    def __init__(self, memory_profile) -> None:
        """The minimum representation of the memory profile timeline
        includes the memory timeline and categories. The timeline
        consists of [timestamp, action, (TensorKey, version), numbytes]
        elements, to denote any actions (pre-existing, create, destroy,
        or increment_version) that occurred to a specific Tensor for a
        chunk of memory. The categories help map each (TensorKey,
        version) pair into a category."""
    def _coalesce_timeline(self, device_str):
        """Convert the memory timeline and categories into a memory plot
        consisting of timestamps and their respective sizes by category
        for a given device.

        Input: device
        Output: [timestamps, sizes by category]
        """
    def export_memory_timeline(self, path, device_str) -> None:
        """Saves the memory timeline as [times, sizes by category]
        as a JSON formatted file to the given path for the given
        device."""
    def export_memory_timeline_raw(self, path, device_str) -> None:
        """Saves the memory timeline as raw memory event tuples in the
        form of (timestamp, action, numbytes, category)
        as a JSON formatted file to the given path for the given
        device."""
    def export_memory_timeline_html(self, path, device_str, figsize=(20, 12), title=None) -> None:
        """Exports the memory timeline as an HTML file which contains
        the memory timeline plot embedded as a PNG file."""
