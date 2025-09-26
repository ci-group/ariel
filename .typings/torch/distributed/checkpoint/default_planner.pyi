import io
import torch
from _typeshed import Incomplete
from torch.distributed.checkpoint._nested_dict import FLATTEN_MAPPING
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, STATE_DICT_TYPE, StorageMeta
from torch.distributed.checkpoint.planner import LoadPlan, LoadPlanner, ReadItem, SavePlan, SavePlanner, WriteItem
from typing import Any

__all__ = ['DefaultSavePlanner', 'DefaultLoadPlanner', 'create_default_local_load_plan', 'create_default_global_load_plan', 'create_default_local_save_plan', 'create_default_global_save_plan']

class DefaultSavePlanner(SavePlanner):
    mappings: FLATTEN_MAPPING
    flatten_state_dict: Incomplete
    flatten_sharded_tensors: Incomplete
    dedup_save_to_lowest_rank: Incomplete
    _cached_plans_key: str
    _enable_plan_caching: Incomplete
    def __init__(self, flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, dedup_replicated_tensors: bool | None = None, dedup_save_to_lowest_rank: bool = False, enable_plan_caching: bool = False) -> None: ...
    state_dict: Incomplete
    is_coordinator: Incomplete
    def set_up_planner(self, state_dict: STATE_DICT_TYPE, storage_meta: StorageMeta | None = None, is_coordinator: bool = False) -> None: ...
    plan: Incomplete
    def create_local_plan(self) -> SavePlan: ...
    def _dedup_save_plans(self, all_plans: list[SavePlan]) -> list[SavePlan]: ...
    def _create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]: ...
    def _create_global_plan_with_caching(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], list[SavePlan], Metadata]:
        """
        Create global plan with caching.
        Returns a tuple of global_plan_delta, global_plan, metadata.
        """
    global_plan: Incomplete
    metadata: Incomplete
    def create_global_plan(self, all_plans: list[SavePlan]) -> tuple[list[SavePlan], Metadata]: ...
    def _finish_plan_with_caching(self, new_plan: SavePlan) -> SavePlan: ...
    def finish_plan(self, new_plan: SavePlan) -> SavePlan: ...
    def resolve_data(self, write_item: WriteItem) -> torch.Tensor | io.BytesIO: ...
    def lookup_object(self, index: MetadataIndex) -> Any:
        """Extension from the planner interface to make it easy to extend the default planner."""
    def transform_object(self, write_item: WriteItem, object: Any):
        """Extension from the planner interface to make it easy to extend the default planner."""

class DefaultLoadPlanner(LoadPlanner):
    """
    DefaultLoadPlanner that adds multiple features on top of LoadPlanner.

    In particular it adds the following:

    flatten_state_dict: Handle state_dict with nested dicts
    flatten_sharded_tensors: For FSDP in 2D parallel mode
    allow_partial_load: If False, will raise a runtime error if a key is present in state_dict, but not in the checkpoint.
    """
    original_state_dict: STATE_DICT_TYPE
    mappings: FLATTEN_MAPPING
    flatten_state_dict: Incomplete
    flatten_sharded_tensors: Incomplete
    allow_partial_load: Incomplete
    def __init__(self, flatten_state_dict: bool = True, flatten_sharded_tensors: bool = True, allow_partial_load: bool = False) -> None: ...
    state_dict: Incomplete
    metadata: Incomplete
    is_coordinator: Incomplete
    def set_up_planner(self, state_dict: STATE_DICT_TYPE, metadata: Metadata | None = None, is_coordinator: bool = False) -> None: ...
    def create_local_plan(self) -> LoadPlan: ...
    def create_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]: ...
    def finish_plan(self, new_plan: LoadPlan) -> LoadPlan: ...
    def load_bytes(self, read_item: ReadItem, value: io.BytesIO) -> None: ...
    def resolve_tensor(self, read_item: ReadItem): ...
    def commit_tensor(self, read_item: ReadItem, tensor: torch.Tensor) -> None: ...
    def lookup_tensor(self, index: MetadataIndex) -> torch.Tensor:
        """Extension from the planner interface to make it easy to extend the default planner."""
    def transform_tensor(self, read_item: ReadItem, tensor: torch.Tensor):
        """Extension from the planner interface to make it easy to extend the default planner."""

class _EmptyStateDictLoadPlanner(DefaultLoadPlanner):
    """
    Extension of DefaultLoadPlanner, which rebuilds state_dict from the saved metadata.
    Useful for loading in state_dict without first initializing a model, such as
    when converting a DCP checkpoint into a Torch save file.

    . N.B. `state_dict` must be an empty dictionary when used with this LoadPlanner

    .. warning::
        Because the entire state dict is initialized, It's recommended to only utilize
        this LoadPlanner on a single rank or process to avoid OOM.

    """
    keys: Incomplete
    def __init__(self, keys=None, *args, **kwargs) -> None: ...
    def _should_include_key(self, key: str, metadata: Metadata) -> bool: ...
    def set_up_planner(self, state_dict: STATE_DICT_TYPE, metadata: Metadata | None = None, is_coordinator: bool = False) -> None: ...

def create_default_local_load_plan(state_dict: dict[str, Any], metadata: Metadata, strict: bool = True) -> LoadPlan: ...
def create_default_global_load_plan(all_plans: list[LoadPlan]) -> list[LoadPlan]:
    """
    Create global load plan used by DefaultLoadPlanner.

    The default load behavior involved no global coordination and this function
    currently doesn't change the local plans.
    """
def create_default_local_save_plan(state_dict: dict[str, Any], is_coordinator: bool) -> SavePlan:
    """
    Create the ``SavePlan`` used by DefaultSavePlanner.

    On non-coordinator ranks, this function ignores tensors and non-tensor objects,
    only producing writes for ShardedTensor objects.

    On the coordinator rank, produce writes for all values.
    """
def create_default_global_save_plan(all_plans: list[SavePlan], rewrite_index_hints: bool = True) -> tuple[list[SavePlan], Metadata]:
    """
    Create the global plan and metadata used by DefaultSavePlanner.

    Metadata is produced by concatenating the metadata of all ``WriteItem`` from the supplied plans.

    The only global planning change is to update index hints in all ``MetadataIndex`` objects if
    ``rewrite_index_hints`` is True.
    """
