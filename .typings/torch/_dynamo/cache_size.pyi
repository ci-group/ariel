from . import config as config
from .types import DynamoFrameType as DynamoFrameType
from _typeshed import Incomplete
from dataclasses import dataclass
from torch._guards import CompileId as CompileId

log: Incomplete

@dataclass
class CacheSizeRelevantForFrame:
    """
    We track the number of cache entries that have same id_match objects as the
    given frame.

    TODO(janimesh) - Consider adding a map from tuple_of_match_ids to count -
    https://github.com/pytorch/pytorch/pull/107496#discussion_r1304564682 - this
    could be useful for debugging as well.
    """
    num_cache_entries: int = ...
    num_cache_entries_with_same_id_matched_objs: int = ...
    def will_compilation_exceed(self, limit: int) -> bool: ...
    def will_compilation_exceed_accumulated_limit(self) -> bool: ...
    def will_compilation_exceed_specific_limit(self, limit: int) -> bool: ...

def _get_weakref_from_f_locals(frame: DynamoFrameType, local_name: str): ...
def _has_same_id_matched_objs(frame: DynamoFrameType, cache_entry) -> bool:
    """
    Checks if the ID_MATCH'd objects saved on cache_entry are same as the ones
    in frame.f_locals.
    """
def compute_cache_size(frame: DynamoFrameType, cache_entry) -> CacheSizeRelevantForFrame: ...
def is_recompilation(cache_size: CacheSizeRelevantForFrame) -> bool:
    """
    If the frame (earlier parsed by compute_cache_size) has more than 1 cache
    entry with same ID_MATCH'd objects, then its a recompilation.
    """
def exceeds_recompile_limit(cache_size: CacheSizeRelevantForFrame, compile_id: CompileId) -> tuple[bool, str]:
    """
    Checks if we are exceeding the cache size limit.
    """
