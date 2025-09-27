import abc
from abc import abstractmethod
from torch.compiler._cache import CacheArtifact as CacheArtifact, CacheArtifactFactory as CacheArtifactFactory, CacheArtifactManager as CacheArtifactManager, CacheArtifactsResult as CacheArtifactsResult, CacheInfo as CacheInfo, _serialize_single_cache as _serialize_single_cache
from torch.utils._appending_byte_serializer import AppendingByteSerializer as AppendingByteSerializer
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any, Generic, TypeVar
from typing_extensions import override

T = TypeVar('T')

class PrecompileCacheArtifact(CacheArtifact, Generic[T], metaclass=abc.ABCMeta):
    """
    Data for each cache artifact that will be serialized and deserialized by
    PrecompileContext, rather than CacheArtifactManager.
    T represents the deserialized type of the artifact, i.e. the return type of after_deserialization

    PrecompileCacheArtifact is a frozen dataclass - you can add new serializable fields and metadata specific to your own artifacts
    as needed, and use them in after_deserialization.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """
    @override
    def populate_cache(self) -> None: ...
    @override
    def precompile_compatible(self) -> bool: ...
    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """

class PrecompileContext(CacheArtifactManager):
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact
     - CodeStateArtifact (from torch._dynamo.package once available)
    """
    _new_cache_artifacts_by_key: dict[str, CacheArtifact]
    _new_cache_artifacts: CacheArtifactsResult
    _seen_artifacts: OrderedSet[CacheArtifact]
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]]
    _cache_info: CacheInfo
    @classmethod
    def clear(cls) -> None: ...
    @override
    @classmethod
    def record_artifact(cls, artifact_type: str, key: str, content: Any) -> None:
        '''
        Called from each caching operation to record the artifact in this
        "mega" list
        '''
    @classmethod
    def _save_artifacts_by_type(cls) -> None:
        """
        We normally record artifacts by key, but serialization expects them to be organized
        by artifact type. This function transfers artifacts from _new_cache_artifacts_by_key to _new_cache_artifacts
        """
    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> CacheArtifact | None:
        """
        Serialize all artifacts with the given key returned in a list.
        """
    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None: ...
    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo: ...
    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None: ...
