import abc
import dataclasses
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from torch.utils._appending_byte_serializer import AppendingByteSerializer as AppendingByteSerializer, BytesReader as BytesReader, BytesWriter as BytesWriter
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any

log: Incomplete

@dataclasses.dataclass(frozen=True)
class CacheArtifact(ABC, metaclass=abc.ABCMeta):
    """
    Data for each cache artifact that will be serialized and deserialized
    """
    key: str
    content: bytes = dataclasses.field(repr=False)
    @staticmethod
    def serialize(writer: BytesWriter, cls: CacheArtifact) -> None: ...
    @staticmethod
    def deserialize(artifact_type: str, reader: BytesReader) -> CacheArtifact: ...
    @staticmethod
    def encode(content: Any) -> bytes: ...
    @abstractmethod
    def populate_cache(self) -> None: ...
    def precompile_compatible(self) -> bool: ...
    @staticmethod
    def type() -> str:
        """
        Returns the type of the artifact. Must be unique across all CacheArtifact classes.

        CacheArtifactFactory.register will add property method to CacheInfo based on this (def {type}_artifacts)
        that returns all artifacts for specific cache.
        """

class CacheArtifactFactory:
    """
    Factory for creating CacheArtifact objects based on their type
    """
    _artifact_types: dict[str, type[CacheArtifact]]
    @classmethod
    def register(cls, artifact_cls: type[CacheArtifact]) -> type[CacheArtifact]: ...
    @classmethod
    def _get_artifact_type(cls, artifact_type_key: str) -> type[CacheArtifact]: ...
    @classmethod
    def create(cls, artifact_type_key: str, key: str, content: bytes) -> CacheArtifact: ...
    @classmethod
    def encode_create(cls, artifact_type_key: str, key: str, content: Any) -> CacheArtifact: ...

@dataclasses.dataclass
class CacheInfo:
    """
    Return value of serialization and deserialization for the purpose of
    instrumentation
    """
    artifacts: defaultdict[str, list[str]] = dataclasses.field(default_factory=Incomplete)
    @property
    def inductor_artifacts(self) -> list[str]: ...
    @property
    def autotune_artifacts(self) -> list[str]: ...
    @property
    def aot_autograd_artifacts(self) -> list[str]: ...
    @property
    def pgo_artifacts(self) -> list[str]: ...
    @property
    def precompile_aot_autograd_artifacts(self) -> list[str]: ...
    def add(self, artifact: CacheArtifact) -> None: ...
    def clear(self) -> None: ...
    def empty(self) -> bool: ...

def _serialize_single_cache(writer: BytesWriter, cls: tuple[str, list[CacheArtifact]]) -> None: ...
def _deserialize_single_cache(reader: BytesReader) -> tuple[str, list[CacheArtifact]]: ...
CacheArtifactsResult = dict[str, list[CacheArtifact]]

class CacheArtifactManager:
    """
    Lightweight manager class for collecting and processing cache artifacts for
    hot loading

    Intended Lifecycle:
    - Execute code via torch.compile, this will call
        CacheArtifactManager.record_artifact on each cache artifact
    - Call CacheArtifactManager.serialize to convert all the cache artifacts
        to portable format
    - Call CacheArtifactManager.deserialize to hot load the cache artifacts on
        a potentially different process

    NOTE: There's no FB/FC guarentees, results of cache artifacts will not be
          used unless code version matches.
    """
    _new_cache_artifacts: CacheArtifactsResult
    _seen_artifacts: OrderedSet[CacheArtifact]
    _serializer: AppendingByteSerializer[tuple[str, list[CacheArtifact]]]
    _cache_info: CacheInfo
    @classmethod
    def clear(cls) -> None: ...
    @classmethod
    @contextmanager
    def with_fresh_cache(cls) -> Generator[None, None, None]: ...
    @classmethod
    def record_artifact(cls, artifact_type: str, key: str, content: Any) -> None:
        '''
        Called from each caching operation to record the artifact in this
        "mega" list
        '''
    @classmethod
    def need_serialize(cls) -> bool:
        """
        Have we seen new artifacts since last serialize call?
        """
    @classmethod
    def serialize(cls) -> tuple[bytes, CacheInfo] | None:
        '''
        Converts the "mega" list into portable format
        '''
    @staticmethod
    def deserialize(serialized_artifacts: bytes) -> CacheArtifactsResult | None:
        """
        Converts the portable format back into CacheArtifacts
        """
    @staticmethod
    def populate_caches(artifacts: CacheArtifactsResult) -> CacheInfo: ...
    @classmethod
    def _ensure_cache_artifacts_registered(cls) -> None:
        """When deserializing caches in fresh process, we need to ensure that all
        cache artifacts are registered in the cache registry. This is done by
        simply importing all the cache artifacts already wrapped with register call.
        """
