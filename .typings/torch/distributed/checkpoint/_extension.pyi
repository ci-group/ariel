import abc
from _typeshed import Incomplete
from collections.abc import Sequence
from typing import IO

__all__ = ['Extension', 'StreamTransformExtension', 'ZStandard', 'ExtensionRegistry']

class Extension(abc.ABC, metaclass=abc.ABCMeta):
    """
    Extensions provide modular additions to functionality within distributed checkpointing,
    which affect the layout or format of the written artifacts.  Extensions may be
    built into pytorch, or provided externally.

    When writing, the caller provides a list of extension instances of the appropriate
    type.  Each extension can output a descriptor which is used to reconstitute the
    extension at read-time.
    """
    @staticmethod
    @abc.abstractmethod
    def registry_name() -> str:
        """
        See ExtensionRegistry.from_descriptor_list
        """
    @staticmethod
    @abc.abstractmethod
    def from_descriptor(version: str) -> Extension:
        """
        See ExtensionRegistry.from_descriptor_list
        """
    @abc.abstractmethod
    def get_descriptor(self) -> str:
        '''
        Return descriptor name to be included in metadata.  The form should be
        "extension_name[@local-domain][/version]".
        '''

class StreamTransformExtension(Extension, metaclass=abc.ABCMeta):
    """
    An extension which performs transformation on a byte stream, such as compression
    or encryption.

    Implementations should try to be memory friendly and performant.  For example, don't
    read the whole input, then transform it, and write it back.  If at all possible, do it in
    chunks.  But, don't read/transform/write one byte at a time, either.
    """
    @abc.abstractmethod
    def transform_to(self, output: IO[bytes]) -> IO[bytes]:
        """
        Takes a writeable output stream, and generates a new stream which implements the
        output transform.  Input data written to the returned stream will be transformed
        and written to the `output` argument stream.
        """
    @abc.abstractmethod
    def transform_from(self, input: IO[bytes]) -> IO[bytes]:
        """
        Takes a readable input stream, and generates a new stream which implements the
        input transform.  When the returned stream is read, data will be read from the
        'input' stream, transformed, and returned.
        """

class ZStandard(StreamTransformExtension):
    @staticmethod
    def is_available() -> bool: ...
    @staticmethod
    def from_descriptor(version: str) -> ZStandard: ...
    @staticmethod
    def registry_name() -> str: ...
    def __init__(self) -> None: ...
    def get_descriptor(self) -> str: ...
    output: Incomplete
    compressor: Incomplete
    def transform_to(self, output: IO[bytes]) -> IO[bytes]: ...
    input: Incomplete
    decompressor: Incomplete
    def transform_from(self, input: IO[bytes]) -> IO[bytes]: ...

class ExtensionRegistry:
    extensions: dict[str, type[Extension]]
    def __init__(self) -> None: ...
    def register(self, cls: type[Extension]) -> None: ...
    def from_descriptor_list(self, descriptors: Sequence[str]) -> Sequence[Extension]:
        """
        Given a seuquence of descriptor strings as returned by
        Extension.get_descriptor at save time, creates a sequence of
        Extension instances.  The name[@local-domain] preceding the
        version number is used to look up an implementation class in
        the registry, and the version is passed to the class's
        from_descriptor static method.  If the registry contains no
        match, this will throw ValueError.  If the from_descriptor
        method raises an exception, that will pass through to the
        caller.
        """
