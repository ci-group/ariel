import os
import threading
import torch
import torch._weights_only_unpickler as _weights_only_unpickler
import types
from _typeshed import Incomplete
from collections.abc import Generator
from contextlib import contextmanager
from enum import Enum
from torch.types import FileLike, Storage
from typing import Any, Callable, Generic, IO, TypeVar
from typing_extensions import TypeAlias

__all__ = ['SourceChangeWarning', 'mkdtemp', 'register_package', 'check_module_version_greater_or_equal', 'validate_cuda_device', 'validate_hpu_device', 'location_tag', 'default_restore_location', 'normalize_storage_type', 'storage_to_tensor_type', 'save', 'load', 'StorageType', 'LoadEndianness', 'get_crc32_options', 'set_crc32_options', 'get_default_load_endianness', 'set_default_load_endianness', 'get_default_mmap_options', 'set_default_mmap_options', 'clear_safe_globals', 'get_safe_globals', 'add_safe_globals', 'safe_globals', 'get_unsafe_globals_in_checkpoint', 'skip_data']

MAP_LOCATION: TypeAlias = Callable[[Storage, str], Storage] | torch.device | str | dict[str, str] | None
STORAGE: TypeAlias = Storage | torch.storage.TypedStorage | torch.UntypedStorage

class _SerializationLocal(threading.local):
    map_location: MAP_LOCATION | None
    skip_data: bool
    materialize_fake_tensors: bool
    def __init__(self) -> None: ...

class SourceChangeWarning(Warning): ...

@contextmanager
def mkdtemp() -> Generator[Incomplete]: ...

class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3

def get_default_load_endianness() -> LoadEndianness | None:
    '''
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it\'s "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    '''
def set_default_load_endianness(endianness) -> None:
    '''
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it\'s "native" byte order.

    Args:
        endianness: the new fallback byte order
    '''
def get_crc32_options() -> bool:
    """
    Get whether :func:`torch.save` computes and writes crc32 for each record.

    Defaults to ``True``.
    """
def set_crc32_options(compute_crc32: bool):
    """
    Set whether :func:`torch.save` computes and writes crc32 for each record.

    .. note::
        Setting this to ``False`` may make unzipping of the ``torch.save`` output
        fail or warn due to corrupted CRC32. However ``torch.load`` will be
        able to load the file.

    Args:
        compute_crc32 (bool): set crc32 compuation flag
    """
def get_default_mmap_options() -> int | None:
    """
    Get default mmap options for :func:`torch.load` with ``mmap=True``.

    Defaults to ``mmap.MAP_PRIVATE``.


    Returns:
        default_mmap_options: int
    """

class set_default_mmap_options:
    """
    Context manager or function to set default mmap options for :func:`torch.load` with ``mmap=True`` to flags.

    For now, only either ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED`` are supported.
    Please open an issue if you need any other option to be added here.

    .. note::
        This feature is currently not supported for Windows.

    Args:
        flags: ``mmap.MAP_PRIVATE`` or ``mmap.MAP_SHARED``
    """
    prev: Incomplete
    def __init__(self, flags: int) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None: ...

def clear_safe_globals() -> None:
    """
    Clears the list of globals that are safe for ``weights_only`` load.
    """
def get_safe_globals() -> list[Callable | tuple[Callable, str]]:
    """
    Returns the list of user-added globals that are safe for ``weights_only`` load.
    """
def add_safe_globals(safe_globals: list[Callable | tuple[Callable, str]]) -> None:
    '''
    Marks the given globals as safe for ``weights_only`` load. For example, functions
    added to this list can be called during unpickling, classes could be instantiated
    and have state set.

    Each item in the list can either be a function/class or a tuple of the form
    (function/class, string) where string is the full path of the function/class.

    Within the serialized format, each function is identified with its full
    path as ``{__module__}.{__qualname__}``. When calling this API, you can provide this
    full path that should match the one in the checkpoint otherwise the default
    ``{fn.__module__}.{fn.__qualname__}`` will be used.

    Args:
        safe_globals (List[Union[Callable, Tuple[Callable, str]]]): list of globals to mark as safe

    Example:
        >>> # xdoctest: +SKIP("Can\'t torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    '''

class safe_globals(_weights_only_unpickler._safe_globals):
    '''Context-manager that adds certain globals as safe for ``weights_only`` load.

    Args:
        safe_globals: List of globals for weights_only load.

    Example:
        >>> # xdoctest: +SKIP("Can\'t torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # Running `torch.load(f.name, weights_only=True)` will fail with
        # Unsupported global: GLOBAL __main__.MyTensor was not an allowed global by default.
        # Check the code and make sure MyTensor is safe to be used when loaded from an arbitrary checkpoint.
        ...     with torch.serialization.safe_globals([MyTensor]):
        ...         torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
        >>> assert torch.serialization.get_safe_globals() == []
    '''

def get_unsafe_globals_in_checkpoint(f: FileLike) -> list[str]:
    """Returns a list of strings of functions/classes in a ``torch.save`` object that are not safe for ``weights_only``.

    For a given function or class ``f``, the corresponding string will be of the form
    ``{f.__module__}.{f.__name__}``.

    This function will return any GLOBALs in the checkpoint that are not in the set marked safe
    for ``weights_only`` (either via :func:`add_safe_globals` or :class:`safe_globals` context or
    allowlisted by ``torch`` by default).

    .. note::
        This function will statically disassemble the pickle file in the checkpoint.
        The implication is any classes dynamically pushed onto the stack during unpickling
        will not be included in the output.

    Args:
        f: File-like object or string containing the checkpoint object saved via ``torch.save``

    Returns:
        A list of strings of pickle GLOBALs in the checkpoint that are not allowlisted for ``weights_only``.
    """

class skip_data:
    '''
    Context-manager that skips writing/reading storage bytes for ``torch.save`` / ``torch.load`` calls.

    For the save path, storages will still be saved, but the space that their bytes would usually be written to
    will be empty space. The storage bytes can then be populated in a separate pass.

    For the load path, tensors will be loaded per the checkpoint but their storages will not be populated with data.

    .. warning::
        The ``skip_data`` context manager is an early prototype and is subject to change.

    Args:
        materialize_fake_tensors: Whether to materialize FakeTensors during save. This is a no-op for the load path.

    Example:
        >>> # xdoctest: +SKIP("NamedTemporaryFile on Windows")
        >>> import tempfile
        >>> t = torch.randn(2, 3)
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     with torch.serialization.skip_data():
        ...         torch.save(t, f.name)
        ...     torch.load(f.name, weights_only=True)
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
    '''
    materialize_fake_tensors: Incomplete
    def __init__(self, materialize_fake_tensors: bool = False) -> None: ...
    _old_skip_data: Incomplete
    _old_materialize_fake_tensors: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, tb: types.TracebackType | None) -> None: ...

def register_package(priority: int, tagger: Callable[[STORAGE], str | None], deserializer: Callable[[STORAGE, str], STORAGE | None]):
    '''
    Registers callables for tagging and deserializing storage objects with an associated priority.
    Tagging associates a device with a storage object at save time while deserializing moves a
    storage object to an appropriate device at load time. :attr:`tagger` and :attr:`deserializer`
    are run in the order given by their :attr:`priority` until a tagger/deserializer returns a
    value that is not `None`.

    To override the deserialization behavior for a device in the global registry, one can register a
    tagger with a higher priority than the existing tagger.

    This function can also be used to register a tagger and deserializer for new devices.

    Args:
        priority: Indicates the priority associated with the tagger and deserializer, where a lower
            value indicates higher priority.
        tagger: Callable that takes in a storage object and returns its tagged device as a string
            or None.
        deserializer: Callable that takes in storage object and a device string and returns a storage
            object on the appropriate device or None.

    Returns:
        `None`

    Example:
        >>> def ipu_tag(obj):
        >>>     if obj.device.type == \'ipu\':
        >>>         return \'ipu\'
        >>> def ipu_deserialize(obj, location):
        >>>     if location.startswith(\'ipu\'):
        >>>         ipu = getattr(torch, "ipu", None)
        >>>         assert ipu is not None, "IPU device module is not loaded"
        >>>         assert torch.ipu.is_available(), "ipu is not available"
        >>>         return obj.ipu(location)
        >>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
    '''
def check_module_version_greater_or_equal(module, req_version_tuple, error_if_malformed: bool = True):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
def validate_cuda_device(location): ...
def validate_hpu_device(location): ...
def location_tag(storage: Storage | torch.storage.TypedStorage | torch.UntypedStorage): ...
def default_restore_location(storage, location):
    """
    Restores `storage` using a deserializer function registered for the `location`.

    This function looks in the registry for deserializer functions that match the `location`.
    If found, it attempts to use them, in priority order, to restore `storage` until one
    returns a not `None` result. If no deserializer can be found in the registry, or all found fail
    to bear a result, it raises a `RuntimeError`.

    Args:
        storage (STORAGE): the storage object to restore
        location (str): the location tag associated with the storage object

    Returns:
        storage: Optional[STORAGE]

    Raises:
        RuntimeError: If no deserializer matching `location` is found in the registry or if
           all matching ones return `None`.
    """
def normalize_storage_type(storage_type): ...
def storage_to_tensor_type(storage): ...
T = TypeVar('T')

class _opener(Generic[T]):
    file_like: T
    def __init__(self, file_like: T) -> None: ...
    def __enter__(self): ...
    def __exit__(self, *args) -> None: ...

class _open_file(_opener[IO[bytes]]):
    def __init__(self, name: str | os.PathLike[str], mode: str) -> None: ...
    def __exit__(self, *args) -> None: ...

class _open_buffer_reader(_opener[IO[bytes]]):
    def __init__(self, buffer: IO[bytes]) -> None: ...

class _open_buffer_writer(_opener[IO[bytes]]):
    def __exit__(self, *args) -> None: ...

class _open_zipfile_reader(_opener[torch._C.PyTorchFileReader]):
    def __init__(self, name_or_buffer: str | IO[bytes]) -> None: ...

class _open_zipfile_writer_file(_opener[torch._C.PyTorchFileWriter]):
    file_stream: Incomplete
    name: Incomplete
    def __init__(self, name: str) -> None: ...
    def __exit__(self, *args) -> None: ...

class _open_zipfile_writer_buffer(_opener[torch._C.PyTorchFileWriter]):
    buffer: Incomplete
    def __init__(self, buffer: IO[bytes]) -> None: ...
    def __exit__(self, *args) -> None: ...

def save(obj: object, f: FileLike, pickle_module: Any = ..., pickle_protocol: int = ..., _use_new_zipfile_serialization: bool = True, _disable_byteorder_record: bool = False) -> None:
    '''save(obj, f, pickle_module=pickle, pickle_protocol=2, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    See :ref:`layout-control` for more advanced tools to manipulate a checkpoint.

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    .. note::
        PyTorch preserves storage sharing across serialization. See
        :ref:`preserve-storage-sharing` for more details.

    .. note::
        The 1.6 release of PyTorch switched ``torch.save`` to use a new
        zipfile-based file format. ``torch.load`` still retains the ability to
        load files in the old format. If for any reason you want ``torch.save``
        to use the old format, pass the kwarg ``_use_new_zipfile_serialization=False``.

    Example:
        >>> # xdoctest: +SKIP("makes cwd dirty")
        >>> # Save to file
        >>> x = torch.tensor([0, 1, 2, 3, 4])
        >>> torch.save(x, "tensor.pt")
        >>> # Save to io.BytesIO buffer
        >>> buffer = io.BytesIO()
        >>> torch.save(x, buffer)
    '''
def load(f: FileLike, map_location: MAP_LOCATION = None, pickle_module: Any = None, *, weights_only: bool | None = None, mmap: bool | None = None, **pickle_load_args: Any) -> Any:
    '''load(f, map_location=None, pickle_module=pickle, *, weights_only=True, mmap=None, **pickle_load_args)

    Loads an object saved with :func:`torch.save` from a file.

    :func:`torch.load` uses Python\'s unpickling facilities but treats storages,
    which underlie tensors, specially. They are first deserialized on the
    CPU and are then moved to the device they were saved from. If this fails
    (e.g. because the run time system doesn\'t have certain devices), an exception
    is raised. However, storages can be dynamically remapped to an alternative
    set of devices using the :attr:`map_location` argument.

    If :attr:`map_location` is a callable, it will be called once for each serialized
    storage with two arguments: storage and location. The storage argument
    will be the initial deserialization of the storage, residing on the CPU.
    Each serialized storage has a location tag associated with it which
    identifies the device it was saved from, and this tag is the second
    argument passed to :attr:`map_location`. The builtin location tags are ``\'cpu\'``
    for CPU tensors and ``\'cuda:device_id\'`` (e.g. ``\'cuda:2\'``) for CUDA tensors.
    :attr:`map_location` should return either ``None`` or a storage. If
    :attr:`map_location` returns a storage, it will be used as the final deserialized
    object, already moved to the right device. Otherwise, :func:`torch.load` will
    fall back to the default behavior, as if :attr:`map_location` wasn\'t specified.

    If :attr:`map_location` is a :class:`torch.device` object or a string containing
    a device tag, it indicates the location where all tensors should be loaded.

    Otherwise, if :attr:`map_location` is a dict, it will be used to remap location tags
    appearing in the file (keys), to ones that specify where to put the
    storages (values).

    User extensions can register their own location tags and tagging and
    deserialization methods using :func:`torch.serialization.register_package`.

    See :ref:`layout-control` for more advanced tools to manipulate a checkpoint.

    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
        weights_only: Indicates whether unpickler should be restricted to
            loading only tensors, primitive types, dictionaries
            and any types added via :func:`torch.serialization.add_safe_globals`.
            See :ref:`weights-only` for more details.
        mmap: Indicates whether the file should be mmaped rather than loading all the storages into memory.
            Typically, tensor storages in the file will first be moved from disk to CPU memory, after which they
            are moved to the location that they were tagged with when saving, or specified by ``map_location``. This
            second step is a no-op if the final location is CPU. When the ``mmap`` flag is set, instead of copying the
            tensor storages from disk to CPU memory in the first step, ``f`` is mmaped, which means tensor storages
            will be lazily loaded when their data is accessed.
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.

    .. warning::
        :func:`torch.load()` unless `weights_only` parameter is set to `True`,
        uses ``pickle`` module implicitly, which is known to be insecure.
        It is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never load data that could have come from an untrusted
        source in an unsafe mode, or that could have been tampered with. **Only load data you trust**.

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location=\'cpu\')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: \'ascii\' codec can\'t decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding=\'latin1\'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding=\'bytes\'` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.

    Example:
        >>> # xdoctest: +SKIP("undefined filepaths")
        >>> torch.load("tensors.pt", weights_only=True)
        # Load all tensors onto the CPU
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=torch.device("cpu"),
        ...     weights_only=True,
        ... )
        # Load all tensors onto the CPU, using a function
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=lambda storage, loc: storage,
        ...     weights_only=True,
        ... )
        # Load all tensors onto GPU 1
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location=lambda storage, loc: storage.cuda(1),
        ...     weights_only=True,
        ... )  # type: ignore[attr-defined]
        # Map tensors from GPU 1 to GPU 0
        >>> torch.load(
        ...     "tensors.pt",
        ...     map_location={"cuda:1": "cuda:0"},
        ...     weights_only=True,
        ... )
        # Load tensor from io.BytesIO object
        # Loading from a buffer setting weights_only=False, warning this can be unsafe
        >>> with open("tensor.pt", "rb") as f:
        ...     buffer = io.BytesIO(f.read())
        >>> torch.load(buffer, weights_only=False)
        # Load a module with \'ascii\' encoding for unpickling
        # Loading from a module setting weights_only=False, warning this can be unsafe
        >>> torch.load("module.pt", encoding="ascii", weights_only=False)
    '''

class StorageType:
    _dtype: Incomplete
    def __init__(self, name) -> None: ...
    @property
    def dtype(self): ...
    def __str__(self) -> str: ...
