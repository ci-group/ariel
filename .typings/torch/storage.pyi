import functools
import torch
from _typeshed import Incomplete
from torch._prims_common import DeviceLikeType
from torch.types import _bool, _int
from typing import Any, TypeVar
from typing_extensions import Self

__all__ = ['TypedStorage', 'UntypedStorage']

T = TypeVar('T', bound='Union[_StorageBase, TypedStorage]')

class _StorageBase:
    _cdata: Any
    is_sparse: _bool
    is_sparse_csr: _bool
    device: torch.device
    _fake_device: torch.device | None
    _checkpoint_offset: int | None
    def __init__(self, *args, **kwargs) -> None: ...
    def __len__(self) -> _int: ...
    def __getitem__(self, idx) -> None: ...
    def __setitem__(self, *args, **kwargs) -> None: ...
    def copy_(self, source: T, non_blocking: _bool | None = None) -> T: ...
    def new(self) -> _StorageBase | TypedStorage: ...
    def nbytes(self) -> _int: ...
    def size(self) -> _int: ...
    def type(self, dtype: str | None = None, non_blocking: _bool = False) -> _StorageBase | TypedStorage: ...
    def cuda(self, device=None, non_blocking: bool = False) -> _StorageBase | TypedStorage:
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination GPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
    def hpu(self, device=None, non_blocking: bool = False) -> _StorageBase | TypedStorage:
        """Returns a copy of this object in HPU memory.

        If this object is already in HPU memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination HPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
    def element_size(self) -> _int: ...
    def get_device(self) -> _int: ...
    def data_ptr(self) -> _int: ...
    def resizable(self) -> _bool: ...
    def _share_filename_cpu_(self, *args, **kwargs) -> None: ...
    def _share_fd_cpu_(self, *args, **kwargs) -> None: ...
    @classmethod
    def _new_using_filename_cpu(cls, size: _int) -> Self: ...
    @classmethod
    def _new_using_fd_cpu(cls, size: _int) -> Self: ...
    @classmethod
    def from_buffer(cls, *args, **kwargs) -> Self: ...
    @classmethod
    def _new_shared_filename_cpu(cls, manager, obj, size, *, device=None, dtype=None) -> Self: ...
    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs): ...
    @classmethod
    def _release_ipc_counter_cuda(cls, *args, **kwargs) -> Self: ...
    @classmethod
    def _new_with_weak_ptr(cls, *args, **kwargs) -> Self: ...
    def _shared_decref(self) -> _StorageBase | TypedStorage: ...
    def _write_file(self, *args, **kwargs) -> None: ...
    def resize_(self, size: _int): ...
    def _weak_ref(self, *args, **kwargs) -> _StorageBase | TypedStorage: ...
    def _set_from_file(self, *args, **kwargs) -> None: ...
    def _set_cdata(self, *args, **kwargs) -> None: ...
    def _share_cuda_(self, *args, **kwargs) -> None: ...
    def is_shared(self) -> _bool: ...
    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs) -> Self: ...
    def _shared_incref(self, *args, **kwargs) -> None: ...
    @classmethod
    def _free_weak_ref(cls, *args, **kwargs) -> None: ...
    @property
    def is_cuda(self) -> None: ...
    @property
    def is_hpu(self) -> None: ...
    @classmethod
    def from_file(cls, filename, shared, nbytes) -> _StorageBase | TypedStorage: ...
    @classmethod
    def _expired(cls, *args, **kwargs) -> _StorageBase | TypedStorage: ...
    def _byteswap(self, *args, **kwargs) -> None: ...
    def _get_filename(self, *args, **kwargs) -> str | None: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def __reduce__(self): ...
    def __sizeof__(self) -> int: ...
    def clone(self):
        """Return a copy of this storage."""
    def tolist(self):
        """Return a list containing the elements of this storage."""
    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
    def mps(self):
        """Return a MPS copy of this storage if it's not already on the MPS."""
    def _to(self, dtype): ...
    def to(self, *, device: DeviceLikeType, non_blocking: _bool = False): ...
    def double(self):
        """Casts this storage to double type."""
    def float(self):
        """Casts this storage to float type."""
    def half(self):
        """Casts this storage to half type."""
    def long(self):
        """Casts this storage to long type."""
    def int(self):
        """Casts this storage to int type."""
    def short(self):
        """Casts this storage to short type."""
    def char(self):
        """Casts this storage to char type."""
    def byte(self):
        """Casts this storage to byte type."""
    def bool(self):
        """Casts this storage to bool type."""
    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
    def complex_double(self):
        """Casts this storage to complex double type."""
    def complex_float(self):
        """Casts this storage to complex float type."""
    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
    def is_pinned(self, device: str | torch.device = 'cuda'):
        """Determine whether the CPU storage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
    def pin_memory(self, device: str | torch.device = 'cuda'):
        """Copy the CPU storage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
    @classmethod
    def _new_shared(cls, size, *, device: str = 'cpu'):
        """Create a new storage in shared memory with the same data type."""
    def untyped(self): ...
    def byteswap(self, dtype) -> None:
        """Swap bytes in underlying data."""

class UntypedStorage(torch._C.StorageBase, _StorageBase):
    def __getitem__(self, *args, **kwargs): ...
    @property
    def is_cuda(self): ...
    @property
    def is_hpu(self): ...
    @property
    def filename(self) -> str | None:
        """Returns the file name associated with this storage.

        The file name will be a string if the storage is on CPU and was created via
        :meth:`~torch.from_file()` with ``shared`` as ``True``. This attribute is ``None`` otherwise.
        """
    @_share_memory_lock_protected
    def share_memory_(self, *args, **kwargs):
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs): ...
    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs): ...

class TypedStorage:
    is_sparse: _bool
    _fake_device: torch.device | None
    dtype: torch.dtype
    @property
    def _dtype(self): ...
    @property
    def filename(self) -> str | None:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
        or ``None`` if the storage was not created by memory mapping a file."""
    def fill_(self, value): ...
    def __new__(cls, *args, wrap_storage=None, dtype=None, device=None, _internal: bool = False): ...
    _untyped_storage: Incomplete
    def __init__(self, *args, device=None, dtype=None, wrap_storage=None, _internal: bool = False) -> None: ...
    @property
    def is_cuda(self): ...
    @property
    def is_hpu(self): ...
    def untyped(self):
        """Return the internal :class:`torch.UntypedStorage`."""
    def _new_wrapped_storage(self, untyped_storage) -> Self: ...
    def __len__(self) -> int: ...
    def _maybe_wrap_index(self, idx, is_stop: bool = False): ...
    def __setitem__(self, idx, value) -> None: ...
    def _setitem(self, idx, value) -> None: ...
    def __getitem__(self, idx): ...
    def _getitem(self, idx): ...
    def copy_(self, source: T, non_blocking: bool | None = None): ...
    def nbytes(self): ...
    def _nbytes(self): ...
    def type(self, dtype: str | None = None, non_blocking: bool = False) -> _StorageBase | TypedStorage | str: ...
    def cuda(self, device=None, non_blocking: bool = False) -> Self: ...
    def hpu(self, device=None, non_blocking: bool = False) -> Self: ...
    def to(self, *, device: DeviceLikeType, non_blocking: bool = False) -> Self: ...
    def element_size(self): ...
    def _element_size(self): ...
    def get_device(self) -> _int: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __iter__(self): ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def _deepcopy(self, memo): ...
    def __sizeof__(self) -> int: ...
    def clone(self):
        """Return a copy of this storage."""
    def tolist(self):
        """Return a list containing the elements of this storage."""
    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
    def is_pinned(self, device: str | torch.device = 'cuda'):
        """Determine whether the CPU TypedStorage is already pinned on device.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A boolean variable.
        """
    def pin_memory(self, device: str | torch.device = 'cuda'):
        """Copy the CPU TypedStorage to pinned memory, if it's not already pinned.

        Args:
            device (str or torch.device): The device to pin memory on (default: ``'cuda'``).
                This argument is discouraged and subject to deprecated.

        Returns:
            A pinned CPU storage.
        """
    def share_memory_(self):
        """See :meth:`torch.UntypedStorage.share_memory_`"""
    def _share_memory_(self): ...
    def _new_shared(self, size, *, device=None):
        """Create a new storage in shared memory with the same data type."""
    @property
    def _cdata(self): ...
    @property
    def device(self): ...
    def size(self): ...
    def _size(self): ...
    def pickle_storage_type(self): ...
    def _pickle_storage_type(self): ...
    def __reduce__(self): ...
    def data_ptr(self): ...
    def _data_ptr(self): ...
    def resizable(self): ...
    def resize_(self, size) -> None: ...
    def _resize_(self, size) -> None: ...
    @classmethod
    def _free_weak_ref(cls, *args, **kwargs): ...
    def _weak_ref(self, *args, **kwargs): ...
    @classmethod
    def from_buffer(cls, *args, **kwargs): ...
    @classmethod
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs): ...
    def _to(self, dtype): ...
    def double(self):
        """Casts this storage to double type."""
    def float(self):
        """Casts this storage to float type."""
    def half(self):
        """Casts this storage to half type."""
    def long(self):
        """Casts this storage to long type."""
    def int(self):
        """Casts this storage to int type."""
    def short(self):
        """Casts this storage to short type."""
    def char(self):
        """Casts this storage to char type."""
    def byte(self):
        """Casts this storage to byte type."""
    def bool(self):
        """Casts this storage to bool type."""
    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
    def complex_double(self):
        """Casts this storage to complex double type."""
    def complex_float(self):
        """Casts this storage to complex float type."""
    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
    @classmethod
    def from_file(cls, filename, shared, size):
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
    @classmethod
    def _expired(cls, *args, **kwargs): ...
    def _write_file(self, *args, **kwargs): ...
    def _set_from_file(self, *args, **kwargs): ...
    def _set_cdata(self, *args, **kwargs): ...
    def _share_cuda_(self, *args, **kwargs): ...
    def is_shared(self): ...
    def _is_shared(self): ...
    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs): ...
    def _share_filename_cpu_(self, *args, **kwargs): ...
    def _shared_decref(self): ...
    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs): ...
    def _shared_incref(self, *args, **kwargs): ...
    def _share_fd_cpu_(self, *args, **kwargs): ...
    def _get_legacy_storage_class(self): ...

class _LegacyStorageMeta(type):
    dtype: torch.dtype
    def __instancecheck__(cls, instance): ...

class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    @classmethod
    def _new_shared(cls, size):
        """Create a new storage in shared memory with the same data type."""
    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs): ...
    @classmethod
    def _new_shared_filename(cls, manager, obj, size): ...
