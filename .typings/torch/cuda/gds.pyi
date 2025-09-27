from _typeshed import Incomplete
from torch.types import Storage

__all__ = ['gds_register_buffer', 'gds_deregister_buffer', 'GdsFile']

def gds_register_buffer(s: Storage) -> None:
    '''Registers a storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)

    Args:
        s (Storage): Buffer to register.
    '''
def gds_deregister_buffer(s: Storage) -> None:
    '''Deregisters a previously registered storage on a CUDA device as a cufile buffer.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src = torch.randn(1024, device="cuda")
        >>> s = src.untyped_storage()
        >>> gds_register_buffer(s)
        >>> gds_deregister_buffer(s)

    Args:
        s (Storage): Buffer to register.
    '''

class GdsFile:
    '''Wrapper around cuFile.

    cuFile is a file-like interface to the GPUDirect Storage (GDS) API.

    See the `cufile docs <https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufile-io-api>`_
    for more details.

    Args:
        filename (str): Name of the file to open.
        flags (int): Flags to pass to ``os.open`` when opening the file. ``os.O_DIRECT`` will
            be added automatically.

    Example::

        >>> # xdoctest: +SKIP("gds filesystem requirements")
        >>> src1 = torch.randn(1024, device="cuda")
        >>> src2 = torch.randn(2, 1024, device="cuda")
        >>> file = torch.cuda.gds.GdsFile(f, os.O_CREAT | os.O_RDWR)
        >>> file.save_storage(src1.untyped_storage(), offset=0)
        >>> file.save_storage(src2.untyped_storage(), offset=src1.nbytes)
        >>> dest1 = torch.empty(1024, device="cuda")
        >>> dest2 = torch.empty(2, 1024, device="cuda")
        >>> file.load_storage(dest1.untyped_storage(), offset=0)
        >>> file.load_storage(dest2.untyped_storage(), offset=src1.nbytes)
        >>> torch.equal(src1, dest1)
        True
        >>> torch.equal(src2, dest2)
        True

    '''
    filename: Incomplete
    flags: Incomplete
    fd: Incomplete
    handle: int | None
    def __init__(self, filename: str, flags: int) -> None: ...
    def __del__(self) -> None: ...
    def register_handle(self) -> None:
        """Registers file descriptor to cuFile Driver.

        This is a wrapper around ``cuFileHandleRegister``.
        """
    def deregister_handle(self) -> None:
        """Deregisters file descriptor from cuFile Driver.

        This is a wrapper around ``cuFileHandleDeregister``.
        """
    def load_storage(self, storage: Storage, offset: int = 0) -> None:
        """Loads data from the file into the storage.

        This is a wrapper around ``cuFileRead``. ``storage.nbytes()`` of data
        will be loaded from the file at ``offset`` into the storage.

        Args:
            storage (Storage): Storage to load data into.
            offset (int, optional): Offset into the file to start loading from. (Default: 0)
        """
    def save_storage(self, storage: Storage, offset: int = 0) -> None:
        """Saves data from the storage into the file.

        This is a wrapper around ``cuFileWrite``. All bytes of the storage
        will be written to the file at ``offset``.

        Args:
            storage (Storage): Storage to save data from.
            offset (int, optional): Offset into the file to start saving to. (Default: 0)
        """
