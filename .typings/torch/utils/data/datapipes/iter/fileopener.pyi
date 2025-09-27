from collections.abc import Iterable
from io import IOBase
from torch.utils.data.datapipes.datapipe import IterDataPipe

__all__ = ['FileOpenerIterDataPipe']

class FileOpenerIterDataPipe(IterDataPipe[tuple[str, IOBase]]):
    '''
    Given pathnames, opens files and yield pathname and file stream in a tuple (functional name: ``open_files``).

    Args:
        datapipe: Iterable datapipe that provides pathnames
        mode: An optional string that specifies the mode in which
            the file is opened by ``open()``. It defaults to ``r``, other options are
            ``b`` for reading in binary mode and ``t`` for text mode.
        encoding: An optional string that specifies the encoding of the
            underlying file. It defaults to ``None`` to match the default encoding of ``open``.
        length: Nominal length of the datapipe

    Note:
        The opened file handles will be closed by Python\'s GC periodically. Users can choose
        to close them explicitly.

    Example:
        >>> # xdoctest: +SKIP
        >>> from torchdata.datapipes.iter import FileLister, FileOpener, StreamReader
        >>> dp = FileLister(root=".").filter(lambda fname: fname.endswith(\'.txt\'))
        >>> dp = FileOpener(dp)
        >>> dp = StreamReader(dp)
        >>> list(dp)
        [(\'./abc.txt\', \'abc\')]
    '''
    datapipe: Iterable
    mode: str
    encoding: str | None
    length: int
    def __init__(self, datapipe: Iterable[str], mode: str = 'r', encoding: str | None = None, length: int = -1) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
