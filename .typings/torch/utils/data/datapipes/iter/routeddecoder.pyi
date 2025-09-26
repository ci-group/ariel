from _typeshed import Incomplete
from collections.abc import Iterable, Iterator
from io import BufferedIOBase
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Any, Callable

__all__ = ['RoutedDecoderIterDataPipe']

class RoutedDecoderIterDataPipe(IterDataPipe[tuple[str, Any]]):
    """
    Decodes binary streams from input DataPipe, yields pathname and decoded data in a tuple.

    (functional name: ``routed_decode``)

    Args:
        datapipe: Iterable datapipe that provides pathname and binary stream in tuples
        handlers: Optional user defined decoder handlers. If ``None``, basic and image decoder
            handlers will be set as default. If multiple handles are provided, the priority
            order follows the order of handlers (the first handler has the top priority)
        key_fn: Function for decoder to extract key from pathname to dispatch handlers.
            Default is set to extract file extension from pathname

    Note:
        When ``key_fn`` is specified returning anything other than extension, the default
        handler will not work and users need to specify custom handler. Custom handler
        could use regex to determine the eligibility to handle data.
    """
    datapipe: Iterable[tuple[str, BufferedIOBase]]
    decoder: Incomplete
    def __init__(self, datapipe: Iterable[tuple[str, BufferedIOBase]], *handlers: Callable, key_fn: Callable = ...) -> None: ...
    def add_handler(self, *handler: Callable) -> None: ...
    def __iter__(self) -> Iterator[tuple[str, Any]]: ...
    def __len__(self) -> int: ...
