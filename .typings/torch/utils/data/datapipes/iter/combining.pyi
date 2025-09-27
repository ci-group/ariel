import abc
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from torch.utils.data.datapipes.datapipe import IterDataPipe
from typing import Any, Callable, Literal, TypeVar

__all__ = ['ConcaterIterDataPipe', 'DemultiplexerIterDataPipe', 'ForkerIterDataPipe', 'MultiplexerIterDataPipe', 'ZipperIterDataPipe']

_T_co = TypeVar('_T_co', covariant=True)

class ConcaterIterDataPipe(IterDataPipe):
    """
    Concatenates multiple Iterable DataPipes (functional name: ``concat``).

    The resulting DataPipe will yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """
    datapipes: tuple[IterDataPipe]
    def __init__(self, *datapipes: IterDataPipe) -> None: ...
    def __iter__(self) -> Iterator: ...
    def __len__(self) -> int: ...

class ForkerIterDataPipe(IterDataPipe):
    '''
    Creates multiple instances of the same Iterable DataPipe (functional name: ``fork``).

    Args:
        datapipe: Iterable DataPipe being copied
        num_instances: number of instances of the datapipe to create
        buffer_size: this restricts how far ahead the leading child DataPipe
           can read relative to the slowest child DataPipe.
           Defaults to ``1000``. Use ``-1`` for the unlimited buffer.
        copy: copy strategy to use for items yielded by each branch. Supported
            options are ``None`` for no copying, ``"shallow"`` for shallow object
            copies, and ``"deep"`` for deep object copies. Defaults to ``None``.

    Note:
        All branches of the forked pipeline return the identical object unless
        the copy parameter is supplied. If the object is mutable or contains
        mutable objects, changing them in one branch will affect all others.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.fork(num_instances=2)
        >>> list(dp1)
        [0, 1, 2, 3, 4]
        >>> list(dp2)
        [0, 1, 2, 3, 4]
    '''
    def __new__(cls, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000, copy: Literal['shallow', 'deep'] | None = None): ...

class _ContainerTemplate(ABC, metaclass=abc.ABCMeta):
    """Abstract class for container ``DataPipes``. The followings are three required methods."""
    @abstractmethod
    def get_next_element_by_instance(self, instance_id: int): ...
    @abstractmethod
    def is_every_instance_exhausted(self) -> bool: ...
    @abstractmethod
    def reset(self) -> None: ...
    @abstractmethod
    def get_length_by_instance(self, instance_id: int):
        """Raise TypeError if it's not supposed to be implemented to support `list(datapipe)`."""

class _ForkerIterDataPipe(IterDataPipe, _ContainerTemplate):
    """
    Container to hold instance-specific information on behalf of ForkerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, and yields the next value
    as requested by the child DataPipes.
    """
    main_datapipe: Incomplete
    _datapipe_iterator: Iterator[Any] | None
    num_instances: Incomplete
    buffer: deque
    buffer_size: Incomplete
    copy_fn: Incomplete
    child_pointers: list[int]
    slowest_ptr: int
    leading_ptr: int
    end_ptr: int | None
    _child_stop: list[bool]
    def __init__(self, datapipe: IterDataPipe, num_instances: int, buffer_size: int = 1000, copy: Literal['shallow', 'deep'] | None = None) -> None: ...
    def __len__(self) -> int: ...
    _snapshot_state: Incomplete
    def get_next_element_by_instance(self, instance_id: int): ...
    def is_every_instance_exhausted(self) -> bool: ...
    def get_length_by_instance(self, instance_id: int) -> int: ...
    def reset(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def _cleanup(self) -> None: ...
    def __del__(self) -> None: ...

class _ChildDataPipe(IterDataPipe):
    """
    Iterable Datapipe that is a child of a main DataPipe.

    The instance of this class will pass its instance_id to get the next value from its main DataPipe.

    Note:
        ChildDataPipe, like all other IterDataPipe, follows the single iterator per IterDataPipe constraint.
        Since ChildDataPipes share a common buffer, when an iterator is created for one of the ChildDataPipes,
        the previous iterators  for all ChildDataPipes must be invalidated, with the exception when a ChildDataPipe
        hasn't had an iterator created from it since the last invalidation. See the example below.

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> # Singler Iterator per IteraDataPipe Invalidation
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> source_dp = IterableWrapper(range(10))
        >>> cdp1, cdp2 = source_dp.fork(num_instances=2)
        >>> it1, it2 = iter(cdp1), iter(cdp2)
        >>> it3 = iter(cdp1)
        >>> # The line above invalidates `it1` and `it2`, and resets `ForkerIterDataPipe`.
        >>> it4 = iter(cdp2)
        >>> # The line above doesn't invalidate `it3`, because an iterator for `cdp2` hasn't been created since
        >>> # the last invalidation.

    Args:
        main_datapipe: Main DataPipe with a method 'get_next_element_by_instance(instance_id)'
        instance_id: integer identifier of this instance
    """
    _is_child_datapipe: bool
    main_datapipe: IterDataPipe
    instance_id: Incomplete
    def __init__(self, main_datapipe: IterDataPipe, instance_id: int) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    _valid_iterator_id: Incomplete
    def _set_main_datapipe_valid_iterator_id(self) -> int:
        """
        Update the valid iterator ID for both this DataPipe object and `main_datapipe`.

        `main_datapipe.reset()` is called when the ID is incremented to a new generation.
        """
    def _check_valid_iterator_id(self, iterator_id) -> bool:
        """Check the valid iterator ID against that of DataPipe object and that of `main_datapipe`."""

class DemultiplexerIterDataPipe(IterDataPipe):
    """
    Splits the input DataPipe into multiple child DataPipes, using the given classification function (functional name: ``demux``).

    A list of the child DataPipes is returned from this operation.

    Args:
        datapipe: Iterable DataPipe being filtered
        num_instances: number of instances of the DataPipe to create
        classifier_fn: a function that maps values to an integer within the range ``[0, num_instances - 1]`` or ``None``
        drop_none: defaults to ``False``, if ``True``, the function will skip over elements classified as ``None``
        buffer_size: this defines the maximum number of inputs that the buffer can hold across all child
            DataPipes while waiting for their values to be yielded.
            Defaults to ``1000``. Use ``-1`` for the unlimited buffer.

    Examples:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> def odd_or_even(n):
        ...     return n % 2
        >>> source_dp = IterableWrapper(range(5))
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even)
        >>> list(dp1)
        [0, 2, 4]
        >>> list(dp2)
        [1, 3]
        >>> # It can also filter out any element that gets `None` from the `classifier_fn`
        >>> def odd_or_even_no_zero(n):
        ...     return n % 2 if n != 0 else None
        >>> dp1, dp2 = source_dp.demux(num_instances=2, classifier_fn=odd_or_even_no_zero, drop_none=True)
        >>> list(dp1)
        [2, 4]
        >>> list(dp2)
        [1, 3]
    """
    def __new__(cls, datapipe: IterDataPipe, num_instances: int, classifier_fn: Callable[[_T_co], int | None], drop_none: bool = False, buffer_size: int = 1000): ...

class _DemultiplexerIterDataPipe(IterDataPipe, _ContainerTemplate):
    """
    Container to hold instance-specific information on behalf of DemultiplexerIterDataPipe.

    It tracks the state of its child DataPipes, maintains the buffer, classifies and yields the next correct value
    as requested by the child DataPipes.
    """
    main_datapipe: Incomplete
    _datapipe_iterator: Iterator[Any] | None
    num_instances: Incomplete
    buffer_size: Incomplete
    current_buffer_usage: int
    child_buffers: list[deque[_T_co]]
    classifier_fn: Incomplete
    drop_none: Incomplete
    main_datapipe_exhausted: bool
    _child_stop: list[bool]
    def __init__(self, datapipe: IterDataPipe[_T_co], num_instances: int, classifier_fn: Callable[[_T_co], int | None], drop_none: bool, buffer_size: int) -> None: ...
    def _find_next(self, instance_id: int) -> _T_co: ...
    _snapshot_state: Incomplete
    def get_next_element_by_instance(self, instance_id: int): ...
    def is_every_instance_exhausted(self) -> bool: ...
    def get_length_by_instance(self, instance_id: int) -> int: ...
    def reset(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def _cleanup(self, instance_id: int | None = None): ...
    def __del__(self) -> None: ...

class MultiplexerIterDataPipe(IterDataPipe):
    """
    Yields one element at a time from each of the input Iterable DataPipes (functional name: ``mux``).

    As in, one element from the 1st input DataPipe, then one element from the 2nd DataPipe in the next iteration,
    and so on. It ends when the shortest input DataPipe is exhausted.

    Args:
        datapipes: Iterable DataPipes that will take turn to yield their elements, until the shortest DataPipe is exhausted

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.mux(dp2, dp3))
        [0, 10, 20, 1, 11, 21, 2, 12, 22]
    """
    datapipes: Incomplete
    buffer: list
    def __init__(self, *datapipes) -> None: ...
    def __iter__(self): ...
    def __len__(self) -> int: ...
    def reset(self) -> None: ...
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    def __del__(self) -> None: ...

class ZipperIterDataPipe(IterDataPipe[tuple[_T_co]]):
    """
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).

    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """
    datapipes: tuple[IterDataPipe]
    def __init__(self, *datapipes: IterDataPipe) -> None: ...
    def __iter__(self) -> Iterator[tuple[_T_co]]: ...
    def __len__(self) -> int: ...
