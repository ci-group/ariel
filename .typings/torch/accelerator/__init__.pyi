import torch
from ._utils import _device_t
from _typeshed import Incomplete

__all__ = ['current_accelerator', 'current_device_idx', 'current_device_index', 'current_stream', 'device_count', 'device_index', 'is_available', 'set_device_idx', 'set_device_index', 'set_stream', 'synchronize']

def device_count() -> int:
    """Return the number of current :ref:`accelerator<accelerators>` available.

    Returns:
        int: the number of the current :ref:`accelerator<accelerators>` available.
            If there is no available accelerators, return 0.

    .. note:: This API delegates to the device-specific version of `device_count`.
        On CUDA, this API will NOT poison fork if NVML discovery succeeds.
        Otherwise, it will. For more details, see :ref:`multiprocessing-poison-fork-note`.
    """
def is_available() -> bool:
    '''Check if the current accelerator is available at runtime: it was build, all the
    required drivers are available and at least one device is visible.
    See :ref:`accelerator<accelerators>` for details.

    Returns:
        bool: A boolean indicating if there is an available :ref:`accelerator<accelerators>`.

    .. note:: This API delegates to the device-specific version of `is_available`.
        On CUDA, when the environment variable ``PYTORCH_NVML_BASED_CUDA_CHECK=1`` is set,
        this function will NOT poison fork. Otherwise, it will. For more details, see
        :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> assert torch.accelerator.is_available() "No available accelerators detected."
    '''
def current_accelerator(check_available: bool = False) -> torch.device | None:
    """Return the device of the accelerator available at compilation time.
    If no accelerator were available at compilation time, returns None.
    See :ref:`accelerator<accelerators>` for details.

    Args:
        check_available (bool, optional): if True, will also do a runtime check to see
            if the device :func:`torch.accelerator.is_available` on top of the compile-time
            check.
            Default: ``False``

    Returns:
        torch.device: return the current accelerator as :class:`torch.device`.

    .. note:: The index of the returned :class:`torch.device` will be ``None``, please use
        :func:`torch.accelerator.current_device_index` to know the current index being used.
        This API does NOT poison fork. For more details, see :ref:`multiprocessing-poison-fork-note`.

    Example::

        >>> # xdoctest:
        >>> # If an accelerator is available, sent the model to it
        >>> model = torch.nn.Linear(2, 2)
        >>> if (current_device := current_accelerator(check_available=True)) is not None:
        >>>     model.to(current_device)
    """
def current_device_index() -> int:
    """Return the index of a currently selected device for the current :ref:`accelerator<accelerators>`.

    Returns:
        int: the index of a currently selected device.
    """

current_device_idx: Incomplete

def set_device_index(device: _device_t, /) -> None:
    """Set the current device index to a given device.

    Args:
        device (:class:`torch.device`, str, int): a given device that must match the current
            :ref:`accelerator<accelerators>` device type.

    .. note:: This function is a no-op if this device index is negative.
    """

set_device_idx: Incomplete

def current_stream(device: _device_t = None, /) -> torch.Stream:
    """Return the currently selected stream for a given device.

    Args:
        device (:class:`torch.device`, str, int, optional): a given device that must match the current
            :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    Returns:
        torch.Stream: the currently selected stream for a given device.
    """
def set_stream(stream: torch.Stream) -> None:
    """Set the current stream to a given stream.

    Args:
        stream (torch.Stream): a given stream that must match the current :ref:`accelerator<accelerators>` device type.

    .. note:: This function will set the current device index to the device index of the given stream.
    """
def synchronize(device: _device_t = None, /) -> None:
    '''Wait for all kernels in all streams on the given device to complete.

    Args:
        device (:class:`torch.device`, str, int, optional): device for which to synchronize. It must match
            the current :ref:`accelerator<accelerators>` device type. If not given,
            use :func:`torch.accelerator.current_device_index` by default.

    .. note:: This function is a no-op if the current :ref:`accelerator<accelerators>` is not initialized.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> assert torch.accelerator.is_available() "No available accelerators detected."
        >>> start_event = torch.Event(enable_timing=True)
        >>> end_event = torch.Event(enable_timing=True)
        >>> start_event.record()
        >>> tensor = torch.randn(100, device=torch.accelerator.current_accelerator())
        >>> sum = torch.sum(tensor)
        >>> end_event.record()
        >>> torch.accelerator.synchronize()
        >>> elapsed_time_ms = start_event.elapsed_time(end_event)
    '''

class device_index:
    """Context manager to set the current device index for the current :ref:`accelerator<accelerators>`.
    Temporarily changes the current device index to the specified value for the duration
    of the context, and automatically restores the previous device index when exiting
    the context.

    Args:
        device (Optional[int]): a given device index to temporarily set. If None,
            no device index switching occurs.

    Examples:

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # Set device 0 as the current device temporarily
        >>> with torch.accelerator.device_index(0):
        ...     # Code here runs with device 0 as the current device
        ...     pass
        >>> # Original device is now restored
        >>> # No-op when None is passed
        >>> with torch.accelerator.device_index(None):
        ...     # No device switching occurs
        ...     pass
    """
    idx: Incomplete
    prev_idx: int
    def __init__(self, device: int | None, /) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *exc_info: object) -> None: ...
