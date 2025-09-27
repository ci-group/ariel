import torch
from torch._utils import _dummy_type as _dummy_type

class Stream(torch._C._CudaStreamBase):
    """Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.  See :ref:`cuda-semantics` for details.

    Args:
        device(torch.device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.
            If the value falls outside of the allowed priority range, it will automatically be
            mapped to the nearest valid priority (lowest for large positive numbers or
            highest for large negative numbers).

    """
    def __new__(cls, device=None, priority: int = 0, **kwargs): ...
    def wait_event(self, event) -> None:
        """Make all future work submitted to the stream wait for an event.

        Args:
            event (torch.cuda.Event): an event to wait for.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
           `CUDA Stream documentation`_ for more info.

           This function returns without waiting for :attr:`event`: only future
           operations are affected.

        .. _CUDA Stream documentation:
           https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
        """
    def wait_stream(self, stream) -> None:
        """Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.

        .. note:: This function returns without waiting for currently enqueued
           kernels in :attr:`stream`: only future operations are affected.
        """
    def record_event(self, event=None):
        """Record an event.

        Args:
            event (torch.cuda.Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
    def query(self) -> bool:
        """Check if all the work submitted has been completed.

        Returns:
            A boolean indicating if all kernels in this stream are completed.
        """
    def synchronize(self) -> None:
        """Wait for all the kernels in this stream to complete.

        .. note:: This is a wrapper around ``cudaStreamSynchronize()``: see
           `CUDA Stream documentation`_ for more info.
        """
    @property
    def _as_parameter_(self): ...
    def __eq__(self, o) -> bool: ...
    def __hash__(self): ...
    def __repr__(self) -> str: ...

class ExternalStream(Stream):
    """Wrapper around an externally allocated CUDA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cudaStream_t` value.
            allocated externally.
        device(torch.device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """
    def __new__(cls, stream_ptr, device=None, **kwargs): ...

class Event(torch._C._CudaEventBase):
    """Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    The underlying CUDA events are lazily initialized when the event is first
    recorded or exported to another process. After creation, only streams on the
    same device may record the event. However, streams on any device can wait on
    the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
        external (bool, optional): indicates whether this event should create event record and event wait nodes, or create an internal cross-stream dependency, when captured in a cuda graph. See `cross-stream dependencies <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-c-programming-guide/index.html#cross-stream-dependencies-and-events>`_, `cudaEventRecordExternal <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3457b81d1d32c6a00f6132fbc2693d47>`_, and `cudaEventWaitExternal <https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g0c23426b7252eaa9cef695859991304e>`_ for more information about internal vs. external events. (default: ``False``)

    .. _CUDA Event Documentation:
       https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html
    """
    def __new__(cls, enable_timing: bool = False, blocking: bool = False, interprocess: bool = False, external: bool = False): ...
    @classmethod
    def from_ipc_handle(cls, device, handle):
        """Reconstruct an event from an IPC handle on the given device."""
    def record(self, stream=None) -> None:
        """Record the event in a given stream.

        Uses ``torch.cuda.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
    def wait(self, stream=None) -> None:
        """Make all future work submitted to the given stream wait for this event.

        Use ``torch.cuda.current_stream()`` if no stream is specified.

        .. note:: This is a wrapper around ``cudaStreamWaitEvent()``: see
            `CUDA Event documentation`_ for more info.
        """
    def query(self):
        """Check if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has
            completed.
        """
    def elapsed_time(self, end_event):
        """Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.
        """
    def synchronize(self) -> None:
        """Wait for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.

         .. note:: This is a wrapper around ``cudaEventSynchronize()``: see
            `CUDA Event documentation`_ for more info.
        """
    def ipc_handle(self):
        """Return an IPC handle of this event.

        If not recorded yet, the event will use the current device.
        """
    @property
    def _as_parameter_(self): ...
    def __repr__(self) -> str: ...
