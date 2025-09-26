import torch
from torch._utils import _dummy_type as _dummy_type

class Stream(torch._C._XpuStreamBase):
    """Wrapper around a XPU stream.

    A XPU stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.

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
            event (torch.xpu.Event): an event to wait for.
        """
    def wait_stream(self, stream) -> None:
        """Synchronize with another stream.

        All future work submitted to this stream will wait until all kernels
        submitted to a given stream at the time of call complete.

        Args:
            stream (Stream): a stream to synchronize.
        """
    def record_event(self, event=None):
        """Record an event.

        Args:
            event (torch.xpu.Event, optional): event to record. If not given, a new one
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
        """Wait for all the kernels in this stream to complete."""
    @property
    def _as_parameter_(self): ...
    def __eq__(self, o): ...
    def __hash__(self): ...
    def __repr__(self) -> str: ...

class Event(torch._C._XpuEventBase):
    """Wrapper around a XPU event.

    XPU events are synchronization markers that can be used to monitor the
    device's progress, and to synchronize XPU streams.

    The underlying XPU events are lazily initialized when the event is first
    recorded. After creation, only streams on the same device may record the
    event. However, streams on any device can wait on the event.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
    """
    def __new__(cls, enable_timing: bool = False): ...
    def record(self, stream=None) -> None:
        """Record the event in a given stream.

        Uses ``torch.xpu.current_stream()`` if no stream is specified. The
        stream's device must match the event's device.
        """
    def wait(self, stream=None) -> None:
        """Make all future work submitted to the given stream wait for this event.

        Use ``torch.xpu.current_stream()`` if no stream is specified.
        """
    def query(self) -> bool:
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
        """
    @property
    def _as_parameter_(self): ...
    def __repr__(self) -> str: ...
