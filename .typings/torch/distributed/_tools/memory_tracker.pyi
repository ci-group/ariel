from _typeshed import Incomplete
from torch.utils._python_dispatch import TorchDispatchMode as TorchDispatchMode
from torch.utils.hooks import RemovableHandle as RemovableHandle
from typing import Callable, no_type_check

BYTES_PER_MB: Incomplete

class MemoryProfileDispatchMode(TorchDispatchMode):
    """Run in ``TorchDispatchMode`` to get memory stats at operator level."""
    memory_tracker: Incomplete
    def __init__(self, memory_tracker) -> None: ...
    def __torch_dispatch__(self, func, types, args=..., kwargs=None): ...

class MemoryTracker:
    """
    Collect and plot the memory stats at operator level.

    Includes ``memories_allocated``, ``memories_active`` and ``memories_reserved``.
    It also prints a summary for the top 20 operators that generate the most memories.

    Example usage:

        >>> # xdoctest: +SKIP(failing)
        >>> net.cuda()
        >>> input = input.cuda()

        >>> mem_tracker = MemoryTracker()
        >>> mem_tracker.start_monitor(net)

        >>> net.zero_grad(True)
        >>> loss = net(input)
        >>> if isinstance(loss, dict):
        >>>    loss = loss['out']
        >>> loss.sum().backward()
        >>> net.zero_grad(set_to_none=True)

        >>> mem_tracker.stop()
        >>> mem_tracker.summary()
        >>> mem_tracker.show_traces()
    """
    _hooks: list[RemovableHandle]
    _operator_names: dict[str, int]
    memories_allocated: dict[int, dict[str, float]]
    memories_active: dict[int, dict[str, float]]
    memories_reserved: dict[int, dict[str, float]]
    _markers: dict[str, int]
    _cur_module_name: str
    _op_index: int
    _num_alloc_retries: int
    _device_module: Incomplete
    def __init__(self) -> None: ...
    profile_mode: Incomplete
    @no_type_check
    def start_monitor(self, root_module) -> None:
        """
        Register module hooks and entering ``MemoryProfileDispatchMode``.

        This enables operator level memory stats can be tracked during module runtime.
        """
    @no_type_check
    def stop(self) -> None:
        """
        Remove module hooks and exit ``MemoryProfileDispatchMode`` to stop tracking memory stats at operator level.

        Get some aggregated stats when the memory_tracker() is enabled, like ``num_alloc_retries``.
        """
    @no_type_check
    def summary(self, top: int = 20) -> None:
        """
        Print out the top operators that generate the most memories.

        The number of the top operators can be configured.
        """
    @no_type_check
    def show_traces(self, path: str = '') -> None: ...
    def save_stats(self, path: str) -> None:
        """Save the stats using pickle during runtime if users want to plot the traces in other places like notebook."""
    def load(self, path: str) -> None:
        """Load the pickled memory stats to plot the traces or print the summary."""
    def _create_pre_forward_hook(self, name: str) -> Callable:
        """Prefix operator name with current module and 'forward', and insert 'fw_start' marker at forward pass start."""
    def _create_post_forward_hook(self, name: str) -> Callable:
        """Insert the marker 'fw_bw_boundary' at the boundary of forward and backward pass."""
    def _create_backward_hook(self, name: str) -> Callable:
        """Insert the current module name with backward prefix for the operator name."""
    @no_type_check
    def _record_memory_stats(self, fn_name) -> None:
        """
        Record current memory allocated, current memory active and current memory reserved.

        The memory stats dict is indexed with ``self._op_index``.
        """
    def _add_marker(self, marker_name: str) -> None:
        """Set the marker's x-axis value."""
    def _clear_state(self) -> None:
        """Clear states when start_monitor() is called."""
