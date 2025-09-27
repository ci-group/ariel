from . import _device_t
from typing import Any

__all__ = ['memory_stats', 'max_memory_allocated', 'reset_peak_memory_stats']

def memory_stats(device: _device_t | None = None) -> dict[str, Any]:
    """Return a dictionary of MTIA memory allocator statistics for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
def max_memory_allocated(device: _device_t | None = None) -> int:
    """Return the maximum memory allocated in bytes for a given device.

    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
def reset_peak_memory_stats(device: _device_t | None = None) -> None:
    """Reset the peak memory stats for a given device.


    Args:
        device (torch.device, str, or int, optional) selected device. Returns
            statistics for the current device, given by current_device(),
            if device is None (default).
    """
