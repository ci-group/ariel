import torch
from _typeshed import Incomplete

class _remote_device:
    '''
    Represents a device on a remote worker.

    Args:
        remote_device (str or torch.device): Represents a device on a remote worker.
            The string format should be one of the following:

                1. "<workername>/<device>", where the device field can be parsed as torch.device type.
                   E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
                   In addition, the device field can be optional and the default value is "cpu".
                2. "rank:<rank>/<device>", where <rank> is the rank of the
                   process and device can be parsed as torch.device type.
                   E.g., "rank:0/cpu", "rank:0", "rank:0/cuda:0"
                3. <workername> and <rank> are optional and formats like "cpu"
                    and "cuda:1", just represent local devices.
    '''
    _worker_name: Incomplete
    _rank: Incomplete
    _device: str | int | torch.device | None
    def __init__(self, remote_device: str | torch.device) -> None: ...
    @staticmethod
    def _is_valid_local_device(device): ...
    def worker_name(self) -> str | None:
        """Return the name of remote worker representing the remote device and ``None`` if no worker name is available."""
    def rank(self) -> int | None:
        """
        Returns the rank of remote worker representing the remote device.
        Returns ``None`` if no rank is available.
        """
    def device(self) -> torch.device:
        """Return the local device on the remote worker."""
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    def __hash__(self): ...
