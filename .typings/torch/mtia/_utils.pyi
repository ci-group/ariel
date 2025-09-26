from typing import Any

def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    """Get the device index from :attr:`device`, which can be a torch.device object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    is a MTIA device. Note that for a MTIA device without a specified index,
    i.e., ``torch.device('mtia')``, this will return the current default MTIA
    device if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default MTIA
    device if :attr:`optional` is ``True``.
    """
