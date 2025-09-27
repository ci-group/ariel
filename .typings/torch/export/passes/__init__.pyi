import torch
from torch.export.exported_program import ExportedProgram

__all__ = ['move_to_device_pass']

def move_to_device_pass(ep: ExportedProgram, location: torch.device | str | dict[str, str]) -> ExportedProgram:
    """
    Move the exported program to the given device.

    Args:
        ep (ExportedProgram): The exported program to move.
        location (Union[torch.device, str, Dict[str, str]]): The device to move the exported program to.
            If a string, it is interpreted as a device name.
            If a dict, it is interpreted as a mapping from
            the existing device to the intended one

    Returns:
        ExportedProgram: The moved exported program.
    """
