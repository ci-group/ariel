import torch
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass

def get_untyped_storages(t: torch.Tensor) -> set[torch.UntypedStorage]:
    """
    Recursively extracts untyped storages from a tensor or its subclasses.

    Args:
        t (torch.Tensor): The tensor to extract storages from.

    Returns:
        Set[torch.UntypedStorage]: A set of untyped storages.
    """
