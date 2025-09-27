import torch
from collections.abc import Iterable
from torch import Tensor

__all__ = ['get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all', 'manual_seed', 'manual_seed_all', 'seed', 'seed_all', 'initial_seed']

def get_rng_state(device: int | str | torch.device = 'cuda') -> Tensor:
    """Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
def get_rng_state_all() -> list[Tensor]:
    """Return a list of ByteTensor representing the random number states of all devices."""
def set_rng_state(new_state: Tensor, device: int | str | torch.device = 'cuda') -> None:
    """Set the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
def set_rng_state_all(new_states: Iterable[Tensor]) -> None:
    """Set the random number generator state of all devices.

    Args:
        new_states (Iterable of torch.ByteTensor): The desired state for each device.
    """
def manual_seed(seed: int) -> None:
    """Set the seed for generating random numbers for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
def manual_seed_all(seed: int) -> None:
    """Set the seed for generating random numbers on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
def seed() -> None:
    """Set the seed for generating random numbers to a random number for the current GPU.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """
def seed_all() -> None:
    """Set the seed for generating random numbers to a random number on all GPUs.

    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """
def initial_seed() -> int:
    """Return the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
