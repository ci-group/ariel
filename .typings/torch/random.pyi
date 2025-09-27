import contextlib
import torch
from collections.abc import Generator
from torch._C import default_generator as default_generator

def set_rng_state(new_state: torch.Tensor) -> None:
    """Sets the random number generator state.

    .. note:: This function only works for CPU. For CUDA, please use
        :func:`torch.manual_seed`, which works for both CPU and CUDA.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
def get_rng_state() -> torch.Tensor:
    """Returns the random number generator state as a `torch.ByteTensor`.

    .. note:: The returned state is for the default generator on CPU only.

    See also: :func:`torch.random.fork_rng`.
    """
def manual_seed(seed) -> torch._C.Generator:
    """Sets the seed for generating random numbers on all devices. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed. Value must be within the inclusive range
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`. Otherwise, a RuntimeError
            is raised. Negative inputs are remapped to positive values with the formula
            `0xffff_ffff_ffff_ffff + seed`.
    """
def seed() -> int:
    """Sets the seed for generating random numbers to a non-deterministic
    random number on all devices. Returns a 64 bit number used to seed the RNG.
    """
def _seed_custom_device(seed) -> None:
    """Sets the seed to generate random numbers for custom device.

    Args:
        seed (int): The desired seed.

    See [Note: support the custom device with privateuse1]
    """
def initial_seed() -> int:
    """Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """

_fork_rng_warned_already: bool

@contextlib.contextmanager
def fork_rng(devices=None, enabled: bool = True, _caller: str = 'fork_rng', _devices_kw: str = 'devices', device_type: str = 'cuda') -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is `cuda`. As for supported device,
            see details in :ref:`accelerator<accelerators>`
    """
