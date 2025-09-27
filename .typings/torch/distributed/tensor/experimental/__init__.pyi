from collections.abc import Iterator
from contextlib import contextmanager
from torch.distributed.tensor.experimental._attention import context_parallel as context_parallel
from torch.distributed.tensor.experimental._func_map import local_map as local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding as register_sharding

__all__ = ['context_parallel', 'implicit_replication', 'local_map', 'register_sharding']

@contextmanager
def implicit_replication() -> Iterator[None]:
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
