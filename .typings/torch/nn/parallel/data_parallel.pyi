import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch.nn.modules import Module
from typing import Any, Generic, TypeVar

__all__ = ['DataParallel', 'data_parallel']

T = TypeVar('T', bound=Module)

class DataParallel(Module, Generic[T]):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given :attr:`module` by
    splitting the input across the specified devices by chunking in the batch
    dimension (other objects will be copied once per device). In the forward
    pass, the module is replicated on each device, and each replica handles a
    portion of the input. During the backwards pass, gradients from each replica
    are summed into the original module.

    The batch size should be larger than the number of GPUs used.

    .. warning::
        It is recommended to use :class:`~torch.nn.parallel.DistributedDataParallel`,
        instead of this class, to do multi-GPU training, even if there is only a single
        node. See: :ref:`cuda-nn-ddp-instead` and :ref:`ddp`.

    Arbitrary positional and keyword inputs are allowed to be passed into
    DataParallel but some types are specially handled. tensors will be
    **scattered** on dim specified (default 0). tuple, list and dict types will
    be shallow copied. The other types will be shared among different threads
    and can be corrupted if written to in the model's forward pass.

    The parallelized :attr:`module` must have its parameters and buffers on
    ``device_ids[0]`` before running this :class:`~torch.nn.DataParallel`
    module.

    .. warning::
        In each forward, :attr:`module` is **replicated** on each device, so any
        updates to the running module in ``forward`` will be lost. For example,
        if :attr:`module` has a counter attribute that is incremented in each
        ``forward``, it will always stay at the initial value because the update
        is done on the replicas which are destroyed after ``forward``. However,
        :class:`~torch.nn.DataParallel` guarantees that the replica on
        ``device[0]`` will have its parameters and buffers sharing storage with
        the base parallelized :attr:`module`. So **in-place** updates to the
        parameters or buffers on ``device[0]`` will be recorded. E.g.,
        :class:`~torch.nn.BatchNorm2d` and :func:`~torch.nn.utils.spectral_norm`
        rely on this behavior to update the buffers.

    .. warning::
        Forward and backward hooks defined on :attr:`module` and its submodules
        will be invoked ``len(device_ids)`` times, each with inputs located on
        a particular device. Particularly, the hooks are only guaranteed to be
        executed in correct order with respect to operations on corresponding
        devices. For example, it is not guaranteed that hooks set via
        :meth:`~torch.nn.Module.register_forward_pre_hook` be executed before
        `all` ``len(device_ids)`` :meth:`~torch.nn.Module.forward` calls, but
        that each such hook be executed before the corresponding
        :meth:`~torch.nn.Module.forward` call of that device.

    .. warning::
        When :attr:`module` returns a scalar (i.e., 0-dimensional tensor) in
        :func:`forward`, this wrapper will return a vector of length equal to
        number of devices used in data parallelism, containing the result from
        each device.

    .. note::
        There is a subtlety in using the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`pack-rnn-unpack-with-data-parallelism` section in FAQ for
        details.


    Args:
        module (Module): module to be parallelized
        device_ids (list of int or torch.device): CUDA devices (default: all devices)
        output_device (int or torch.device): device location of output (default: device_ids[0])

    Attributes:
        module (Module): the module to be parallelized

    Example::

        >>> # xdoctest: +SKIP
        >>> net = torch.nn.DataParallel(model, device_ids=[0, 1, 2])
        >>> output = net(input_var)  # input_var can be on any device, including CPU
    """
    module: Incomplete
    device_ids: Incomplete
    dim: Incomplete
    output_device: Incomplete
    src_device_obj: Incomplete
    def __init__(self, module: T, device_ids: Sequence[int | torch.device] | None = None, output_device: int | torch.device | None = None, dim: int = 0) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any) -> Any: ...
    def replicate(self, module: T, device_ids: Sequence[int | torch.device]) -> list[T]: ...
    def scatter(self, inputs: tuple[Any, ...], kwargs: dict[str, Any] | None, device_ids: Sequence[int | torch.device]) -> Any: ...
    def parallel_apply(self, replicas: Sequence[T], inputs: Sequence[Any], kwargs: Any) -> list[Any]: ...
    def gather(self, outputs: Any, output_device: int | torch.device) -> Any: ...

def data_parallel(module: Module, inputs: Any, device_ids: Sequence[int | torch.device] | None = None, output_device: int | torch.device | None = None, dim: int = 0, module_kwargs: Any | None = None) -> torch.Tensor:
    """Evaluate module(input) in parallel across the GPUs given in device_ids.

    This is the functional version of the DataParallel module.

    Args:
        module (Module): the module to evaluate in parallel
        inputs (Tensor): inputs to the module
        device_ids (list of int or torch.device): GPU ids on which to replicate module
        output_device (list of int or torch.device): GPU location of the output  Use -1 to indicate the CPU.
            (default: device_ids[0])
    Returns:
        a Tensor containing the result of module(input) located on
        output_device
    """
