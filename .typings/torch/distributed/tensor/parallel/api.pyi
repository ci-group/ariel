import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel.style import ParallelStyle

__all__ = ['parallelize_module']

def parallelize_module(module: nn.Module, device_mesh: DeviceMesh | None = None, parallelize_plan: ParallelStyle | dict[str, ParallelStyle] | None = None, *, src_data_rank: int | None = 0) -> nn.Module:
    '''
    Apply Tensor Parallelism in PyTorch by parallelizing modules or sub-modules based on a user-specified plan.

    We parallelize module or sub_modules based on a parallelize_plan. The parallelize_plan contains
    :class:`ParallelStyle`, which indicates how user wants the module or sub_module
    to be parallelized.

    User can also specify different parallel style per module fully qualified name (FQN).

    Note that ``parallelize_module`` only accepts a 1-D :class:`DeviceMesh`, if you have a 2-D or N-D :class:`DeviceMesh`,
    slice the DeviceMesh to a 1-D sub DeviceMesh first then pass to this API(i.e. ``device_mesh["tp"]``)

    Args:
        module (:class:`nn.Module`):
            Module to be parallelized.
        device_mesh (:class:`DeviceMesh`, optional):
            Object which describes the mesh topology of devices for the DTensor.
            If not specified, the call must be under a DeviceMesh context.
        parallelize_plan (Union[:class:`ParallelStyle`, Dict[str, :class:`ParallelStyle`]], optional):
            The plan used to parallelize the module. It can be either a
            :class:`ParallelStyle` object which contains how we prepare
            input/output for Tensor Parallelism or it can be a dict of module
            FQN and its corresponding :class:`ParallelStyle` object. If not
            specified, the call will do nothing at the moment.
    Keyword args:
        src_data_rank (int, optional): the rank of the source data for the logical/global tensor, it is used by
            :meth:`distribute_tensor` to scatter/broadcast the shards/replicas to other ranks. By default,
            we use ``group_rank=0`` on each DeviceMesh dimension as the source data to preserve the single-device
            semantic. If passing ``None`` explicitly, :meth:`parallelize_module` simply uses its local data instead
            of trying to preserve the single-device semantic via scatter/broadcast. Default: 0
    Return:
        A :class:`nn.Module` object parallelized.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel
        >>> from torch.distributed.device_mesh import init_device_mesh
        >>>
        >>> # Define the module.
        >>> m = Model(...)
        >>> tp_mesh = init_device_mesh("cuda", (8,))
        >>> m = parallelize_module(m, tp_mesh, {"w1": ColwiseParallel(), "w2": RowwiseParallel()})
        >>>

    .. note:: For complex module architecture like Attention, MLP layers, we recommend composing
        different ParallelStyles together (i.e. ``ColwiseParallel`` and ``RowwiseParallel``) and pass
        as a parallelize_plan, to achieves the desired sharding computation.
    '''
