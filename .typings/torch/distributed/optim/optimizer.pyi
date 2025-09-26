import torch.jit as jit
import torch.nn as nn
from _typeshed import Incomplete

__all__ = ['DistributedOptimizer']

class _ScriptLocalOptimizerInterface:
    def step(self, autograd_ctx_id: int) -> None: ...

class _ScriptLocalOptimizer(nn.Module):
    compile_lock: Incomplete
    _local_params: Incomplete
    optim: Incomplete
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs) -> None: ...
    @jit.export
    def step(self, autograd_ctx_id: int): ...

class _LocalOptimizer:
    global_lock: Incomplete
    _local_params: Incomplete
    optim: Incomplete
    def __init__(self, optim_cls, local_params_rref, *args, **kwargs) -> None: ...
    def step(self, autograd_ctx_id) -> None: ...

class DistributedOptimizer:
    '''
    DistributedOptimizer takes remote references to parameters scattered
    across workers and applies the given optimizer locally for each parameter.

    This class uses :meth:`~torch.distributed.autograd.get_gradients` in order
    to retrieve the gradients for specific parameters.

    Concurrent calls to
    :meth:`~torch.distributed.optim.DistributedOptimizer.step`,
    either from the same or different clients, will
    be serialized on each worker -- as each worker\'s optimizer can only work
    on one set of gradients at a time. However, there is no guarantee that
    the full forward-backward-optimizer sequence will execute for one client
    at a time. This means that the gradients being applied may not correspond
    to the latest forward pass executed on a given worker. Also, there is no
    guaranteed ordering across workers.

    `DistributedOptimizer` creates the local optimizer with TorchScript enabled
    by default, so that optimizer updates are not blocked by the Python Global
    Interpreter Lock (GIL) in the case of multithreaded training (e.g. Distributed
    Model Parallel). This feature is currently enabled for most optimizers. You
    can also follow `the recipe`__ in PyTorch tutorials to enable TorchScript support
    for your own custom optimizers.

    Args:
        optimizer_class (optim.Optimizer): the class of optimizer to
            instantiate on each worker.
        params_rref (list[RRef]): list of RRefs to local or remote parameters
            to optimize.
        args: arguments to pass to the optimizer constructor on each worker.
        kwargs: arguments to pass to the optimizer constructor on each worker.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> import torch.distributed.autograd as dist_autograd
        >>> import torch.distributed.rpc as rpc
        >>> from torch import optim
        >>> from torch.distributed.optim import DistributedOptimizer
        >>>
        >>> with dist_autograd.context() as context_id:
        >>>   # Forward pass.
        >>>   rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>>   rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>>   loss = rref1.to_here() + rref2.to_here()
        >>>
        >>>   # Backward pass.
        >>>   dist_autograd.backward(context_id, [loss.sum()])
        >>>
        >>>   # Optimizer.
        >>>   dist_optim = DistributedOptimizer(
        >>>      optim.SGD,
        >>>      [rref1, rref2],
        >>>      lr=0.05,
        >>>   )
        >>>   dist_optim.step(context_id)

    __ https://github.com/pytorch/tutorials/pull/1465
    '''
    is_functional_optim: Incomplete
    remote_optimizers: Incomplete
    def __init__(self, optimizer_class, params_rref, *args, **kwargs) -> None: ...
    def step(self, context_id) -> None:
        """
        Performs a single optimization step.

        This will call :meth:`torch.optim.Optimizer.step` on each worker
        containing parameters to be optimized, and will block until all workers
        return. The provided ``context_id`` will be used to retrieve the
        corresponding :class:`~torch.distributed.autograd.context` that
        contains the gradients that should be applied to the parameters.

        Args:
            context_id: the autograd context id for which we should run the
                optimizer step.
        """
