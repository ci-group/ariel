import types
from _typeshed import Incomplete
from torch._C._distributed_autograd import DistAutogradContext as DistAutogradContext, _current_context as _current_context, _get_debug_info as _get_debug_info, _get_max_id as _get_max_id, _init as _init, _is_valid_context as _is_valid_context, _new_context as _new_context, _release_context as _release_context, _retrieve_context as _retrieve_context, backward as backward, get_gradients as get_gradients

def is_available(): ...

class context:
    '''
    Context object to wrap forward and backward passes when using
    distributed autograd. The ``context_id`` generated in the ``with``
    statement  is required to uniquely identify a distributed backward pass
    on all workers. Each worker stores metadata associated with this
    ``context_id``, which is required to correctly execute a distributed
    autograd pass.

    Example::
        >>> # xdoctest: +SKIP
        >>> import torch.distributed.autograd as dist_autograd
        >>> with dist_autograd.context() as context_id:
        >>>     t1 = torch.rand((3, 3), requires_grad=True)
        >>>     t2 = torch.rand((3, 3), requires_grad=True)
        >>>     loss = rpc.rpc_sync("worker1", torch.add, args=(t1, t2)).sum()
        >>>     dist_autograd.backward(context_id, [loss])
    '''
    autograd_context: Incomplete
    def __enter__(self): ...
    def __exit__(self, type: type[BaseException] | None, value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
