import contextlib
from .internal import _internal_rpc_pickler
from _typeshed import Incomplete
from torch._C._distributed_rpc import PyRRef
from typing import Generic, TypeVar

__all__ = ['shutdown', 'get_worker_info', 'remote', 'rpc_sync', 'rpc_async', 'RRef', 'AllGatherStates', 'method_factory', 'new_method']

_default_pickler = _internal_rpc_pickler

class AllGatherStates:
    gathered_objects: Incomplete
    proceed_signal: Incomplete
    def __init__(self) -> None: ...

@_require_initialized
def shutdown(graceful: bool = True, timeout=...) -> None:
    '''
    Perform a shutdown of the RPC agent, and then destroy the RPC agent. This
    stops the local agent from accepting outstanding requests, and shuts
    down the RPC framework by terminating all RPC threads. If ``graceful=True``,
    this will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete. Otherwise, if
    ``graceful=False``, this is a local shutdown, and it does not wait for other
    RPC processes to reach this method.

    .. warning::
        For :class:`~torch.futures.Future` objects returned by
        :meth:`~torch.distributed.rpc.rpc_async`, ``future.wait()`` should not
        be called after ``shutdown()``.

    Args:
        graceful (bool): Whether to do a graceful shutdown or not. If True,
                         this will 1) wait until there is no pending system
                         messages for ``UserRRefs`` and delete them; 2) block
                         until all local and remote RPC processes have reached
                         this method and wait for all outstanding work to
                         complete.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> # do some work
        >>> result = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(1), 1))
        >>> # ready to shutdown
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    '''
@_require_initialized
def get_worker_info(worker_name=None):
    """
    Get :class:`~torch.distributed.rpc.WorkerInfo` of a given worker name.
    Use this :class:`~torch.distributed.rpc.WorkerInfo` to avoid passing an
    expensive string on every invocation.

    Args:
        worker_name (str): the string name of a worker. If ``None``, return the
                           the id of the current worker. (default ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` instance for the given
        ``worker_name`` or :class:`~torch.distributed.rpc.WorkerInfo` of the
        current worker if ``worker_name`` is ``None``.
    """
T = TypeVar('T')
GenericWithOneTypeVar = Generic[T]

class RRef(PyRRef[T], Generic[T]): ...

def method_factory(method_name, docstring): ...

new_method: Incomplete

@_require_initialized
def remote(to, func, args=None, kwargs=None, timeout=...):
    '''
    Make a remote call to run ``func`` on worker ``to`` and return an
    :class:`~torch.distributed.rpc.RRef` to the result value immediately.
    Worker ``to`` will be the owner of the returned
    :class:`~torch.distributed.rpc.RRef`, and the worker calling ``remote`` is
    a user. The owner manages the global reference count of its
    :class:`~torch.distributed.rpc.RRef`, and the owner
    :class:`~torch.distributed.rpc.RRef` is only destructed when globally there
    are no living references to it.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

        timeout (float, optional): timeout in seconds for this remote call. If the
                                   creation of this
                                   :class:`~torch.distributed.rpc.RRef` on worker
                                   ``to`` is not successfully processed on this
                                   worker within this timeout, then the next time
                                   there is an attempt to use the RRef (such as
                                   ``to_here()``), a timeout will be raised
                                   indicating this failure. A value of 0 indicates
                                   an infinite timeout, i.e. a timeout error will
                                   never be raised. If not provided, the default
                                   value set during initialization or with
                                   ``_set_rpc_timeout`` is used.

    Returns:
        A user :class:`~torch.distributed.rpc.RRef` instance to the result
        value. Use the blocking API :meth:`torch.distributed.rpc.RRef.to_here`
        to retrieve the result value locally.

    .. warning ::
        The ``remote`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned RRef is
        confirmed by the owner, which can be checked using the
        :meth:`torch.distributed.rpc.RRef.confirmed_by_owner` API.

    .. warning ::
        Errors such as timeouts for the ``remote`` API are handled on a
        best-effort basis. This means that when remote calls initiated by
        ``remote`` fail, such as with a timeout error, we take a best-effort
        approach to error handling. This means that errors are handled and set
        on the resulting RRef on an asynchronous basis. If the RRef has not been
        used by the application before this handling (such as ``to_here`` or
        fork call), then future uses of the ``RRef`` will appropriately raise
        errors. However, it is possible that the user application will use the
        ``RRef`` before the errors are handled. In this case, errors may not be
        raised as they have not yet been handled.

    Example::

        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref = rpc.remote("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rref.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    '''
@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None, timeout: float = ...):
    '''
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.

    Returns:
        Returns the result of running ``func`` with ``args`` and ``kwargs``.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

    '''
@_require_initialized
def rpc_async(to, func, args=None, kwargs=None, timeout=...):
    '''
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a
    :class:`~torch.futures.Future` that can be awaited on.

    Args:
        to (str or WorkerInfo or int): name/rank/``WorkerInfo`` of the destination worker.
        func (Callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.
        timeout (float, optional): timeout in seconds to use for this RPC. If
                                   the RPC does not complete in this amount of
                                   time, an exception indicating it has
                                   timed out will be raised. A value of 0
                                   indicates an infinite timeout, i.e. a timeout
                                   error will never be raised. If not provided,
                                   the default value set during initialization
                                   or with ``_set_rpc_timeout`` is used.


    Returns:
        Returns a :class:`~torch.futures.Future` object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the :class:`~torch.futures.Future`
        object.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don\'t support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``rpc_async`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned
        :class:`~torch.futures.Future` completes.

    Example::
        Make sure that ``MASTER_ADDR`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        export MASTER_ADDR=localhost
        export MASTER_PORT=5678

        Then run the following code in two different processes:

        >>> # xdoctest: +SKIP
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut1 = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async("worker1", min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(tensor: torch.Tensor, scalar: int):
        >>>    return torch.add(tensor, scalar)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut = rpc.rpc_async("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> ret = fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    '''
