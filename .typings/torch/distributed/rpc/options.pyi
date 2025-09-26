import torch
from _typeshed import Incomplete
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase

__all__ = ['TensorPipeRpcBackendOptions']

DeviceType = int | str | torch.device
_TensorPipeRpcBackendOptionsBase = object

class TensorPipeRpcBackendOptions(_TensorPipeRpcBackendOptionsBase):
    """
    The backend options for
    :class:`~torch.distributed.rpc.TensorPipeAgent`, derived from
    :class:`~torch.distributed.rpc.RpcBackendOptions`.

    Args:
        num_worker_threads (int, optional): The number of threads in the
            thread-pool used by
            :class:`~torch.distributed.rpc.TensorPipeAgent` to execute
            requests (default: 16).
        rpc_timeout (float, optional): The default timeout, in seconds,
            for RPC requests (default: 60 seconds). If the RPC has not
            completed in this timeframe, an exception indicating so will
            be raised. Callers can override this timeout for individual
            RPCs in :meth:`~torch.distributed.rpc.rpc_sync` and
            :meth:`~torch.distributed.rpc.rpc_async` if necessary.
        init_method (str, optional): The URL to initialize the distributed
            store used for rendezvous. It takes any value accepted for the
            same argument of :meth:`~torch.distributed.init_process_group`
            (default: ``env://``).
        device_maps (Dict[str, Dict], optional): Device placement mappings from
            this worker to the callee. Key is the callee worker name and value
            the dictionary (``Dict`` of ``int``, ``str``, or ``torch.device``)
            that maps this worker's devices to the callee worker's devices.
            (default: ``None``)
        devices (List[int, str, or ``torch.device``], optional): all local
            CUDA devices used by RPC agent. By Default, it will be initialized
            to all local devices from its own ``device_maps`` and corresponding
            devices from its peers' ``device_maps``. When processing CUDA RPC
            requests, the agent will properly synchronize CUDA streams for
            all devices in this ``List``.
    """
    def __init__(self, *, num_worker_threads: int = ..., rpc_timeout: float = ..., init_method: str = ..., device_maps: dict[str, dict[DeviceType, DeviceType]] | None = None, devices: list[DeviceType] | None = None, _transports: list | None = None, _channels: list | None = None) -> None: ...
    def set_device_map(self, to: str, device_map: dict[DeviceType, DeviceType]):
        '''
        Set device mapping between each RPC caller and callee pair. This
        function can be called multiple times to incrementally add
        device placement configurations.

        Args:
            to (str): Callee name.
            device_map (Dict of int, str, or torch.device): Device placement
                mappings from this worker to the callee. This map must be
                invertible.

        Example:
            >>> # xdoctest: +SKIP("distributed")
            >>> # both workers
            >>> def add(x, y):
            >>>     print(x)  # tensor([1., 1.], device=\'cuda:1\')
            >>>     return x + y, (x + y).to(2)
            >>>
            >>> # on worker 0
            >>> options = TensorPipeRpcBackendOptions(
            >>>     num_worker_threads=8,
            >>>     device_maps={"worker1": {0: 1}}
            >>> # maps worker0\'s cuda:0 to worker1\'s cuda:1
            >>> )
            >>> options.set_device_map("worker1", {1: 2})
            >>> # maps worker0\'s cuda:1 to worker1\'s cuda:2
            >>>
            >>> rpc.init_rpc(
            >>>     "worker0",
            >>>     rank=0,
            >>>     world_size=2,
            >>>     backend=rpc.BackendType.TENSORPIPE,
            >>>     rpc_backend_options=options
            >>> )
            >>>
            >>> x = torch.ones(2)
            >>> rets = rpc.rpc_sync("worker1", add, args=(x.to(0), 1))
            >>> # The first argument will be moved to cuda:1 on worker1. When
            >>> # sending the return value back, it will follow the invert of
            >>> # the device map, and hence will be moved back to cuda:0 and
            >>> # cuda:1 on worker0
            >>> print(rets[0])  # tensor([2., 2.], device=\'cuda:0\')
            >>> print(rets[1])  # tensor([2., 2.], device=\'cuda:1\')
        '''
    devices: Incomplete
    def set_devices(self, devices: list[DeviceType]):
        """
        Set local devices used by the TensorPipe RPC agent. When processing
        CUDA RPC requests, the TensorPipe RPC agent will properly synchronize
        CUDA streams for all devices in this ``List``.

        Args:
            devices (List of int, str, or torch.device): local devices used by
                the TensorPipe RPC agent.
        """
