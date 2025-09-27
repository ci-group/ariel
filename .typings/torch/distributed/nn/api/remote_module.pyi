import torch
import torch.distributed.rpc as rpc
from _typeshed import Incomplete
from collections.abc import Iterator, Mapping
from torch import Tensor, device, dtype, nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
from typing import Any, Callable, TypeVar
from typing_extensions import Self

__all__ = ['RemoteModule']

_grad_t = tuple[Tensor, ...] | Tensor
T = TypeVar('T', bound='Module')
_SerializedRemoteModule: Incomplete

class _RemoteModule(nn.Module):
    def __new__(cls, *args, **kwargs): ...
    is_scriptable: bool
    module_rref: Incomplete
    generated_methods: Incomplete
    def __init__(self, remote_device: str, module_cls: type[nn.Module], args: tuple | None = None, kwargs: dict[str, Any] | None = None, _module_interface_cls: Any = None) -> None:
        '''
        RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.
        It can be shared across processors using `RPC framework <https://pytorch.org/docs/stable/rpc.html>`__,
        without incurring any overheads of copying the actual module,
        which is equivalent to an :class:`~torch.distributed.rpc.RRef`
        pointing to the remote module.

        The arguments of ``forward_async`` and ``forward`` are the same as
        the ``forward`` method of the module returned by the ``module_cls``.

        Apart from ``forward_async`` and ``forward``, no other methods are supported from nn.Module for now.

        Particularly, to create a hybrid model, typically the local modules should be
        created outside of remote modules, rather than as submodules of any remote module (by calling ``add_module``).
        Hybrid Example:
                >>> class HybridModel(nn.Module):
                >>>     def __init__(self) -> None:
                >>>         nn.Module.__init__(self)
                >>>         self.remote_embedding = RemoteModule(...)
                >>>         self.local_linear = nn.Linear(...)

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature, ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods in signature of
        ``def forward(input: Tensor) -> Tensor:`` and
        ``def forward_async(input: Tensor) -> Future[Tensor]:``.

        .. note::
            If the remote module is placed on a cuda device,
            any input CPU tensors will be automatically moved to the same cuda device,
            and GPU tensors are returned over the wire according to the device map of the remote worker on TensorPipe RPC backend.

        Args:
            remote_device (str): Device on the destination worker where we\'d like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_cls (nn.Module): For example,
                >>> class MyModule(nn.Module):
                >>>     def forward(input):
                >>>         return input + 1
                >>>
                >>> module_cls = MyModule
            args (Sequence, optional): args to be passed to ``module_cls``.
            kwargs (Dict, optional): kwargs to be passed to ``module_cls``.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_cls``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_linear_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>> input = torch.randn(128, 20)
            >>> ret_fut = remote_linear_module.forward_async(input)
            >>> ret = ret_fut.wait()
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()
        '''
    def remote_parameters(self, recurse: bool = True) -> list[rpc.RRef[Parameter]]:
        """
        Return a list of :class:`~torch.distributed.rpc.RRef` pointing to the remote module's parameters.

        This can typically be used in conjunction
        with :class:`~torch.distributed.optim.DistributedOptimizer`.

        Args:
            recurse (bool): if True, then returns parameters of the remote
                module and all submodules of the remote module. Otherwise,
                returns only parameters that are direct members of the
                remote module.

        Returns:
            A list of :class:`~torch.distributed.rpc.RRef` (``List[RRef[nn.Parameter]]``)
            to remote module's parameters.
        """
    def get_module_rref(self) -> rpc.RRef[nn.Module]:
        """Return an :class:`~torch.distributed.rpc.RRef` (``RRef[nn.Module]``) pointing to the remote module."""
    @torch.jit.export
    def __getstate__(self) -> None: ...
    @torch.jit.export
    def __setstate__(self, state) -> None: ...
    def register_buffer(self, name: str, tensor: Tensor | None, persistent: bool = True) -> None: ...
    def register_parameter(self, name: str, param: Parameter | None) -> None: ...
    def add_module(self, name: str, module: Module | None) -> None: ...
    def apply(self, fn: Callable[[Module], None]) -> Self: ...
    def cuda(self, device: int | device | None = None) -> Self: ...
    def ipu(self, device: int | device | None = None) -> Self: ...
    def xpu(self, device: int | device | None = None) -> Self: ...
    def cpu(self) -> Self: ...
    def type(self, dst_type: dtype | str) -> Self: ...
    def float(self) -> Self: ...
    def double(self) -> Self: ...
    def half(self) -> Self: ...
    def bfloat16(self) -> Self: ...
    def to(self, *args, **kwargs) -> T: ...
    def register_backward_hook(self, hook: Callable[[Module, _grad_t, _grad_t], None | _grad_t]) -> RemovableHandle: ...
    def register_forward_pre_hook(self, hook: Callable[[T, tuple[Any, ...]], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None], prepend: bool = False, with_kwargs: bool = False) -> RemovableHandle: ...
    def register_forward_hook(self, hook: Callable[[T, tuple[Any, ...], Any], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any], Any], Any | None], prepend: bool = False, with_kwargs: bool = False) -> RemovableHandle: ...
    def state_dict(self, *args, **kwargs) -> None: ...
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False): ...
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]: ...
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[tuple[str, Parameter]]: ...
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]: ...
    def named_buffers(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[tuple[str, Tensor]]: ...
    def children(self) -> Iterator[Module]: ...
    def named_children(self) -> Iterator[tuple[str, Module]]: ...
    def modules(self) -> Iterator[Module]: ...
    def named_modules(self, memo: set[Module] | None = None, prefix: str = '', remove_duplicate: bool = True): ...
    def train(self, mode: bool = True) -> Self: ...
    def eval(self) -> Self: ...
    def requires_grad_(self, requires_grad: bool = True) -> Self: ...
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    def share_memory(self) -> Self: ...
    def extra_repr(self) -> str: ...
    on: Incomplete
    device: Incomplete
    is_device_map_set: Incomplete
    def _prepare_init(self, remote_device_str: str) -> bool:
        """Prepare the initialization and returns whether to enable automatically moving CPU tensors to CUDA devices."""
    def _init_template(self, module_interface_cls, enable_moving_cpu_tensors_to_cuda) -> None:
        """Instantiate template on local side."""
    def _check_attribute_picklability(self) -> None:
        """Check if all the attribute has explicitly defined whether to be pickled (i.e., picklability)."""
    def _install_generated_methods(self) -> None: ...
    @staticmethod
    def init_from_module_rref(remote_device: str, module_rref: rpc.RRef[nn.Module], _module_interface_cls: Any = None):
        '''
        Besides the constructor, a RemoteModule instance can also be initialized given a module RRef.

        This alternate initialization method can be particularly useful if we want to create multiple
        RemoteModule instances that share the same underlying module and reduce memory consumption.

        Moreover, this also provides a workaround for passing script RemoteModule over RPC,
        which is not supported. The recommended way is as follows:

            1. the sender creates a RemoteModule;
            2. the sender sends its ``module_rref`` over RPC;
            3. the receiver calls this method to initialize another RemoteModule using the same ``module_rref``.

        Example::
            Run the following code in two different processes:

            >>> # xdoctest: +SKIP("distributed")
            >>> # On worker 0:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>> from torch import nn, Tensor
            >>> from torch.distributed.nn.api.remote_module import RemoteModule
            >>>
            >>> rpc.init_rpc("worker0", rank=0, world_size=2)
            >>> remote_module = RemoteModule(
            >>>     "worker1/cpu", nn.Linear, args=(20, 30),
            >>> )
            >>>
            >>> remote_module1 = rpc.rpc_sync(
            >>>     "worker1/cpu",
            >>>     RemoteModule.init_from_module_rref,
            >>>     ("worker1/cpu", remote_module1.get_module_rref()),
            >>> )
            >>> rpc.shutdown()

            >>> # On worker 1:
            >>> import torch
            >>> import torch.distributed.rpc as rpc
            >>>
            >>> rpc.init_rpc("worker1", rank=1, world_size=2)
            >>> rpc.shutdown()

        Args:
            remote_device (str): Device on the destination worker where we\'d like to place this module.
                The device can be a local device or a remote device specified by one of the following remote
                formats:

                    1. "rank:<rank>/<device>" (ex: "rank:0/cuda:0").
                    2. "<worker_name>/<device>" (ex: "trainer0/cuda:0").

                In addition, the device field can be optional and the default value is "cpu".
            module_rref (RRef[nn.Module]): The module reference shared by both the caller and
                the created remote module.
            _module_interface_cls (type, optional): The TorchScript interface type for the module
                to be created. The type object should be decorated by @torch.jit.interface.
                If not provided, the generated RemoteModule is not torchscript-able.
                Warning, this is an experimental API and susceptible to frequent changes.

        Returns:
            A remote module instance which wraps the :class:`~nn.Module` created by the
            user-provided ``module_rref``, it has a blocking ``forward`` method and an
            asynchronous ``forward_async`` method that returns a future of the ``forward`` call
            on the user-provided module on the remote side.
        '''

class RemoteModule(_RemoteModule):
    '''
        A RemoteModule instance can only be created after RPC initialization.

        It creates a user-specified module on a specified remote node.
        It behaves like a regular ``nn.Module`` except that the ``forward`` method is
        executed on the remote node.
        It takes care of autograd recording to ensure the backward pass propagates
        gradients back to the corresponding remote module.

        It generates two methods ``forward_async`` and ``forward`` based on the
        signature of the ``forward`` method of ``module_cls``. ``forward_async``
        runs asynchronously and returns a Future. The arguments of ``forward_async``
        and ``forward`` are the same as the ``forward`` method of the module
        returned by the ``module_cls``.

        For example, if ``module_cls`` returns an instance of ``nn.Linear``,
        that has ``forward`` method signature: ``def forward(input: Tensor) -> Tensor:``,
        the generated ``RemoteModule`` will have 2 methods with the signatures:

        | ``def forward(input: Tensor) -> Tensor:``
        | ``def forward_async(input: Tensor) -> Future[Tensor]:``

    Args:
        remote_device (str): Device on the destination worker where we\'d like to place this module.
            The format should be "<workername>/<device>", where the device field can be parsed as torch.device type.
            E.g., "trainer0/cpu", "trainer0", "ps0/cuda:0".
            In addition, the device field can be optional and the default value is "cpu".
        module_cls (nn.Module): Class for the module to be created remotely. For example,

            >>> class MyModule(nn.Module):
            >>>     def forward(input):
            >>>         return input + 1
            >>>
            >>> module_cls = MyModule

        args (Sequence, optional): args to be passed to ``module_cls``.
        kwargs (Dict, optional): kwargs to be passed to ``module_cls``.

    Returns:
        A remote module instance which wraps the :class:`~nn.Module` created by the
        user-provided ``module_cls``, it has a blocking ``forward`` method and an
        asynchronous ``forward_async`` method that returns a future of the ``forward`` call
        on the user-provided module on the remote side.

    Example::
        Run the following code in two different processes:

        >>> # xdoctest: +SKIP("distributed")
        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> from torch import nn, Tensor
        >>> from torch.distributed.nn.api.remote_module import RemoteModule
        >>>
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> remote_linear_module = RemoteModule(
        >>>     "worker1/cpu", nn.Linear, args=(20, 30),
        >>> )
        >>> input = torch.randn(128, 20)
        >>> ret_fut = remote_linear_module.forward_async(input)
        >>> ret = ret_fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>>
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Furthermore, a more practical example that is combined with
        `DistributedDataParallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__ (DDP)
        can be found in this `tutorial <https://pytorch.org/tutorials/advanced/rpc_ddp_tutorial.html>`__.
    '''
    def __init__(self, remote_device: str, module_cls: type[nn.Module], args: tuple | None = None, kwargs: dict[str, Any] | None = None) -> None: ...
