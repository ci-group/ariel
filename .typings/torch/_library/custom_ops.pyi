import torch
import weakref
from . import autograd as autograd, utils as utils
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from torch import Tensor as Tensor, _C as _C, _ops as _ops
from torch.types import _dtype as _dtype
from torch.utils._exposed_in import exposed_in as exposed_in
from typing import Any, Callable, Literal, overload

device_types_t = str | Sequence[str] | None
log: Incomplete

@overload
def custom_op(name: str, fn: Literal[None] = None, /, *, mutates_args: str | Iterable[str], device_types: device_types_t = None, schema: str | None = None) -> Callable[[Callable[..., object]], 'CustomOpDef']: ...
@overload
def custom_op(name: str, fn: Callable[..., object], /, *, mutates_args: str | Iterable[str], device_types: device_types_t = None, schema: str | None = None) -> CustomOpDef: ...

class CustomOpDef:
    """CustomOpDef is a wrapper around a function that turns it into a custom op.

    It has various methods for registering additional behavior for this
    custom op.

    You should not instantiate CustomOpDef directly; instead, use the
    :func:`torch.library.custom_op` API.
    """
    _namespace: Incomplete
    _name: Incomplete
    _schema: Incomplete
    _tags: Incomplete
    _init_fn: Incomplete
    _backend_fns: dict[str | None, Callable]
    _abstract_fn: Callable | None
    _setup_context_fn: Callable | None
    _backward_fn: Callable | None
    _torch_dispatch_fns: dict[type, Callable]
    _vmap_fn: Callable | None
    _autocast_cuda_dtype: _dtype | None
    _autocast_cpu_dtype: _dtype | None
    _lib: Incomplete
    _disabled_kernel: set
    def __init__(self, namespace: str, name: str, schema: str, fn: Callable, tags: Sequence[_C.Tag] | None = None) -> None: ...
    @property
    def _qualname(self) -> str: ...
    def __repr__(self) -> str: ...
    @contextmanager
    def set_kernel_enabled(self, device_type: str, enabled: bool = True):
        '''
        Disable or re-enable an already registered kernel for this custom operator.

        If the kernel is already disabled/enabled, this is a no-op.

        Note:
            If a kernel is first disabled and then registered, it is disabled until enabled again.

        Args:
            device_type (str): The device type to disable/enable the kernel for.
            disable (bool): Whether to disable or enable the kernel.

        Example:
            >>> inp = torch.randn(1)
            >>>
            >>> # define custom op `f`.
            >>> @custom_op("mylib::f", mutates_args=())
            >>> def f(x: Tensor) -> Tensor:
            >>>     return torch.zeros(1)
            >>>
            >>> print(f(inp))  # tensor([0.]), default kernel
            >>>
            >>> @f.register_kernel("cpu")
            >>> def _(x):
            >>>     return torch.ones(1)
            >>>
            >>> print(f(inp))  # tensor([1.]), CPU kernel
            >>>
            >>> # temporarily disable the CPU kernel
            >>> with f.set_kernel_enabled("cpu", enabled = False):
            >>>     print(f(inp))  # tensor([0.]) with CPU kernel disabled

        '''
    def register_kernel(self, device_types: device_types_t, fn: Callable | None = None, /) -> Callable:
        '''Register an implementation for a device type for this operator.

        Some valid device_types are: "cpu", "cuda", "xla", "mps", "ipu", "xpu".
        This API may be used as a decorator.

        Args:
            fn (Callable): The function to register as the implementation for
                the given device types.
            device_types (str | Sequence[str]): The device device_types to register an impl to.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>> import numpy as np
            >>>
            >>> # Create a custom op that works on cpu
            >>> @custom_op("mylib::numpy_sin", mutates_args=(), device_types="cpu")
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np)
            >>>
            >>> # Add implementations for the cuda device
            >>> @numpy_sin.register_kernel("cuda")
            >>> def _(x):
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> x_cpu = torch.randn(3)
            >>> x_cuda = x_cpu.cuda()
            >>> assert torch.allclose(numpy_sin(x_cpu), x_cpu.sin())
            >>> assert torch.allclose(numpy_sin(x_cuda), x_cuda.sin())

        '''
    def register_fake(self, fn: Callable, /) -> Callable:
        '''Register a FakeTensor implementation for this custom op.

        This is necessary to get the operator to work efficiently with torch.compile.

        The Fake impl (sometimes also known as a meta kernel or abstract impl)
        specifies the behavior of this operator on Tensors that carry no data.
        Given some input Tensors with certain properties
        (sizes/strides/storage_offset/device), it specifies what the properties of
        the output Tensors are.

        Please see :func:`torch.library.impl_abstract` for more details.

        Args:
            fn (Callable): The function to register as the FakeTensor
                implementation.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> # Example 1: an operator without data-dependent output shape
            >>> @torch.library.custom_op("mylib::linear", mutates_args=())
            >>> def linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            >>>     return (x @ weight.t()) + bias
            >>>
            >>> @linear.register_fake
            >>> def _(x, weight, bias):
            >>>     assert x.dim() == 2
            >>>     assert weight.dim() == 2
            >>>     assert bias.dim() == 1
            >>>     assert x.shape[1] == weight.shape[1]
            >>>     assert weight.shape[0] == bias.shape[0]
            >>>     assert x.device == weight.device
            >>>     return x.new_empty(x.size(0), weight.size(0))
            >>>
            >>> x = torch.randn(2, 2)
            >>> weight = torch.randn(2, 2)
            >>> bias = torch.randn(2)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(linear, fullgraph=True)(x, weight, bias)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, torch.nn.functional.linear(x, weight, bias))
            >>>
            >>> # Example 2: an operator with data-dependent output shape
            >>> @torch.library.custom_op("mylib::nonzero", mutates_args=())
            >>> def nonzero(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)
            >>>
            >>> @nonzero.register_fake
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an abstract impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> x = torch.tensor([0, 1, 2, 0, 0, 1])
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> out = torch.compile(nonzero, fullgraph=True)(x)
            >>> # xdoctest: +SKIP("Requires Python <= 3.11")
            >>> assert torch.allclose(out, x.nonzero())

        '''
    def register_torch_dispatch(self, torch_dispatch_class: Any, fn: Callable | None = None, /) -> Callable:
        """Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

        This allows for open registration to specify the behavior between the operator
        and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
        or the operator directly.

        Please see :func:`torch.library.register_torch_dispatch` for examples and more details.
        """
    def register_autograd(self, backward: Callable, /, *, setup_context: Callable | None = None) -> None:
        '''Register a backward formula for this custom op.

        In order for an operator to work with autograd, you need to register
        a backward formula:
        1. You must tell us how to compute gradients during the backward pass
        by providing us a "backward" function.
        2. If you need any values from the forward to compute gradients, you can
        use `setup_context` to save values for backward.

        ``backward_fn`` runs during the backward pass. It accepts ``(ctx, *grads)``:
        - ``grads`` is one or more gradients. The number of gradients matches
        the number of outputs of the operator.
        The ``ctx`` object is `the same ctx object <context_method_mixins>`_ used by
        :class:`torch.autograd.Function`. The semantics of ``backward_fn`` are the
        same as :meth:`torch.autograd.Function.backward`.

        ``setup_context(ctx, inputs, output)`` runs during the forward pass.
        Please save quantities needed for backward onto the ``ctx`` object via
        either :meth:`torch.autograd.function.FunctionCtx.save_for_backward`
        or assigning them as attributes of ``ctx``. If your custom op has
        kwarg-only arguments, we expect the signature of ``setup_context``
        to be ``setup_context(ctx, inputs, keyword_only_inputs, output)``.

        Both ``setup_context_fn`` and ``backward_fn`` must be traceable. That is,
        they may not directly access :meth:`torch.Tensor.data_ptr` and they must
        not depend on or mutate global state. If you need a non-traceable backward,
        you can make it a separate custom_op that you call inside ``backward_fn``.

        If you need different autograd behavior on different devices, then we
        recommend creating two different custom operators, one for each device
        that needs different behavior, and switching between them at runtime.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            >>> def numpy_sin(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, output) -> Tensor:
            >>>     x, = inputs
            >>>     ctx.save_for_backward(x)
            >>>
            >>> def backward(ctx, grad):
            >>>     x, = ctx.saved_tensors
            >>>     return grad * x.cos()
            >>>
            >>> numpy_sin.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_sin(x)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, x.cos())
            >>>
            >>> # Example with a keyword-only arg
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, *, val: float) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = x_np * val
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context(ctx, inputs, keyword_only_inputs, output) -> Tensor:
            >>>     ctx.val = keyword_only_inputs["val"]
            >>>
            >>> def backward(ctx, grad):
            >>>     return grad * ctx.val
            >>>
            >>> numpy_mul.register_autograd(backward, setup_context=setup_context)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_mul(x, val=3.14)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

        '''
    _opoverload: Incomplete
    def _register_to_dispatcher(self, tags: Sequence[_C.Tag]) -> None: ...
    def _register_backend_select_dispatcher(self, device_arg_index: int):
        """
        Switch on the device argument to select the correct backend to dispatch to.
        """
    def __call__(self, *args, **kwargs): ...
    def register_vmap(self, func: Callable | None = None):
        '''Register a vmap implementation to support :func:`torch.vmap` for this custom op.

        This API may be used as a decorator.

        In order for an operator to work with :func:`torch.vmap`, you may need to register a
        vmap implementation in the following signature:

            ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

        where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.

        It specifies how do we compute the batched version of ``op`` given inputs with an additional
        dimension (specified by ``in_dims``).

        For each arg in ``args``, ``in_dims`` has a corresponding ``Optional[int]``. It is ``None``
        if the arg is not a Tensor or if the arg is not being vmapped over, otherwise, it is an integer
        specifying what dimension of the Tensor is being vmapped over.

        ``info`` is a collection of additional metadata that may be helpful:
        ``info.batch_size`` specifies the size of the dimension being vmapped over, while
        ``info.randomness`` is the ``randomness`` option that was passed to :func:`torch.vmap`.

        The return of the function ``func`` is a tuple of ``(output, out_dims)``. Similar to ``in_dims``,
        ``out_dims`` should be of the same structure as ``output`` and contain one ``out_dim``
        per output that specifies if the output has the vmapped dimension and what index it is in.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>> from typing import Tuple
            >>>
            >>> def to_numpy(tensor):
            >>>     return tensor.cpu().numpy()
            >>>
            >>> lib = torch.library.Library("mylib", "FRAGMENT")
            >>> @torch.library.custom_op("mylib::numpy_cube", mutates_args=())
            >>> def numpy_cube(x: Tensor) -> Tuple[Tensor, Tensor]:
            >>>     x_np = to_numpy(x)
            >>>     dx = torch.tensor(3 * x_np ** 2, device=x.device)
            >>>     return torch.tensor(x_np ** 3, device=x.device), dx
            >>>
            >>> def numpy_cube_vmap(info, in_dims, x):
            >>>     result = numpy_cube(x)
            >>>     return result, (in_dims[0], in_dims[0])
            >>>
            >>> numpy_cube.register_vmap(numpy_cube_vmap)
            >>>
            >>> x = torch.randn(3)
            >>> torch.vmap(numpy_cube)(x)
            >>>
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
            >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
            >>>
            >>> @numpy_mul.register_vmap
            >>> def numpy_mul_vmap(info, in_dims, x, y):
            >>>     x_bdim, y_bdim = in_dims
            >>>     x = x.movedim(x_bdim, -1) if x_bdim is not None else x.unsqueeze(-1)
            >>>     y = y.movedim(y_bdim, -1) if y_bdim is not None else y.unsqueeze(-1)
            >>>     result = x * y
            >>>     result = result.movedim(-1, 0)
            >>>     return result, 0
            >>>
            >>>
            >>> x = torch.randn(3)
            >>> y = torch.randn(3)
            >>> torch.vmap(numpy_mul)(x, y)
        '''
    def register_autocast(self, device_type: str, cast_inputs: _dtype):
        '''Register an autocast dispatch rule for this custom op.

        Valid `device_type` include: "cpu" and "cuda".

        Args:
            op (str | OpOverload): The operator to register an autocast dispatch rule to.
            device_type(str):  Device type to use. \'cuda\' or \'cpu\'.
                The type is the same as the `type` attribute of a :class:`torch.device`.
                Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
            cast_inputs (:class:`torch.dtype`): When custom op runs in an autocast-enabled region,
                casts incoming floating-point Tensors to the target dtype (non-floating-point Tensors
                are not affected), then executes custom op with autocast disabled.
            lib (Optional[Library]): If provided, the lifetime of this registration

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> import torch
            >>> from torch import Tensor
            >>> from torch.library import custom_op
            >>>
            >>> # Create a custom op that works on cuda
            >>> @torch.library.custom_op("mylib::my_sin", mutates_args=())
            >>> def my_sin(x: Tensor) -> Tensor:
            >>>     return torch.sin(x)
            >>>
            >>> # Register autocast dispatch rule for the cuda device
            >>> torch.library.register_autocast("mylib::my_sin", "cuda", torch.float16)
            >>>
            >>> x = torch.randn(3, dtype=torch.float32, device="cuda")
            >>> with torch.autocast("cuda", dtype=torch.float16):
            >>>     y = torch.ops.mylib.my_sin(x)
            >>> assert y.dtype == torch.float16

        '''

def _cast(value, device_type: str, dtype: _dtype): ...
def increment_version(val: Any) -> None: ...

OPDEF_TO_LIB: dict[str, 'torch.library.Library']
OPDEFS: weakref.WeakValueDictionary

def get_library_allowing_overwrite(namespace: str, name: str) -> torch.library.Library: ...
def _maybe_get_opdef(op: CustomOpDef | _ops.OpOverload | str) -> CustomOpDef | None: ...
