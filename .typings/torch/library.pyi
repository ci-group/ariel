import contextlib
import functools
import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._library.custom_ops import custom_op as custom_op
from torch._library.infer_schema import infer_schema as infer_schema
from torch._library.triton import triton_op as triton_op, wrap_triton as wrap_triton
from torch.types import _dtype
from typing import Any, Callable, Literal, TypeVar, overload
from typing_extensions import ParamSpec

__all__ = ['Library', 'impl', 'define', 'fallthrough_kernel', 'impl_abstract', 'register_autocast', 'register_fake', 'register_torch_dispatch', 'register_vmap', 'get_ctx', 'custom_op', 'triton_op', 'wrap_triton', 'infer_schema']

_T = TypeVar('_T')
_P = ParamSpec('_P')

def fallthrough_kernel() -> None:
    """
    A dummy function to pass to ``Library.impl`` in order to register a fallthrough.
    """

class Library:
    '''
    A class to create libraries that can be used to register new operators or
    override operators in existing libraries from Python.
    A user can optionally pass in a dispatch keyname if they only want to register
    kernels corresponding to only one specific dispatch key.

    To create a library to override operators in an existing library (with name ns), set the kind to "IMPL".
    To create a new library (with name ns) to register new operators, set the kind to "DEF".
    To create a fragment of a possibly existing library to register operators (and bypass
    the limitation that there is only one library for a given namespace), set the kind to
    "FRAGMENT".

    Args:
        ns: library name
        kind: "DEF", "IMPL", "FRAGMENT"
        dispatch_key: PyTorch dispatch key (default: "")
    '''
    m: Any | None
    ns: Incomplete
    _op_defs: set[str]
    _op_impls: set[str]
    _registration_handles: list[torch._library.utils.RegistrationHandle]
    kind: Incomplete
    dispatch_key: Incomplete
    def __init__(self, ns, kind, dispatch_key: str = '') -> None: ...
    def __repr__(self) -> str: ...
    def define(self, schema, alias_analysis: str = '', *, tags=()):
        '''Defines a new operator and its semantics in the ns namespace.

        Args:
            schema: function schema to define a new operator.
            alias_analysis (optional): Indicates if the aliasing properties of the operator arguments can be
                                       inferred from the schema (default behavior) or not ("CONSERVATIVE").
            tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
                                       operator. Tagging an operator changes the operator\'s behavior
                                       under various PyTorch subsystems; please read the docs for the
                                       torch.Tag carefully before applying it.

        Returns:
            name of the operator as inferred from the schema.

        Example::

            >>> my_lib = Library("mylib", "DEF")
            >>> my_lib.define("sum(Tensor self) -> Tensor")
        '''
    def _register_fake(self, op_name, fn, _stacklevel: int = 1, *, allow_override: bool = False) -> None:
        """Registers the fake impl for an operator defined in the library."""
    def _register_torch_dispatch_rule(self, op_name, torch_dispatch_class, fn) -> None:
        """Registers a torch_dispatch rule for the given operator and torch_dispatch_class.

        This allows for open registration to specify the behavior between the operator
        and the torch_dispatch_class without needing to modify the torch_dispatch_class
        or the operator directly.

        The torch_dispatch_class is either a Tensor subclass with `__torch_dispatch__` or a
        TorchDispatchMode.

        If it is a Tensor subclass, we expect fn to have the following signature:
        (cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any

        If it is a TorchDispatchMode, we expect fn to have the following signature:
        (mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any
        """
    def _impl_with_aoti_compile(self, op_name, dispatch_key: str = '') -> None:
        '''Register the operator to use the AOTI-compiled implementation.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.

        Example::

            >>> my_lib = Library("aten", "IMPL")
            >>> my_lib._impl_with_aoti_compile("div.Tensor", "CPU")
        '''
    def impl(self, op_name, fn, dispatch_key: str = '', *, with_keyset: bool = False, allow_override: bool = False) -> None:
        '''Registers the function implementation for an operator defined in the library.

        Args:
            op_name: operator name (along with the overload) or OpOverload object.
            fn: function that\'s the operator implementation for the input dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.
            allow_override: Flag controlling if we want to override an
                         existing registered kernel implementation. This is by
                         default off, and will error you\'re trying to register a
                         kernel to a dispatch key with a kernel already
                         registered.

        Example::

            >>> my_lib = Library("aten", "IMPL")
            >>> def div_cpu(self, other):
            >>>     return self * (1 / other)
            >>> my_lib.impl("div.Tensor", div_cpu, "CPU")
        '''
    def fallback(self, fn, dispatch_key: str = '', *, with_keyset: bool = False) -> None:
        '''Registers the function implementation as the fallback for the given key.

        This function only works for a library with global namespace ("_").

        Args:
            fn: function used as fallback for the given dispatch key or :func:`~fallthrough_kernel`
                to register a fallthrough.
            dispatch_key: dispatch key that the input function should be registered for. By default, it uses
                          the dispatch key that the library was created with.
            with_keyset: flag controlling if the current dispatcher call keyset should be passed as the first argument
                         to :attr:`fn` when calling. This should be used to create the appropriate keyset for redispatch calls.

        Example::

            >>> my_lib = Library("_", "IMPL")
            >>> def fallback_kernel(op, *args, **kwargs):
            >>>     # Handle all autocast ops generically
            >>>     # ...
            >>> my_lib.fallback(fallback_kernel, "Autocast")
        '''
    def _destroy(self) -> None: ...

@functools.singledispatch
def define(qualname, schema, *, lib=None, tags=()) -> None:
    '''Defines a new operator.

    In PyTorch, defining an op (short for "operator") is a two step-process:
    - we need to define the op (by providing an operator name and schema)
    - we need to implement behavior for how the operator interacts with
    various PyTorch subsystems, like CPU/CUDA Tensors, Autograd, etc.

    This entrypoint defines the custom operator (the first step)
    you must then perform the second step by calling various
    ``impl_*`` APIs, like :func:`torch.library.impl` or
    :func:`torch.library.register_fake`.

    Args:
        qualname (str): The qualified name for the operator. Should be
            a string that looks like "namespace::name", e.g. "aten::sin".
            Operators in PyTorch need a namespace to
            avoid name collisions; a given operator may only be created once.
            If you are writing a Python library, we recommend the namespace to
            be the name of your top-level module.
        schema (str): The schema of the operator. E.g. "(Tensor x) -> Tensor"
            for an op that accepts one Tensor and returns one Tensor. It does
            not contain the operator name (that is passed in ``qualname``).
        lib (Optional[Library]): If provided, the lifetime of this operator
            will be tied to the lifetime of the Library object.
        tags (Tag | Sequence[Tag]): one or more torch.Tag to apply to this
            operator. Tagging an operator changes the operator\'s behavior
            under various PyTorch subsystems; please read the docs for the
            torch.Tag carefully before applying it.

    Example::
        >>> import torch
        >>> import numpy as np
        >>>
        >>> # Define the operator
        >>> torch.library.define("mylib::sin", "(Tensor x) -> Tensor")
        >>>
        >>> # Add implementations for the operator
        >>> @torch.library.impl("mylib::sin", "cpu")
        >>> def f(x):
        >>>     return torch.from_numpy(np.sin(x.numpy()))
        >>>
        >>> # Call the new operator from torch.ops.
        >>> x = torch.randn(3)
        >>> y = torch.ops.mylib.sin(x)
        >>> assert torch.allclose(y, x.sin())

    '''
@overload
def impl(qualname: str, types: str | Sequence[str], func: Literal[None] = None, *, lib: Library | None = None) -> Callable[[Callable[..., object]], None]: ...
@overload
def impl(qualname: str, types: str | Sequence[str], func: Callable[..., object], *, lib: Library | None = None) -> None: ...
@overload
def impl(lib: Library, name: str, dispatch_key: str = '') -> Callable[[Callable[_P, _T]], Callable[_P, _T]]: ...
def impl_abstract(qualname, func=None, *, lib=None, _stacklevel: int = 1):
    """This API was renamed to :func:`torch.library.register_fake` in PyTorch 2.4.
    Please use that instead.
    """
def register_autocast(op: _op_identifier, device_type: str, cast_inputs: _dtype, /, *, lib: Library | None = None):
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
def register_fake(op: _op_identifier, func: Callable | None = None, /, *, lib: Library | None = None, _stacklevel: int = 1, allow_override: bool = False):
    '''Register a FakeTensor implementation ("fake impl") for this operator.

    Also sometimes known as a "meta kernel", "abstract impl".

    An "FakeTensor implementation" specifies the behavior of this operator on
    Tensors that carry no data ("FakeTensor"). Given some input Tensors with
    certain properties (sizes/strides/storage_offset/device), it specifies
    what the properties of the output Tensors are.

    The FakeTensor implementation has the same signature as the operator.
    It is run for both FakeTensors and meta tensors. To write a FakeTensor
    implementation, assume that all Tensor inputs to the operator are
    regular CPU/CUDA/Meta tensors, but they do not have storage, and
    you are trying to return regular CPU/CUDA/Meta tensor(s) as output.
    The FakeTensor implementation must consist of only PyTorch operations
    (and may not directly access the storage or data of any input or
    intermediate Tensors).

    This API may be used as a decorator (see examples).

    For a detailed guide on custom ops, please see
    https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html

    Args:
        op_name: Operator name (along with the overload) or OpOverload object.
        func: Fake tensor implementation.
        lib (Optional[Library]): Library to register the fake tensor to.
        allow_override: Flag controlling if we want to override an
                        existing registered fake impl. This is by default off,
                        and will error you\'re trying to register a fake impl to
                        an operator that already has a fake impl. This also only
                        applies if the custom operator was not created via
                        torch.library.custom_op, as overriding and existing fake
                        impl is already allowed.

    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from torch import Tensor
        >>>
        >>> # Example 1: an operator without data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_linear", mutates_args=())
        >>> def custom_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
        >>>     raise NotImplementedError("Implementation goes here")
        >>>
        >>> @torch.library.register_fake("mylib::custom_linear")
        >>> def _(x, weight, bias):
        >>>     assert x.dim() == 2
        >>>     assert weight.dim() == 2
        >>>     assert bias.dim() == 1
        >>>     assert x.shape[1] == weight.shape[1]
        >>>     assert weight.shape[0] == bias.shape[0]
        >>>     assert x.device == weight.device
        >>>
        >>>     return (x @ weight.t()) + bias
        >>>
        >>> with torch._subclasses.fake_tensor.FakeTensorMode():
        >>>     x = torch.randn(2, 3)
        >>>     w = torch.randn(3, 3)
        >>>     b = torch.randn(3)
        >>>     y = torch.ops.mylib.custom_linear(x, w, b)
        >>>
        >>> assert y.shape == (2, 3)
        >>>
        >>> # Example 2: an operator with data-dependent output shape
        >>> @torch.library.custom_op("mylib::custom_nonzero", mutates_args=())
        >>> def custom_nonzero(x: Tensor) -> Tensor:
        >>>     x_np = x.numpy(force=True)
        >>>     res = np.stack(np.nonzero(x_np), axis=1)
        >>>     return torch.tensor(res, device=x.device)
        >>>
        >>> @torch.library.register_fake("mylib::custom_nonzero")
        >>> def _(x):
        >>> # Number of nonzero-elements is data-dependent.
        >>> # Since we cannot peek at the data in an fake impl,
        >>> # we use the ctx object to construct a new symint that
        >>> # represents the data-dependent size.
        >>>     ctx = torch.library.get_ctx()
        >>>     nnz = ctx.new_dynamic_size()
        >>>     shape = [nnz, x.dim()]
        >>>     result = x.new_empty(shape, dtype=torch.int64)
        >>>     return result
        >>>
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>>
        >>> x = torch.tensor([0, 1, 2, 3, 4, 0])
        >>> trace = make_fx(torch.ops.mylib.custom_nonzero, tracing_mode="symbolic")(x)
        >>> trace.print_readable()
        >>>
        >>> assert torch.allclose(trace(x), torch.ops.mylib.custom_nonzero(x))

    '''
def register_torch_dispatch(op: _op_identifier, torch_dispatch_class: Any, func: Callable | None = None, /, *, lib: Library | None = None):
    '''Registers a torch_dispatch rule for the given operator and ``torch_dispatch_class``.

    This allows for open registration to specify the behavior between the operator
    and the ``torch_dispatch_class`` without needing to modify the ``torch_dispatch_class``
    or the operator directly.

    The ``torch_dispatch_class`` is either a Tensor subclass with ``__torch_dispatch__`` or a
    TorchDispatchMode.

    If it is a Tensor subclass, we expect ``func`` to have the following signature:
    ``(cls, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    If it is a TorchDispatchMode, we expect ``func`` to have the following signature:
    ``(mode, func: OpOverload, types: Tuple[type, ...], args, kwargs) -> Any``

    ``args`` and ``kwargs`` will have been normalized the same way they are
    in ``__torch_dispatch__`` (see :ref:`torch-dispatch-calling-convention`).

    Examples:

        >>> import torch
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args={})
        >>> def foo(x: torch.Tensor) -> torch.Tensor:
        >>>     return x.clone()
        >>>
        >>> class MyMode(torch.utils._python_dispatch.TorchDispatchMode):
        >>>     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        >>>         return func(*args, **kwargs)
        >>>
        >>> @torch.library.register_torch_dispatch("mylib::foo", MyMode)
        >>> def _(mode, func, types, args, kwargs):
        >>>     x, = args
        >>>     return x + 1
        >>>
        >>> x = torch.randn(3)
        >>> y = foo(x)
        >>> assert torch.allclose(y, x)
        >>>
        >>> with MyMode():
        >>>     y = foo(x)
        >>> assert torch.allclose(y, x + 1)

    '''
def register_vmap(op: _op_identifier, func: Callable | None = None, /, *, lib=None):
    '''Register a vmap implementation to support :func:`torch.vmap` for this custom op.

    This API may be used as a decorator (see examples).

    In order for an operator to work with :func:`torch.vmap`, you may need to register a
    vmap implementation in the following signature:

        ``vmap_func(info, in_dims: Tuple[Optional[int]], *args, **kwargs)``,

    where ``*args`` and ``**kwargs`` are the arguments and kwargs for ``op``.
    We do not support kwarg-only Tensor args.

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
        >>> torch.library.register_vmap(numpy_cube, numpy_cube_vmap)
        >>>
        >>> x = torch.randn(3)
        >>> torch.vmap(numpy_cube)(x)
        >>>
        >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
        >>> def numpy_mul(x: Tensor, y: Tensor) -> Tensor:
        >>>     return torch.tensor(to_numpy(x) * to_numpy(y), device=x.device)
        >>>
        >>> @torch.library.register_vmap("mylib::numpy_mul")
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

    .. note::
        The vmap function should aim to preserve the semantics of the entire custom operator.
        That is, ``grad(vmap(op))`` should be replaceable with a ``grad(map(op))``.

        If your custom operator has any custom behavior in the backward pass, please
        keep this in mind.

    '''
def get_ctx() -> torch._library.fake_impl.FakeImplCtx:
    """get_ctx() returns the current AbstractImplCtx object.

    Calling ``get_ctx()`` is only valid inside of an fake impl
    (see :func:`torch.library.register_fake` for more usage details.
    """
