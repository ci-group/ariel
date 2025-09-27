import contextlib
from .custom_ops import CustomOpDef as CustomOpDef, custom_op as custom_op
from .infer_schema import infer_schema as infer_schema
from _typeshed import Incomplete
from collections.abc import Generator, Iterable
from torch.utils._exposed_in import exposed_in as exposed_in
from typing import Any, Callable

def triton_op(name: str, fn: Callable | None = None, /, *, mutates_args: str | Iterable[str], schema: str | None = None) -> Callable:
    '''Create a custom operator whose implementation is backed by 1+ triton kernels.

    This is a more structured way of using triton kernels with PyTorch.
    Prefer using triton kernels with no ``torch.library`` custom operator wrappers
    (like :func:`torch.library.custom_op`, :func:`torch.library.triton_op`) because
    that is simpler;
    only use :func:`torch.library.custom_op`/:func:`torch.library.triton_op` if you
    want to create an operator that behaves like PyTorch built-in operators.
    For example, you may use a ``torch.library`` wrapper API to define the
    behavior of the triton kernel when passed a tensor subclass or under
    a TorchDispatchMode.

    Use :func:`torch.library.triton_op` instead of :func:`torch.library.custom_op`
    when the implementation
    consists of 1+ triton kernels. :func:`torch.library.custom_op` treats
    custom operators as opaque (:func:`torch.compile` and
    :func:`torch.export.export` will never trace into them), but ``triton_op``
    makes the implementation visible to these subsystems, allowing them
    to optimize the triton kernel(s).

    Note that ``fn`` must only consist of calls to PyTorch-understood
    operators and triton kernels. Any triton kernels called inside ``fn``
    must be wrapped in a call to :func:`torch.library.wrap_triton`.

    Args:
        name (str): A name for the custom op that looks like "{namespace}::{name}",
            e.g. "mylib::my_linear". The name is used as the op\'s stable identifier
            in PyTorch subsystems (e.g. torch.export, FX graphs).
            To avoid name collisions, please use your project name as the namespace;
            e.g. all custom ops in pytorch/fbgemm use "fbgemm" as the namespace.
        mutates_args (Iterable[str] or "unknown"): The names of args that the function mutates.
            This MUST be accurate, otherwise, the behavior is undefined. If "unknown",
            it pessimistically assumes that all inputs to the operator are being mutated.
        schema (None | str): A schema string for the operator. If None
            (recommended) we\'ll infer a schema for the operator from its type
            annotations. We recommend letting us infer a schema unless you
            have a specific reason not to.
            Example: "(Tensor x, int y) -> (Tensor, Tensor)".

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> import torch
        >>> from torch.library import triton_op, wrap_triton
        >>>
        >>> import triton
        >>> from triton import language as tl
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> @triton_op("mylib::add", mutates_args={})
        >>> def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     # NB: we need to wrap the triton kernel in a call to wrap_triton
        >>>     wrap_triton(add_kernel)[grid](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> @torch.compile
        >>> def f(x, y):
        >>>     return add(x, y)
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>>
        >>> z = f(x, y)
        >>> assert torch.allclose(z, x + y)

    '''

wrap_triton_enabled: Incomplete
wrap_triton_enabled_default: bool

@contextlib.contextmanager
def set_wrap_triton_enabled(enabled: bool) -> Generator[None, None, None]:
    """If triton kernels annotated with @wrap_triton should dispatch via HOP
    or go straight to the triton kernel execution.

    We have this switch because eager-mode performance of HOP dispatch is slow
    enough to matter (~1ms) and we know that wrap_triton isn't necessary in
    some situations (eager-mode with regular Tensors)
    """
def is_wrap_triton_enabled() -> bool: ...
def capture_triton(triton_kernel: Callable, /) -> Any:
    """This API has been renamed to wrap_triton"""
def wrap_triton(triton_kernel: Callable, /) -> Any:
    '''Allows capture of a triton kernel into a graph via make_fx or
    non-strict ``torch.export``.

    These technologies perform Dispatcher-based tracing (via
    ``__torch_dispatch__``) and cannot see calls to raw triton kernels.
    The ``wrap_triton`` API wraps a triton kernel into a callable that
    can actually be traced into a graph.

    Please use this API together with :func:`torch.library.triton_op`.

    Examples:

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> import triton
        >>> from triton import language as tl
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.library import wrap_triton
        >>>
        >>> @triton.jit
        >>> def add_kernel(
        >>>     in_ptr0,
        >>>     in_ptr1,
        >>>     out_ptr,
        >>>     n_elements,
        >>>     BLOCK_SIZE: "tl.constexpr",
        >>> ):
        >>>     pid = tl.program_id(axis=0)
        >>>     block_start = pid * BLOCK_SIZE
        >>>     offsets = block_start + tl.arange(0, BLOCK_SIZE)
        >>>     mask = offsets < n_elements
        >>>     x = tl.load(in_ptr0 + offsets, mask=mask)
        >>>     y = tl.load(in_ptr1 + offsets, mask=mask)
        >>>     output = x + y
        >>>     tl.store(out_ptr + offsets, output, mask=mask)
        >>>
        >>> def add(x, y):
        >>>     output = torch.empty_like(x)
        >>>     n_elements = output.numel()
        >>>
        >>>     def grid_fn(meta):
        >>>         return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        >>>
        >>>     wrap_triton(add_kernel)[grid_fn](x, y, output, n_elements, 16)
        >>>     return output
        >>>
        >>> x = torch.randn(3, device="cuda")
        >>> y = torch.randn(3, device="cuda")
        >>> gm = make_fx(add)(x, y)
        >>> print(gm.code)
        >>> # def forward(self, x_1, y_1):
        >>> #     empty_like = torch.ops.aten.empty_like.default(x_1, pin_memory = False)
        >>> #     triton_kernel_wrapper_mutation_proxy = triton_kernel_wrapper_mutation(
        >>> #         kernel_idx = 0, constant_args_idx = 0,
        >>> #         grid = [(1, 1, 1)], kwargs = {
        >>> #             \'in_ptr0\': x_1, \'in_ptr1\': y_1, \'out_ptr\': empty_like,
        >>> #             \'n_elements\': 3, \'BLOCK_SIZE\': 16
        >>> #         })
        >>> #     return empty_like

    '''
