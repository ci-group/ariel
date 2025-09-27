import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from torch._library.utils import Kernel as Kernel, RegistrationHandle as RegistrationHandle
from typing import Callable

class FakeImplHolder:
    """A holder where one can register an fake impl to."""
    qualname: str
    kernels: list[Kernel]
    def __init__(self, qualname: str) -> None: ...
    @property
    def kernel(self): ...
    @kernel.setter
    def kernel(self, value) -> None: ...
    def register(self, func: Callable, source: str, lib, *, allow_override: bool = False) -> RegistrationHandle:
        """Register an fake impl.

        Returns a RegistrationHandle that one can use to de-register this
        fake impl.
        """

def construct_meta_kernel(qualname: str, fake_impl_holder: FakeImplHolder) -> Callable: ...
def get_none() -> None: ...

global_ctx_getter: Callable

@contextlib.contextmanager
def set_ctx_getter(ctx_getter) -> Generator[None]: ...

class FakeImplCtx:
    """
    Context object for writing fake implementations for custom operators.
    """
    _fake_mode: Incomplete
    _shape_env: Incomplete
    _op: Incomplete
    def __init__(self, _fake_mode, _op) -> None: ...
    def create_unbacked_symint(self, *, min: int = 2, max=None) -> torch.SymInt: ...
    def new_dynamic_size(self, *, min: int = 0, max=None) -> torch.SymInt:
        '''Constructs a new symint (symbolic int) representing a data-dependent value.

        This is useful for writing the fake implementation (which is necessary
        for torch.compile) for a CustomOp where an output Tensor has a size
        that depends on the data of the input Tensors.

        Args:
            min (int): A statically known inclusive lower bound for this symint. Default: 0
            max (Optional[int]): A statically known inclusive upper bound for this
                symint. Default: None

        .. warning:

            It is important that the ``min`` and ``max`` (if not None) values are set
            correctly, otherwise, there will be undefined behavior under
            torch.compile. The default value of ``min`` is 2 due to torch.compile
            specializing on 0/1 sizes.

            You must also verify that your implementation on concrete Tensors
            (e.g. CPU/CUDA) only returns Tensors where the size that corresponds
            to the symint also has respects these constraint.
            The easiest way to do this is to add an assertion in the CPU/CUDA/etc
            implementation that the size follows these bounds.

        Example::

            >>> # An operator with data-dependent output shape
            >>> lib = torch.library.Library("mymodule", "FRAGMENT")
            >>> lib.define("mymodule::custom_nonzero(Tensor x) -> Tensor")
            >>>
            >>> @torch.library.register_fake("mymodule::custom_nonzero")
            >>> def _(x):
            >>>     # Number of nonzero-elements is data-dependent.
            >>>     # Since we cannot peek at the data in an fake impl,
            >>>     # we use the ctx object to construct a new symint that
            >>>     # represents the data-dependent size.
            >>>     ctx = torch.library.get_ctx()
            >>>     nnz = ctx.new_dynamic_size()
            >>>     shape = [nnz, x.dim()]
            >>>     result = x.new_empty(shape, dtype=torch.int64)
            >>>     return result
            >>>
            >>> @torch.library.impl(lib, "custom_nonzero", "CPU")
            >>> def _(x):
            >>>     x_np = x.numpy()
            >>>     res = np.stack(np.nonzero(x_np), axis=1)
            >>>     return torch.tensor(res, device=x.device)

        '''

def allocate_size(shape_env, min_val: int = 0, max_val=None): ...
