from torch._prims.context import TorchRefsMode as TorchRefsMode
from torch.fx import GraphModule as GraphModule
from torch.fx.experimental.proxy_tensor import make_fx as make_fx, wrapper_and_args_for_make_fx as wrapper_and_args_for_make_fx
from typing import Any, Callable, TypeVar
from typing_extensions import ParamSpec, TypeVarTuple, Unpack

T = TypeVar('T')
P = ParamSpec('P')
Ts = TypeVarTuple('Ts')

def execute(gm: GraphModule, *args: Unpack[Ts], executor: str = 'aten', executor_parameters: dict | None = None) -> Any:
    """
    Prototype ATen executor.

    Just executes the context's graph.
    """
def make_traced(fn: Callable[P, T]) -> Callable[P, T]:
    """
    Returns a function that, when called, will
    trace its torch operations to prims and then
    execute those prims on the requested trace executor
    (possibly lowering them to that trace executor first).

    Only supports the torch operations defined in _torch_to_reference_map
    in context.py and operations with positional args. All args must
    be tensors.
    In the near future all these restrictions will be lifted.

    Example usage:

    def foo(a, b):
      return torch.add(a, b)

    traced_foo = make_traced(foo)

    a = torch.randn((1, 2, 3, 4, 5), device='cuda')
    b = torch.randn((1, 2, 3, 4, 5), device='cuda')
    result = traced_foo(a, b, executor='aten')
    """
