import abc
import torch
from _typeshed import Incomplete
from torch._C import DispatchKey as DispatchKey
from torch._dispatch.python import suspend_functionalization as suspend_functionalization
from torch._higher_order_ops.utils import HopInstance as HopInstance, check_input_alias_and_mutation_return_outputs as check_input_alias_and_mutation_return_outputs, materialize_as_graph as materialize_as_graph, reenter_make_fx as reenter_make_fx
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch._subclasses.functional_tensor import disable_functional_mode as disable_functional_mode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode, disable_proxy_modes_tracing as disable_proxy_modes_tracing, track_tensor_tree as track_tensor_tree

class BaseHOP(HigherOrderOperator, abc.ABC):
    '''
    This is the "Base" HOP implementation for a HOP that looks like:

        call_subgraph_hop(subgraph, *operands, **kwargs)

    That is:
    1) the HOP stays alive until Inductor
    2) the HOP\'s semantics are subgraph(*operands)
    3) kwargs may be some config options but aren\'t passed directly to the subgraph.

    To use this, please subclass this class and override methods as necessary:
    ```
    class InvokeQuant(BaseHOP):
        def __init__(self):
            return super().__init__("invoke_quant")

    invoke_quant = InvokeQuant()

    def g(x):
        return x.sin().cos()

    @torch.compile(backend="aot_eager")
    def f(x):
        return invoke_quant(g, x, scheme="nf4")
    ```

    NOTE: don\'t subclass BaseHOP out of tree! That is not allowed. All
    usages must be in tree.
    '''
    def __init__(self, hop_name) -> None: ...
    def __call__(self, subgraph, *operands, **kwargs): ...
    def _call_Autograd(self, subgraph, *operands, **kwargs): ...
    def _call_CompositeExplicitAutograd(self, subgraph, *operands, **kwargs): ...
    def _call_ProxyTorchDispatchMode(self, proxy_mode, subgraph, *operands, **kwargs): ...
    def _call_FakeTensorMode(self, mode, subgraph, *operands, **kwargs): ...
    def _call_Functionalize(self, ctx, subgraph, *operands, **kwargs): ...
    def gen_schema(self, subgraph, *operands, **kwargs): ...

class BaseHOPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hop, subgraph, kwargs, *operands): ...
    @staticmethod
    def backward(ctx, *grad_outputs): ...

class FunctionWithNoFreeVars:
    fn: Incomplete
    def __init__(self, fn) -> None: ...
    def __call__(self, *args, **kwargs): ...
