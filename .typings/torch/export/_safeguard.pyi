from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode as ProxyTorchDispatchMode
from torch.overrides import TorchFunctionMode as TorchFunctionMode

class AutogradStateOpsFailSafeguard(TorchFunctionMode):
    """
    Detect grad state ops during exporting the graph and fail the process by
    raising an error, to avoid unexpected behavior. Those grad mode ops could be:
    `torch.no_grad`
    `torch.enable_grad`
    `torch.set_grad_enabled`

    Export with predispatch mode is exempted.
    """
    def __torch_function__(self, func, types, args=(), kwargs=None): ...
