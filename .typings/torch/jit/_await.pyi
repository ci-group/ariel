from torch._jit_internal import _Await as _Await
from torch.jit._builtins import _register_builtin as _register_builtin
from torch.utils import set_module as set_module

def _awaitable(func, *args, **kwargs):
    """Create Await object that will call specified functioni with specified args, when it is requested for the result."""
def _awaitable_wait(aw):
    """Request await the result of execution, if Await is not completed yet, the func will be called immediately."""
def _awaitable_nowait(o):
    """Create completed Await with specified result."""
