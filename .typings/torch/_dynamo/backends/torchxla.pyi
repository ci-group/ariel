from ..backends.common import aot_autograd as aot_autograd
from .registry import register_backend as register_backend, register_experimental_backend as register_experimental_backend
from _typeshed import Incomplete

log: Incomplete

@register_experimental_backend
def openxla_eval(model, fake_tensor_inputs): ...
def openxla_eval_boxed(model, fake_tensor_inputs): ...
def xla_backend_helper(model, fake_tensor_inputs, boxed: bool = False): ...

openxla: Incomplete
