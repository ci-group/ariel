import functools
from .common import device_from_inputs as device_from_inputs, fake_tensor_unsupported as fake_tensor_unsupported
from .registry import register_backend as register_backend
from _typeshed import Incomplete
from types import MappingProxyType

log: Incomplete

@register_backend
@fake_tensor_unsupported
def tvm(gm, example_inputs, *, options: MappingProxyType | None = ...): ...

tvm_meta_schedule: Incomplete
tvm_auto_scheduler: Incomplete

def has_tvm(): ...
@functools.cache
def llvm_target(): ...
