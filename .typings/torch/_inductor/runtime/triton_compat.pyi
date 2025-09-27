import torch
import triton as triton
import triton.language as tl
from _typeshed import Incomplete
from triton import Config as Config, knobs as knobs
from triton.backends.compiler import GPUTarget as GPUTarget
from triton.compiler import CompiledKernel as CompiledKernel
from triton.compiler.compiler import ASTSource as ASTSource
from triton.language.extra import libdevice as libdevice
from triton.language.standard import _log2 as _log2
from triton.runtime.autotuner import OutOfResources as OutOfResources, PTXASError as PTXASError
from triton.runtime.jit import KernelInterface as KernelInterface
from typing import Any

__all__ = ['Config', 'CompiledKernel', 'OutOfResources', 'KernelInterface', 'PTXASError', 'ASTSource', 'GPUTarget', 'tl', '_log2', 'libdevice', 'math', 'triton', 'cc_warp_size', 'knobs']

class PTXASError(Exception): ...

math: Incomplete
math = tl

class OutOfResources(Exception): ...
class PTXASError(Exception): ...
Config = object
CompiledKernel = object
KernelInterface = object
_log2 = _raise_error

class triton:
    @staticmethod
    def jit(*args: Any, **kwargs: Any) -> Any: ...

class tl:
    @staticmethod
    def constexpr(val: Any) -> Any: ...
    tensor = Any
    dtype = Any

def cc_warp_size(cc: str | int) -> int: ...
autograd_profiler = torch.autograd.profiler

class autograd_profiler:
    _is_profiler_enabled: bool
