import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch._inductor import config as config
from torch._inductor.autotune_process import BenchmarkRequest as BenchmarkRequest, GPUDeviceBenchmarkMixin as GPUDeviceBenchmarkMixin, TensorMeta as TensorMeta
from torch._inductor.codecache import DLLWrapper as DLLWrapper, ROCmCodeCache as ROCmCodeCache
from typing import Any, Callable

log: Incomplete

class ROCmBenchmarkRequest(GPUDeviceBenchmarkMixin, BenchmarkRequest):
    source_code: Incomplete
    workspace_size: int
    workspace: torch.Tensor | None
    DLL: DLLWrapper | None
    _workspace_size_updated: bool
    hash_key: str
    source_file: str
    def __init__(self, kernel_name: str, input_tensor_meta: TensorMeta | list[TensorMeta], output_tensor_meta: TensorMeta | list[TensorMeta], extra_args: Iterable[Any], source_code: str) -> None: ...
    def precompile(self) -> None: ...
    def make_run_fn(self, *input_tensors: torch.Tensor, out: torch.Tensor) -> Callable[[], None]: ...
    def update_workspace_size(self) -> None: ...
    def ensure_dll_loaded(self) -> None: ...
    def cleanup_run_fn(self) -> None: ...
    def __str__(self) -> str: ...
