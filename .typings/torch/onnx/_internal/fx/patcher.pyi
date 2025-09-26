import functools
import io
import types
from _typeshed import Incomplete

@functools.cache
def has_safetensors_and_transformers(): ...

class ONNXTorchPatcher:
    '''Context manager to temporarily patch PyTorch during FX-to-ONNX export.

    This class is a collection of "patches" required by FX-to-ONNX exporter.

    This context overrides several torch functions to support symbolic
    export of large scale models.

    torch.load:
        This function is patched to record the files PyTorch stores model
        parameters and buffers. Downstream FX-to-ONNX exporter can create
        initializers from these files.
    torch.fx._symbolic_trace._wrapped_methods_to_patch:
        This list is extended with (torch.Tensor, "__getitem__") so that
        weight[x, :, y] becomes exportable with torch.fx.symbolic_trace.
    safetensors.torch.load_file:
        This function is patched to allow safetensors to be loaded within
        FakeTensorMode. Remove after https://github.com/huggingface/safetensors/pull/318

    Search for ONNXTorchPatcher in test_fx_to_onnx_with_onnxruntime.py for
    example usage.

    TODO: Should this really be a global patcher? Can we make it a local patcher?
        A reason for splitting this into several patchers is to patch one part of the code
        as a collateral damage of patching another part of the code. For example, we
        for tracing model with torch._dynamo.export, we don\'t need to patch
        `torch.fx._symbolic_trace._wrapped_methods_to_patch`
    '''
    paths: list[str | io.BufferedIOBase]
    torch_load: Incomplete
    torch_load_wrapper: Incomplete
    safetensors_torch_load_file: Incomplete
    safetensors_torch_load_file_wrapper: Incomplete
    transformers_modeling_utils_safe_load_file: Incomplete
    def __init__(self) -> None: ...
    torch_fx__symbolic_trace__wrapped_methods_to_patch: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
