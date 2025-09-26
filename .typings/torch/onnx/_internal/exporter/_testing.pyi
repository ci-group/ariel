from torch.onnx._internal.exporter import _onnx_program
from typing import Any

__all__ = ['assert_onnx_program']

def assert_onnx_program(program: _onnx_program.ONNXProgram, *, rtol: float | None = None, atol: float | None = None, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, strategy: str | None = 'TorchExportNonStrictStrategy') -> None:
    '''Assert that the ONNX model produces the same output as the PyTorch ExportedProgram.

    Args:
        program: The ``ONNXProgram`` to verify.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        args: The positional arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        kwargs: The keyword arguments to pass to the program.
            If None, the default example inputs in the ExportedProgram will be used.
        strategy: Assert the capture strategy used to export the program. Values can be
            class names like "TorchExportNonStrictStrategy".
            If None, the strategy is not asserted.
    '''
