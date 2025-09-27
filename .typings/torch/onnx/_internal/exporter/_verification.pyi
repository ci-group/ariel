import dataclasses
import torch
from _typeshed import Incomplete
from torch.onnx._internal.exporter import _onnx_program
from typing import Any

__all__ = ['VerificationInfo', 'verify_onnx_program']

@dataclasses.dataclass
class VerificationInfo:
    """Verification information for a value in the ONNX program.

    This class contains the maximum absolute difference, maximum relative difference,
    and histograms of absolute and relative differences between the expected and actual
    values. It also includes the expected and actual data types.

    The histograms are represented as tuples of tensors, where the first tensor is the
    histogram counts and the second tensor is the bin edges.

    Attributes:
        name: The name of the value (output or intermediate).
        max_abs_diff: The maximum absolute difference between the expected and actual values.
        max_rel_diff: The maximum relative difference between the expected and actual values.
        abs_diff_hist: A tuple of tensors representing the histogram of absolute differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        rel_diff_hist: A tuple of tensors representing the histogram of relative differences.
            The first tensor is the histogram counts and the second tensor is the bin edges.
        expected_dtype: The data type of the expected value.
        actual_dtype: The data type of the actual value.
    """
    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    @classmethod
    def from_tensors(cls, name: str, expected: torch.Tensor | float | int | bool, actual: torch.Tensor | float | int | bool) -> VerificationInfo:
        """Create a VerificationInfo object from two tensors.

        Args:
            name: The name of the value.
            expected: The expected tensor.
            actual: The actual tensor.

        Returns:
            VerificationInfo: The VerificationInfo object.
        """
    def asdict(self) -> dict[str, Any]:
        """Convert the VerificationInfo object to a dictionary.

        Returns:
            A dictionary representation of the VerificationInfo object.
        """

def verify_onnx_program(onnx_program: _onnx_program.ONNXProgram, args: tuple[Any, ...] | None = None, kwargs: dict[str, Any] | None = None, compare_intermediates: bool = False) -> list[VerificationInfo]:
    """Verify the ONNX model by comparing the values with the expected values from ExportedProgram.

    Args:
        onnx_program: The ONNX program to verify.
        args: The input arguments for the model.
        kwargs: The keyword arguments for the model.
        compare_intermediates: Whether to verify intermediate values. This is going
            to take longer time, so it is disabled by default.

    Returns:
        VerificationInfo objects containing the verification information for each value.
    """

class _VerificationInterpreter(torch.fx.Interpreter):
    '''Interpreter for verifying converted ONNX model accuracy by comparing intermediate values.

    To compare models, first initialize the interpreter with an ONNX program.
    Then, call the :meth:`run` method with the input arguments to execute the model.
    The :meth:`run` method will execute the model and populate the
    :attr:`verification_infos` attribute with the verification information for each value.

    ::
        onnx_program = torch.onnx.export(model, args, dynamo=True)
        interpreter = _VerificationInterpreter(onnx_program)
        interpreter.run(*args)
        verification_infos = interpreter.verification_infos
        for info in verification_infos:
            print("value name:", info.name, info)

    The verification information includes the maximum absolute difference, maximum relative
    difference, and histograms of absolute and relative differences between the expected
    and actual values. See :class:`VerificationInfo` for more details.

    Attributes:
        verification_infos: A list of verification information for each value.
            It is populated when the `run` method is called.
    '''
    _onnx_program: Incomplete
    _onnx_values: Incomplete
    _args: tuple[Any, ...]
    verification_infos: list[VerificationInfo]
    def __init__(self, onnx_program: torch.onnx.ONNXProgram) -> None:
        """Initialize the _VerificationInterpreter with an ONNX program.

        Args:
            onnx_program: The ONNX program to verify.
        """
    def run(self, *args: Any, initial_env: dict[torch.fx.Node, Any] | None = None, enable_io_processing: bool = True) -> Any:
        """Run the interpreter with the given input arguments.

        This method executes the model and populates the :attr:`verification_infos` attribute
        with the verification information for each value.

        Args:
            args: The input arguments for the model.
            initial_env: The initial environment for the interpreter.
            enable_io_processing: Whether to enable IO processing.

        Returns:
            Any: The result of executing the model.
        """
    def run_node(self, n: torch.fx.Node) -> Any: ...
