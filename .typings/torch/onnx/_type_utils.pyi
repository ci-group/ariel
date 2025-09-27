import enum
import torch
from _typeshed import Incomplete
from torch import _C as _C
from torch._C import _onnx as _C_onnx
from torch.onnx import errors as errors

ScalarName: Incomplete
TorchName: Incomplete

class JitScalarType(enum.IntEnum):
    '''Scalar types defined in torch.

    Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_ONNX)
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_value(torch.ones(1, 2)).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_value(torch_c_value_with_type_float).onnx_type()
        TensorProtoDataType.FLOAT

        >>> JitScalarType.from_dtype(torch.get_default_dtype).onnx_type()
        TensorProtoDataType.FLOAT

    '''
    UINT8 = 0
    INT8 = ...
    INT16 = ...
    INT = ...
    INT64 = ...
    HALF = ...
    FLOAT = ...
    DOUBLE = ...
    COMPLEX32 = ...
    COMPLEX64 = ...
    COMPLEX128 = ...
    BOOL = ...
    QINT8 = ...
    QUINT8 = ...
    QINT32 = ...
    BFLOAT16 = ...
    FLOAT8E5M2 = ...
    FLOAT8E4M3FN = ...
    FLOAT8E5M2FNUZ = ...
    FLOAT8E4M3FNUZ = ...
    UNDEFINED = ...
    @classmethod
    def _from_name(cls, name: ScalarName | TorchName | str | None) -> JitScalarType:
        '''Convert a JIT scalar type or torch type name to ScalarType.

        Note: DO NOT USE this API when `name` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
           OnnxExporterError: if name is not a valid scalar type name or if it is None.
        '''
    @classmethod
    def from_dtype(cls, dtype: torch.dtype | None) -> JitScalarType:
        '''Convert a torch dtype to JitScalarType.

        Note: DO NOT USE this API when `dtype` comes from a `torch._C.Value.type()` calls.
            A "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in several scenarios where shape info is not present.
            Instead use `from_value` API which is safer.

        Args:
            dtype: A torch.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        '''
    @classmethod
    def from_onnx_type(cls, onnx_type: int | _C_onnx.TensorProtoDataType | None) -> JitScalarType:
        """Convert a ONNX data type to JitScalarType.

        Args:
            onnx_type: A torch._C._onnx.TensorProtoDataType to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            OnnxExporterError: if dtype is not a valid torch.dtype or if it is None.
        """
    @classmethod
    def from_value(cls, value: None | torch._C.Value | torch.Tensor, default=None) -> JitScalarType:
        """Create a JitScalarType from an value's scalar type.

        Args:
            value: An object to fetch scalar type from.
            default: The JitScalarType to return if a valid scalar cannot be fetched from value

        Returns:
            JitScalarType.

        Raises:
            OnnxExporterError: if value does not have a valid scalar type and default is None.
            SymbolicValueError: when value.type()'s info are empty and default is None
        """
    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
    def torch_name(self) -> TorchName:
        """Convert a JitScalarType to a torch type name."""
    def dtype(self) -> torch.dtype:
        """Convert a JitScalarType to a torch dtype."""
    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""

def valid_scalar_name(scalar_name: ScalarName | str) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
def valid_torch_name(torch_name: TorchName | str) -> bool:
    """Return whether the given torch name is a valid torch type name."""

_SCALAR_TYPE_TO_NAME: dict[JitScalarType, ScalarName]
_SCALAR_NAME_TO_TYPE: dict[ScalarName, JitScalarType]
_SCALAR_TYPE_TO_TORCH_NAME: dict[JitScalarType, TorchName]
_TORCH_NAME_TO_SCALAR_TYPE: dict[TorchName, JitScalarType]
_SCALAR_TYPE_TO_ONNX: Incomplete
_ONNX_TO_SCALAR_TYPE: Incomplete
_SCALAR_TYPE_TO_DTYPE: Incomplete
_DTYPE_TO_SCALAR_TYPE: Incomplete
