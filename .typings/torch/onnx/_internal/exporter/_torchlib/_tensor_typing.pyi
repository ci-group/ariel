from onnxscript import BFLOAT16, BOOL, COMPLEX128, COMPLEX64, DOUBLE, FLOAT, FLOAT16, INT16, INT32, INT64, INT8, STRING, UINT8
from typing import TypeVar

TensorType = BFLOAT16 | BOOL | COMPLEX64 | COMPLEX128 | DOUBLE | FLOAT | FLOAT16 | INT8 | INT16 | INT32 | INT64 | UINT8
_FloatType = FLOAT16 | FLOAT | DOUBLE | BFLOAT16
IntType = INT8 | INT16 | INT32 | INT64
RealType = BFLOAT16 | FLOAT16 | FLOAT | DOUBLE | INT8 | INT16 | INT32 | INT64
TTensor = TypeVar('TTensor', bound=TensorType)
TTensor2 = TypeVar('TTensor2', bound=TensorType)
TTensorOrString = TypeVar('TTensorOrString', bound=TensorType | STRING)
TFloat = TypeVar('TFloat', bound=_FloatType)
TFloatOrUInt8 = TypeVar('TFloatOrUInt8', bound=FLOAT | FLOAT16 | DOUBLE | INT8 | UINT8)
TInt = TypeVar('TInt', bound=IntType)
TReal = TypeVar('TReal', bound=RealType)
TRealUnlessInt16OrInt8 = TypeVar('TRealUnlessInt16OrInt8', bound=FLOAT16 | FLOAT | DOUBLE | BFLOAT16 | INT32 | INT64)
TRealUnlessFloat16OrInt8 = TypeVar('TRealUnlessFloat16OrInt8', bound=DOUBLE | FLOAT | INT16 | INT32 | INT64)
TRealOrUInt8 = TypeVar('TRealOrUInt8', bound=RealType | UINT8)
TFloatHighPrecision = TypeVar('TFloatHighPrecision', bound=FLOAT | DOUBLE)
