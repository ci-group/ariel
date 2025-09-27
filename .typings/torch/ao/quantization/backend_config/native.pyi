from .backend_config import BackendConfig
from _typeshed import Incomplete

__all__ = ['get_test_only_legacy_native_backend_config', 'default_op_quint8_dtype_config', 'default_op_fp16_dtype_config', 'default_dynamic_int8_dtype_config', 'default_dynamic_float16_dtype_config', 'input_output_only_quint8_dtype_config', 'weight_only_quint8_dtype_config', 'weight_only_quint4x2_dtype_config', 'get_native_backend_config', 'get_native_backend_config_dict', 'get_test_only_legacy_native_backend_config_dict']

default_op_quint8_dtype_config: Incomplete
default_op_fp16_dtype_config: Incomplete
default_dynamic_int8_dtype_config: Incomplete
default_dynamic_float16_dtype_config: Incomplete
input_output_only_quint8_dtype_config: Incomplete
weight_only_quint8_dtype_config: Incomplete
weight_only_quint4x2_dtype_config: Incomplete

def get_test_only_legacy_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional fp16 ops.
    """
def get_native_backend_config() -> BackendConfig:
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack).
    """
def get_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) in dictionary form.
    """
def get_test_only_legacy_native_backend_config_dict():
    """
    Return the `BackendConfig` for PyTorch Native backend (fbgemm/qnnpack) with various additional
    fp16 ops in dictionary form.
    """
