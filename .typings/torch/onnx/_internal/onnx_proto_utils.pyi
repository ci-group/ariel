import io
from collections.abc import Mapping
from torch.onnx import errors as errors
from torch.onnx._internal import jit_utils as jit_utils, registration as registration
from typing import Any

def export_as_test_case(model_bytes: bytes, inputs_data, outputs_data, name: str, dir: str) -> str:
    """Export an ONNX model as a self contained ONNX test case.

    The test case contains the model and the inputs/outputs data. The directory structure
    is as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        model_bytes: The ONNX model in bytes.
        inputs_data: The inputs data, nested data structure of numpy.ndarray.
        outputs_data: The outputs data, nested data structure of numpy.ndarray.

    Returns:
        The path to the test case directory.
    """
def load_test_case(dir: str) -> tuple[bytes, Any, Any]:
    """Load a self contained ONNX test case from a directory.

    The test case must contain the model and the inputs/outputs data. The directory structure
    should be as follows:

    dir
    ├── test_<name>
    │   ├── model.onnx
    │   └── test_data_set_0
    │       ├── input_0.pb
    │       ├── input_1.pb
    │       ├── output_0.pb
    │       └── output_1.pb

    Args:
        dir: The directory containing the test case.

    Returns:
        model_bytes: The ONNX model in bytes.
        inputs: the inputs data, mapping from input name to numpy.ndarray.
        outputs: the outputs data, mapping from output name to numpy.ndarray.
    """
def export_data(data, value_info_proto, f: str) -> None:
    """Export data to ONNX protobuf format.

    Args:
        data: The data to export, nested data structure of numpy.ndarray.
        value_info_proto: The ValueInfoProto of the data. The type of the ValueInfoProto
            determines how the data is stored.
        f: The file to write the data to.
    """
def _export_file(model_bytes: bytes, f: io.BytesIO | str, export_map: Mapping[str, bytes]) -> None:
    """export/write model bytes into directory/protobuf/zip"""
def _add_onnxscript_fn(model_bytes: bytes, custom_opsets: Mapping[str, int]) -> bytes:
    """Insert model-included custom onnx-script function into ModelProto"""
def _find_onnxscript_op(graph_proto, included_node_func: set[str], custom_opsets: Mapping[str, int], onnx_function_list: list):
    """Recursively iterate ModelProto to find ONNXFunction op as it may contain control flow Op."""
