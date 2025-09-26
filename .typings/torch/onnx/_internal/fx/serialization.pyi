import onnx
import torch
from _typeshed import Incomplete
from torch.types import FileLike as FileLike

log: Incomplete

def _create_tensor_proto_with_external_data(tensor: torch.Tensor, name: str, location: str, basepath: str, dtype_override: onnx.TypeProto | None = None) -> onnx.TensorProto:
    '''Create a TensorProto with external data from a PyTorch tensor.
    The external data is saved to os.path.join(basepath, location).

    Args:
        tensor: Tensor to be saved.
        name: Name of the tensor (i.e., initializer name in ONNX graph).
        location: Relative location of the external data file
            (e.g., "/tmp/initializers/weight_0" when model is "/tmp/model_name.onnx").
        basepath: Base path of the external data file (e.g., "/tmp/external_data" while model must be in "/tmp").


    Reference for ONNX\'s external data format:
        How to load?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L187
        How to save?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L43
        How to set ONNX fields?
        https://github.com/onnx/onnx/blob/5dac81ac0707bdf88f56c35c0a5e8855d3534673/onnx/external_data_helper.py#L88
    '''
def _convert_safetensors_to_torch_format(safetensors_file): ...
def save_model_with_external_data(basepath: str, model_location: str, initializer_location: str, torch_state_dicts: tuple[dict | FileLike, ...], onnx_model: onnx.ModelProto, rename_initializer: bool = False) -> None:
    '''Load PyTorch tensors from files and add to "onnx_model" as external initializers.

    Output files:
        ONNX model file path:
        ONNX initializer folder: os.path.join(basepath, initializer_location)

    After running this function, you can do
        ort_sess = onnxruntime.InferenceSession(os.path.join(basepath, model_location))
    to execute the model.

    Arguments:
        basepath: Base path of the ONNX external data file (e.g., "/path/to/large_model/").
        model_location: Relative location of the ONNX model file.
            E.g., "model.onnx" so that the model file is saved to
            "<basepath>/model.onnx".
        initializer_location: Relative location of the ONNX initializer folder.
            E.g., "initializers" so that the initializers are saved to
            "<basepath>/initializers/".
            Note: When initializers are >2GB, must be the same as `model_location`.
        torch_state_dicts: Dictionaries or files which contain PyTorch tensors to be saved
            as ONNX initializers. For non-dict arguments, `torch.load` will be used to load them from file-like objects.
        onnx_model: ONNX model to be saved with external initializers.
            If an input name matches a tensor loaded from "torch_state_dicts",
            the tensor will be saved as that input\'s external initializer.
        rename_initializer: Replaces "." by "_" for all ONNX initializer names.
            Not needed by the official torch.onnx.dynamo_export. This is a hack
            for supporting `FXSymbolicTracer` tracer with fake tensor mode.
            In short, `FXSymbolicTracer` lifts FX parameters (self.linear_weight)
            as inputs (`def forward(self, linear_weight)`) and therefore, `.` cannot be used.
    '''
