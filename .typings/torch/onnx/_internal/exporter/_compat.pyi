import os
import torch
from _typeshed import Incomplete
from collections.abc import Mapping, Sequence
from torch.onnx._internal._lazy_import import onnxscript_apis as onnxscript_apis
from torch.onnx._internal.exporter import _constants as _constants, _core as _core, _dynamic_shapes as _dynamic_shapes, _onnx_program as _onnx_program, _registration as _registration
from typing import Any, Callable

logger: Incomplete

def _get_torch_export_args(args: tuple[Any, ...], kwargs: dict[str, Any] | None) -> tuple[tuple[Any, ...], dict[str, Any] | None]:
    """Obtain the arguments for torch.onnx.export from the model and the input arguments."""
def export_compat(model: torch.nn.Module | torch.export.ExportedProgram | torch.jit.ScriptModule | torch.jit.ScriptFunction, args: tuple[Any, ...], f: str | os.PathLike | None = None, *, kwargs: dict[str, Any] | None = None, export_params: bool = True, verbose: bool | None = None, input_names: Sequence[str] | None = None, output_names: Sequence[str] | None = None, opset_version: int | None = ..., custom_translation_table: dict[Callable, Callable | Sequence[Callable]] | None = None, dynamic_axes: Mapping[str, Mapping[int, str]] | Mapping[str, Sequence[int]] | None = None, dynamic_shapes: dict[str, Any] | tuple[Any, ...] | list[Any] | None = None, keep_initializers_as_inputs: bool = False, external_data: bool = True, report: bool = False, optimize: bool = False, verify: bool = False, profile: bool = False, dump_exported_program: bool = False, artifacts_dir: str | os.PathLike = '.', fallback: bool = False, legacy_export_kwargs: dict[str, Any] | None = None) -> _onnx_program.ONNXProgram: ...
