import abc
import contextlib
import dataclasses
import os
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from torch.export import _draft_export as _draft_export
from typing import Any, Callable

logger: Incomplete

def _verbose_printer(verbose: bool | None) -> Callable[..., None]:
    """Prints messages based on `verbose`."""
def _take_first_line(text: str) -> str:
    """Take the first line of a text."""
@contextlib.contextmanager
def _patch_dynamo_unsupported_functions() -> Generator[None]:
    """Patch PyTorch to bypass some functions torch.export.export does not support."""

@dataclasses.dataclass
class Result:
    exported_program: torch.export.ExportedProgram | None
    strategy: str
    exception: Exception | None = ...
    @property
    def success(self) -> bool:
        """Whether the capture was successful.

        An exception can still be recorded even if the capture was successful. In
        this case the exception is informational only. For example, draft_export
        can record an exception if there are warnings during the export. The exceptions
        will go into the onnx export report when report=True.
        """

class CaptureStrategy(abc.ABC, metaclass=abc.ABCMeta):
    """Strategy for capturing a module as ExportedProgram.

    To use a strategy, create an instance and call it with the model, args, kwargs, and dynamic_shapes.
    Example::

        strategy = TorchExportNonStrictStrategy(verbose=True)
        result = strategy(model, args, kwargs, dynamic_shapes)
    """
    _verbose_print: Incomplete
    _dump: Incomplete
    _artifacts_dir: Incomplete
    _timestamp: Incomplete
    _exception: Exception | None
    def __init__(self, *, verbose: bool = False, dump: bool = False, artifacts_dir: str | os.PathLike = '.', timestamp: str | None = None) -> None:
        """Initialize the strategy.

        Args:
            verbose: Whether to print verbose messages.
            dump: Whether to dump the intermediate artifacts to a file.
        """
    def __call__(self, model: torch.nn.Module | torch.jit.ScriptFunction, args: tuple[Any, ...], kwargs: dict[str, Any] | None, dynamic_shapes) -> Result: ...
    @abc.abstractmethod
    def _capture(self, model, args, kwargs, dynamic_shapes) -> torch.export.ExportedProgram: ...
    def _enter(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None: ...
    def _success(self, model: torch.nn.Module | torch.jit.ScriptFunction) -> None: ...
    def _failure(self, model: torch.nn.Module | torch.jit.ScriptFunction, e: Exception) -> None: ...

class TorchExportStrictStrategy(CaptureStrategy):
    def _capture(self, model, args, kwargs, dynamic_shapes) -> torch.export.ExportedProgram: ...
    def _enter(self, model) -> None: ...
    def _success(self, model) -> None: ...
    def _failure(self, model, e) -> None: ...

class TorchExportNonStrictStrategy(CaptureStrategy):
    def _capture(self, model, args, kwargs, dynamic_shapes) -> torch.export.ExportedProgram: ...
    def _enter(self, model) -> None: ...
    def _success(self, model) -> None: ...
    def _failure(self, model, e) -> None: ...

class TorchExportDraftExportStrategy(CaptureStrategy):
    _exception: Incomplete
    def _capture(self, model, args, kwargs, dynamic_shapes) -> torch.export.ExportedProgram: ...
    def _enter(self, model) -> None: ...
    def _success(self, model) -> None: ...
    def _failure(self, model, e) -> None: ...

CAPTURE_STRATEGIES: Incomplete
