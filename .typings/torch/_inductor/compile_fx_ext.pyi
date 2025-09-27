import abc
import contextlib
import logging
import queue
import torch
import torch.fx
import types
import warnings
from . import config as config
from .compile_fx import FxCompile as FxCompile, _CompileFxKwargs as _CompileFxKwargs, _InProcessFxCompile as _InProcessFxCompile, log as log
from .debug import DebugContext as DebugContext
from .graph import GraphLowering as GraphLowering
from .virtualized import V as V
from _typeshed import Incomplete
from abc import abstractmethod
from collections.abc import Generator, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from torch._inductor.codecache import BypassFxGraphCache as BypassFxGraphCache, FxGraphCache as FxGraphCache
from torch._inductor.metrics import CachedMetricsDeltas as CachedMetricsDeltas, CachedMetricsHelper as CachedMetricsHelper
from torch._inductor.output_code import CompiledFxGraph as CompiledFxGraph, CompiledFxGraphConstants as CompiledFxGraphConstants, CompiledFxGraphConstantsWithGm as CompiledFxGraphConstantsWithGm, OutputCode as OutputCode
from torch._inductor.utils import InputType as InputType
from torch._subclasses import FakeTensorMode as FakeTensorMode
from torch.fx import GraphModule as GraphModule
from torch.utils._ordered_set import OrderedSet as OrderedSet
from typing import Any
from typing_extensions import Self, final, override

@dataclass
class _VirtualizedSerializer:
    """
    This handles the data for serializing Virtualized.
    """
    aot_compilation: Any = ...
    choices: Any = ...
    local_buffer_context: Any = ...
    ops: Any = ...
    kernel: Any = ...
    current_node: Any = ...
    @classmethod
    def serialize(cls) -> _VirtualizedSerializer:
        """
        Turn the current state of torch._inductor.virtualized.V into a
        serializable structure.
        """
    def patch(self) -> _VirtualizedSerializerContextManager:
        """
        Returns a context manager which patches the saved values into the
        current environment. While patched, any value not listed above will be
        poisoned so that reads will raise an error.
        """

class _VirtualizedSerializerContextManager(contextlib.ExitStack):
    """
    Helper for _VirtualizedSerializer.patch()
    """
    virtualized: Incomplete
    def __init__(self, virtualized: _VirtualizedSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

def _is_fallback_handler(op: object) -> bool: ...

class _LoweringSerializer:
    """
    This handles the data for serializing lowering.lowering
    """
    fallbacks: OrderedSet[str]
    def __init__(self) -> None: ...
    def patch(self) -> _LoweringSerializerContextManager: ...

class _LoweringSerializerContextManager(contextlib.ExitStack):
    """
    Helper for _LoweringSerializer.patch()
    """
    lowering: Incomplete
    def __init__(self, lowering: _LoweringSerializer) -> None: ...
    @override
    def __enter__(self) -> Self: ...

@dataclass
class _FakeTensorModeSerializer:
    allow_non_fake_inputs: bool
    shape_env = ...
    def __init__(self, fake_mode: FakeTensorMode) -> None: ...
    @contextlib.contextmanager
    def patch(self, fake_mode: FakeTensorMode) -> Generator[None, None, None]: ...

@dataclass
class _WireProtocolInput:
    """
    For _SerializedFxCompile - encapsulates all the data being transferred
    (sent) from the parent to the child.
    """
    gm: torch.fx.GraphModule
    example_inputs: Sequence[InputType]
    inputs_to_check: Sequence[int]
    graph_kwargs: _CompileFxKwargs
    tracing_context: torch._guards.TracingContext | None
    config: dict[str, object]
    virtualized: _VirtualizedSerializer
    deterministic_guard_for_testing: torch.testing._internal.common_utils.DeterministicGuard | None
    logger_state: _LoggerState
    lowering: _LoweringSerializer
    fake_tensor_mode: _FakeTensorModeSerializer
    def serialize(self) -> _WireProtocolPickledInput:
        """
        Turns this object into a _WireProtocolPickledInput which can be
        directly transferred across a stream.
        """

def _current_fake_mode() -> FakeTensorMode: ...

@dataclass
class _WireProtocolPickledInput:
    value: bytes
    def deserialize(self) -> _WireProtocolInput:
        """
        Turn this streamable object back into a _WireProtocolInput.
        """

@dataclass
class _WireProtocolOutput:
    """
    For _SerializedFxCompile - encapsulates all the data being transferred
    (returned) back from the child to the parent.
    """
    graph: OutputCode
    metrics: CachedMetricsDeltas
    logs: list[logging.LogRecord]
    warning_replay: list[warnings.WarningMessage] | None
    shape_env: torch.fx.experimental.symbolic_shapes.ShapeEnv | None
    def serialize(self) -> _WireProtocolPickledOutput:
        """
        Turns this object into a _WireProtocolPickledOutput which can be
        directly transferred across a stream.
        """

@dataclass
class _WireProtocolPickledOutput:
    value: bytes
    def deserialize(self, constants: CompiledFxGraphConstants) -> _WireProtocolOutput:
        """
        Turn this streamable object back into a _WireProtocolOutput.
        """

class _LoggerState:
    '''
    This class is for tracking logging that happens during an out-of-process
    compile so we can "replay" those messages when the compile is done. Used as
    a context manager which returns the captured logs (object).
    '''
    loggers: dict[str, int]
    captured_logs: _CapturedLogs | None
    def __init__(self) -> None: ...
    def __enter__(self) -> _CapturedLogs: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class _CapturedLogs:
    """
    Helper for _LoggerState - this class actually attaches to the logger in
    the child process and grabs the log messages themselves.
    """
    state: _LoggerState
    queue: queue.Queue[logging.LogRecord]
    handlers: dict[str, logging.Handler] | None
    def __init__(self, state: _LoggerState) -> None: ...
    def finish(self) -> list[logging.LogRecord]: ...
    def remove(self) -> None: ...
    def apply(self) -> None: ...

class _SerializedFxCompile(FxCompile, metaclass=abc.ABCMeta):
    """
    This is used to represent an FxCompile which occurs across a serialized
    boundary.
    """
    @override
    def codegen_and_compile(self, gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], graph_kwargs: _CompileFxKwargs) -> OutputCode: ...
    def serialize_compile(self, gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], graph_kwargs: _CompileFxKwargs) -> tuple[_WireProtocolPickledInput, CompiledFxGraphConstantsWithGm] | None:
        """
        Prepare a _WireProtocolInput to compile. If None is returned then it
        wasn't possible to serialize and we should fallback to in-process.
        """
    @abstractmethod
    def _send_to_child(self, pickled_input: _WireProtocolPickledInput) -> _WireProtocolPickledOutput: ...
    def _postprocess(self, output: _WireProtocolOutput) -> None: ...
    @classmethod
    def _run_in_child(cls, pickled_input: _WireProtocolPickledInput, extra_env: Mapping[str, str] | None = None) -> _WireProtocolPickledOutput: ...

class _DebugSerdeFxCompile(_SerializedFxCompile):
    @override
    def _send_to_child(self, pickled_input: _WireProtocolPickledInput) -> _WireProtocolPickledOutput: ...

class _OutOfProcessFxCompile(_SerializedFxCompile, metaclass=abc.ABCMeta):
    """
    Represents an FxCompile which is run outside the current process (in
    either a subprocess or possibly even a separate machine).
    """
    @override
    @final
    def _send_to_child(self, pickled_input: _WireProtocolPickledInput) -> _WireProtocolPickledOutput: ...
    @abstractmethod
    def _send_to_child_async(self, pickled_input: _WireProtocolPickledInput) -> Future[_WireProtocolPickledOutput]: ...
    def _postprocess(self, output: _WireProtocolOutput) -> None: ...

class _DebugFileFxCompile(_SerializedFxCompile):
    file_index: int
    @override
    def _send_to_child(self, pickled_input: _WireProtocolPickledInput) -> _WireProtocolPickledOutput: ...
