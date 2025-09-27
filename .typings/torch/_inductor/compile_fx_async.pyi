from .compile_fx import FxCompile as FxCompile, _CompileFxKwargs as _CompileFxKwargs, _InProcessFxCompile as _InProcessFxCompile
from .compile_fx_ext import _OutOfProcessFxCompile as _OutOfProcessFxCompile, _WireProtocolPickledOutput as _WireProtocolPickledOutput
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from torch._inductor.output_code import CompiledFxGraphConstants as CompiledFxGraphConstants, OutputCode as OutputCode
from torch._inductor.utils import InputType as InputType
from torch.fx import GraphModule as GraphModule
from typing import Any, Callable
from typing_extensions import override

@dataclass
class _PostCompileData:
    example_inputs: Sequence[InputType]
    constants: CompiledFxGraphConstants
    graph_kwargs: _CompileFxKwargs

class _AsyncOutputCode(OutputCode):
    _eager_forward: Callable[..., Any] | None
    _output_code: OutputCode | None
    _future: Future[_WireProtocolPickledOutput] | None
    _callback: Callable[[_WireProtocolPickledOutput], OutputCode]
    _post_compile_data: _PostCompileData | None
    _boxed_call: bool
    def __init__(self, eager_forward: Callable[..., Any], future: Future[_WireProtocolPickledOutput], callback: Callable[[_WireProtocolPickledOutput], OutputCode]) -> None: ...
    @override
    def __call__(self, *args: Any) -> Any: ...
    def _switch_to_compiled_forward(self, args: tuple[Any, ...]) -> tuple[Any, ...]: ...
    @override
    def post_compile(self, example_inputs: Sequence[InputType], constants: CompiledFxGraphConstants, graph_kwargs: _CompileFxKwargs) -> None: ...

class _AsyncFxCompile(FxCompile):
    _compile: _OutOfProcessFxCompile
    _stat_bg_started: int
    _stat_bg_finished: int
    _stat_eager_runs: int
    _stat_compiled_runs: int
    def __init__(self, compile: _OutOfProcessFxCompile) -> None: ...
    @classmethod
    def _reset_stats(cls) -> None: ...
    @override
    def codegen_and_compile(self, gm: GraphModule, example_inputs: Sequence[InputType], inputs_to_check: Sequence[int], graph_kwargs: _CompileFxKwargs) -> OutputCode: ...
