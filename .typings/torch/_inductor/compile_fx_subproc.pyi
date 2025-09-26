import functools
from .compile_fx_ext import _OutOfProcessFxCompile as _OutOfProcessFxCompile, _WireProtocolPickledInput as _WireProtocolPickledInput, _WireProtocolPickledOutput as _WireProtocolPickledOutput
from collections.abc import Mapping
from concurrent.futures import Future
from torch._inductor.compile_worker.subproc_pool import AnyPool as AnyPool, SubprocKind as SubprocKind, SubprocPool as SubprocPool
from torch._inductor.utils import clear_caches as clear_caches
from typing_extensions import override

class _SubprocessFxCompile(_OutOfProcessFxCompile):
    @override
    def _send_to_child_async(self, input: _WireProtocolPickledInput) -> Future[_WireProtocolPickledOutput]: ...
    @staticmethod
    @functools.cache
    def process_pool() -> AnyPool: ...
    @classmethod
    def _run_in_child_subprocess(cls, pickled_input: _WireProtocolPickledInput, extra_env: Mapping[str, str] | None) -> _WireProtocolPickledOutput: ...
