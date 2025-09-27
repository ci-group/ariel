import torch
from _typeshed import Incomplete
from functools import cached_property as cached_property
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed
from torch._inductor.config import use_experimental_benchmarker as use_experimental_benchmarker
from typing import Any, Callable
from typing_extensions import Concatenate, ParamSpec, TypeVar

logger: Incomplete
MILLISECONDS_PER_SECOND: int
P = ParamSpec('P')
T = TypeVar('T')

def time_and_count(fn: Callable[Concatenate[Any, P], T]) -> Callable[Concatenate[Any, P], T]:
    """Wraps `fn` with `dynamo_timed` context, and increments the appropriate dynamo
    counters. It is expected that `fn` is a method of `Benchmarker` or one of its
    subclasses; typing limitations prevent us from declaring this directly.
    """

class Benchmarker:
    def __init__(self) -> None: ...
    @time_and_count
    def benchmark(self, fn: Callable[..., Any], fn_args: tuple[Any, ...], fn_kwargs: dict[str, Any], **kwargs: Any) -> float:
        """Benchmark `fn(*fn_args, *fn_kwargs)` and return the runtime, in milliseconds (the
        actual runtime calculation is dictated by the benchmarking implementation, but may be
        one of [mean, median, minimum, etc.]). Functions as a convenience wrapper around
        device-specific implementations, like `benchmark_cpu` and `benchmark_gpu`. Raises
        `ValueError(...)` if we can't safely infer the device type of `fn`; for example,
        if multiple device types are found in `fn_args` and `fn_kwargs`, or if no device
        types are found.

        Arguments:
        - fn: The function to benchmark.
        - fn_args: The function's arguments.
        - fn_kwargs: The function's kwargs.

        Keyword Arguments:
        - **kwargs: The benchmarking implementation's kwargs.

        Returns:
        - The runtime of `fn(*fn_args, **fn_kwargs)`, in milliseconds.
        """
    @time_and_count
    def benchmark_cpu(self, _callable: Callable[[], Any], warmup: int = 20, rep: int = 100) -> float:
        """Benchmark the CPU callable, `_callable`, and return the median runtime,
        in milliseconds.

        Arguments:
        - _callable: The CPU callable to benchmark.

        Keyword Arguments:
        - warmup: Optionally, the duration, in milliseconds, to run `_callable`
        before benchmarking starts.
        - rep: Optionally, the duration, in milliseconds, to run `_callable`
        during benchmarking.

        Returns:
        - The median runtime of `_callable`, in milliseconds.
        """
    @time_and_count
    def benchmark_gpu(self, *args: Any, **kwargs: Any) -> float: ...

class TritonBenchmarker(Benchmarker):
    @cached_property
    def triton_do_bench(self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
    @time_and_count
    def benchmark_gpu(self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        '''Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton\'s
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton\'s `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        '''

class InductorBenchmarker(TritonBenchmarker):
    @cached_property
    def L2_cache_size(self) -> int:
        """Get the L2 cache size, in bytes, of the current device."""
    def get_event_pairs(self, iters: int) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        """Get `iters` pairs of CUDA events."""
    def get_event_pairs_min_timing(self, event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        """Get the minimum timing, in milliseconds, for a group of CUDA event pairs."""
    @time_and_count
    def benchmark_gpu(self, _callable: Callable[[], Any], estimation_iters: int = 5, memory_warmup_iters: int = 100, benchmark_iters: int = 100, max_benchmark_duration: int = 25, **kwargs: Any) -> float:
        """Benchmark a GPU callable using a custom benchmarking implementation.

        Arguments:
        - _callable: The callable to benchmark.

        Keyword Arguments:
        - estimation_iters: Optionally, the number of iterations to run `_callable`
        during runtime estimation.
        - memory_warmup_iters: Optionally, the number of iterations to flush the L2
        cache before starting benchmarking.
        - benchmark_iters: Optionally, the number of iterations to run `_callable`
        during the benchmarking.
        - max_benchmark_duration: Optionally, the maximum duration of the benchmarking,
        in milliseconds. An estimated duration is calculated based on the values
        of `memory_warmup_iters` and `benchmark_iters`, along with the estimated
        runtime of `_callable` and various other factors, and we then shrink
        `benchmark_iters` to fit in the allotted maximum duration.
        - **kwargs: Additional kwargs that may be passed to the fallback.

        Returns:
        - The minimum runtime of `_callable`, in milliseconds.
        """

benchmarker: Incomplete
