from _typeshed import Incomplete

def format_time(time_us=None, time_ms=None, time_s=None):
    """Define time formatting."""

class ExecutionStats:
    _c_stats: Incomplete
    benchmark_config: Incomplete
    def __init__(self, c_stats, benchmark_config) -> None: ...
    @property
    def latency_avg_ms(self): ...
    @property
    def num_iters(self): ...
    @property
    def iters_per_second(self):
        """Return total number of iterations per second across all calling threads."""
    @property
    def total_time_seconds(self): ...
    def __str__(self) -> str: ...

class ThroughputBenchmark:
    '''
    This class is a wrapper around a c++ component throughput_benchmark::ThroughputBenchmark.

    This wrapper on the throughput_benchmark::ThroughputBenchmark component is responsible
    for executing a PyTorch module (nn.Module or ScriptModule) under an inference
    server like load. It can emulate multiple calling threads to a single module
    provided. In the future we plan to enhance this component to support inter and
    intra-op parallelism as well as multiple models running in a single process.

    Please note that even though nn.Module is supported, it might incur an overhead
    from the need to hold GIL every time we execute Python code or pass around
    inputs as Python objects. As soon as you have a ScriptModule version of your
    model for inference deployment it is better to switch to using it in this
    benchmark.

    Example::

        >>> # xdoctest: +SKIP("undefined vars")
        >>> from torch.utils import ThroughputBenchmark
        >>> bench = ThroughputBenchmark(my_module)
        >>> # Pre-populate benchmark\'s data set with the inputs
        >>> for input in inputs:
        ...     # Both args and kwargs work, same as any PyTorch Module / ScriptModule
        ...     bench.add_input(input[0], x2=input[1])
        >>> # Inputs supplied above are randomly used during the execution
        >>> stats = bench.benchmark(
        ...     num_calling_threads=4,
        ...     num_warmup_iters = 100,
        ...     num_iters = 1000,
        ... )
        >>> print("Avg latency (ms): {}".format(stats.latency_avg_ms))
        >>> print("Number of iterations: {}".format(stats.num_iters))
    '''
    _benchmark: Incomplete
    def __init__(self, module) -> None: ...
    def run_once(self, *args, **kwargs):
        """
        Given input id (input_idx) run benchmark once and return prediction.

        This is useful for testing that benchmark actually runs the module you
        want it to run. input_idx here is an index into inputs array populated
        by calling add_input() method.
        """
    def add_input(self, *args, **kwargs) -> None:
        """
        Store a single input to a module into the benchmark memory and keep it there.

        During the benchmark execution every thread is going to pick up a
        random input from the all the inputs ever supplied to the benchmark via
        this function.
        """
    def benchmark(self, num_calling_threads: int = 1, num_warmup_iters: int = 10, num_iters: int = 100, profiler_output_path: str = ''):
        """
        Run a benchmark on the module.

        Args:
            num_warmup_iters (int): Warmup iters are used to make sure we run a module
                a few times before actually measuring things. This way we avoid cold
                caches and any other similar problems. This is the number of warmup
                iterations for each of the thread in separate

            num_iters (int): Number of iterations the benchmark should run with.
                This number is separate from the warmup iterations. Also the number is
                shared across all the threads. Once the num_iters iterations across all
                the threads is reached, we will stop execution. Though total number of
                iterations might be slightly larger. Which is reported as
                stats.num_iters where stats is the result of this function

            profiler_output_path (str): Location to save Autograd Profiler trace.
                If not empty, Autograd Profiler will be enabled for the main benchmark
                execution (but not the warmup phase). The full trace will be saved
                into the file path provided by this argument


        This function returns BenchmarkExecutionStats object which is defined via pybind11.
        It currently has two fields:
            - num_iters - number of actual iterations the benchmark have made
            - avg_latency_ms - average time it took to infer on one input example in milliseconds
        """
