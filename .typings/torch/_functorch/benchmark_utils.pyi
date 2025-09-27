from _typeshed import Incomplete
from torch.profiler import ProfilerActivity as ProfilerActivity, profile as profile

def synchronize() -> None: ...
def dump_chrome_trace(f, input, trace_filename, optimize_ctx, activities, num_runs: int = 1, devices=None, kwargs_for_f=None, kwargs_for_profiler=None):
    """
    Output the chrome trace of running f(input, **kwargs_for_f) with [optimize_ctx]
    [num_runs] times to [trace_filename].

    [activities] are the activities that the profiler will record, e.g. ProfilerActivity.CUDA.
    Return total runtime without the profiler

    Outputs to trace_filename
    """
def get_chrome_trace_events(filename): ...
def is_gpu_compute_event(event): ...
def get_sorted_gpu_events(events): ...
def get_duration(sorted_gpu_events): ...
def get_sorted_gpu_mm_conv_events(events): ...

gpu_pids: Incomplete

def compute_utilization(filename: str, total_length: float):
    """
    Process the chrome traces outputs by the pytorch profiler to compute GPU Utilization
    and percent of times spent on matmul and convolution

    Args:
        filename(str): Name of chrome traces file produced by pytorch profiler

        total_length(float): total length of the process without profiler in second

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)
    """
def benchmark_utilization(f, input, trace_folder, optimize_ctx=None, trace_file_name: str = 'tmp_chrome_trace', num_runs: int = 1):
    '''
    Benchmark the GPU Utilization and percent of time spent on matmul and convolution operations of
    running f(input, **kwargs_for_f) with [optimize_ctx] [num_runs] times.
    It will produce a chrome trace file in trace_folder/trace_file_name.json

    Example:

    ```
    def f(a):
        return a.sum()
    a = torch.rand(2**20, device="cuda")
    utilization, mm_conv_utilization = benchmark_utilization(f, a, "tmp", trace_file_name = "tmp_chrome_trace")
    ```

    Args:
        f: function to benchmark

        input: input to :attr:`f`

        trace_folder: name of the folder to store the chrome trace

        optimize_ctx: the context in which f will run

        trace_file_name: name of the dumped chrome trace file, default to "tmp_chrome_trace"

        num_runs: number of times to run f, excluding the warm-up runs, default to 1.

    Return:
        tuple: (GPU Utilization, percent of time spent on matmul and convolution)

    '''
