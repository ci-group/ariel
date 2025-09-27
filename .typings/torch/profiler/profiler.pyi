import abc
import json
import torch.autograd.profiler as prof
import types
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import Enum
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
from typing import Any, Callable
from typing_extensions import Self

__all__ = ['supported_activities', 'ProfilerAction', 'schedule', 'tensorboard_trace_handler', 'profile', 'ExecutionTraceObserver']

class _NumpyEncoder(json.JSONEncoder):
    """
    Json encoder for numpy types (np.int, np.float, np.array etc.)
    Returns default encoder if numpy is not available
    """
    def default(self, obj):
        """Encode NumPy types to JSON"""

def supported_activities():
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """

class _ITraceObserver(ABC, metaclass=abc.ABCMeta):
    """Abstract interface for a Trace observer.
    This satisfies 3 methods: start, stop and cleanup"""
    @abstractmethod
    def start(self): ...
    @abstractmethod
    def stop(self): ...
    @abstractmethod
    def cleanup(self): ...

class _KinetoProfile:
    """Low-level profiler wrap the autograd profile

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation (see ``export_memory_timeline``
            for more details).
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles


    .. note::
        This API is experimental and subject to change in the future.

        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.
    """
    activities: Incomplete
    record_shapes: Incomplete
    with_flops: Incomplete
    profile_memory: Incomplete
    with_stack: Incomplete
    with_modules: Incomplete
    experimental_config: Incomplete
    execution_trace_observer: Incomplete
    acc_events: Incomplete
    custom_trace_id_callback: Incomplete
    profiler: prof.profile | None
    has_cudagraphs: bool
    mem_tl: MemoryProfileTimeline | None
    use_device: Incomplete
    preset_metadata: dict[str, str]
    def __init__(self, *, activities: Iterable[ProfilerActivity] | None = None, record_shapes: bool = False, profile_memory: bool = False, with_stack: bool = False, with_flops: bool = False, with_modules: bool = False, experimental_config: _ExperimentalConfig | None = None, execution_trace_observer: _ITraceObserver | None = None, acc_events: bool = False, custom_trace_id_callback: Callable[[], str] | None = None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def prepare_trace(self) -> None: ...
    def start_trace(self) -> None: ...
    def stop_trace(self) -> None: ...
    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format. If kineto is enabled, only
        last cycle in schedule is exported.
        """
    def export_stacks(self, path: str, metric: str = 'self_cpu_time_total'):
        '''Save stack traces to a file

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"
        '''
    def toggle_collection_dynamic(self, enable: bool, activities: Iterable[ProfilerActivity]):
        '''Toggle collection of activities on/off at any point of collection. Currently supports toggling Torch Ops
        (CPU) and CUDA activity supported in Kineto

        Args:
            activities (iterable): list of activity groups to use in profiling, supported values:
                ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``
        Examples:

        .. code-block:: python

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as p:
                code_to_profile_0()
                // turn off collection of all CUDA activity
                p.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_1()
                // turn on collection of all CUDA activity
                p.toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CUDA])
                code_to_profile_2()
            print(p.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
        '''
    def key_averages(self, group_by_input_shape: bool = False, group_by_stack_n: int = 0, group_by_overload_name: bool = False):
        """Averages events, grouping them by operator name and (optionally) input shapes, stack
        and overload name.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
    def preset_metadata_json(self, key: str, value: str):
        """
        Preset a user defined metadata when the profiler is not started
        and added into the trace file later.
        Metadata is in the format of a string key and a valid json value
        """
    def _get_distributed_info(self): ...
    def _memory_profile(self) -> MemoryProfile: ...
    def export_memory_timeline(self, path: str, device: str | None = None) -> None:
        """Export memory event information from the profiler collected
        tree for a given device, and export a timeline plot. There are 3
        exportable files using ``export_memory_timeline``, each controlled by the
        ``path``'s suffix.

        - For an HTML compatible plot, use the suffix ``.html``, and a memory timeline
          plot will be embedded as a PNG file in the HTML file.

        - For plot points consisting of ``[times, [sizes by category]]``, where
          ``times`` are timestamps and ``sizes`` are memory usage for each category.
          The memory timeline plot will be saved a JSON (``.json``) or gzipped JSON
          (``.json.gz``) depending on the suffix.

        - For raw memory points, use the suffix ``.raw.json.gz``. Each raw memory
          event will consist of ``(timestamp, action, numbytes, category)``, where
          ``action`` is one of ``[PREEXISTING, CREATE, INCREMENT_VERSION, DESTROY]``,
          and ``category`` is one of the enums from
          ``torch.profiler._memory_profiler.Category``.

        Output: Memory timeline written as gzipped JSON, JSON, or HTML.
        """

class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3

def schedule(*, wait: int, warmup: int, active: int, repeat: int = 0, skip_first: int = 0, skip_first_wait: int = 0) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.

    The ``skip_first_wait`` parameter controls whether the first ``wait`` stage should be skipped.
    This can be useful if a user wants to wait longer than ``skip_first`` between cycles, but not
    for the first profile. For example, if ``skip_first`` is 10 and ``wait`` is 20, the first cycle will
    wait 10 + 20 = 30 steps before warmup if ``skip_first_wait`` is zero, but will wait only 10
    steps if ``skip_first_wait`` is non-zero. All subsequent cycles will then wait 20 steps between the
    last active and warmup.
    """
def tensorboard_trace_handler(dir_name: str, worker_name: str | None = None, use_gzip: bool = False):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """

class profile(_KinetoProfile):
    '''Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``,
            ``torch.profiler.ProfilerActivity.XPU``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA
            or (when available) ProfilerActivity.XPU.
        schedule (Callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (Callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        record_shapes (bool): save information about operator\'s input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
            (matrix multiplication and 2D convolution).
        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A\'s forward call\'s
            module B\'s forward which contains an aten::add op,
            then aten::add\'s module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.
        experimental_config (_ExperimentalConfig) : A set of experimental options
            used for Kineto library features. Note, backward compatibility is not guaranteed.
        execution_trace_observer (ExecutionTraceObserver) : A PyTorch Execution Trace Observer object.
            `PyTorch Execution Traces <https://arxiv.org/pdf/2305.14516.pdf>`__ offer a graph based
            representation of AI/ML workloads and enable replay benchmarks, simulators, and emulators.
            When this argument is included the observer start() and stop() will be called for the
            same time window as PyTorch profiler. See the examples section below for a code sample.
        acc_events (bool): Enable the accumulation of FunctionEvents across multiple profiling cycles
        use_cuda (bool):
            .. deprecated:: 1.8.1
                use ``activities`` instead.

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.
        When record_shapes=True is specified, profiler will temporarily hold references to the tensors;
        that may further prevent certain optimizations that depend on the reference count and introduce
        extra tensor copies.


    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler\'s ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2, repeat=1,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2,
                repeat=1),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(\'./log\')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()

    The following sample shows how to setup up an Execution Trace Observer (`execution_trace_observer`)

    .. code-block:: python

        with torch.profiler.profile(
            ...
            execution_trace_observer=(
                ExecutionTraceObserver().register_callback("./execution_trace.json")
            ),
        ) as p:
            for iter in range(N):
                code_iteration_to_profile(iter)
                p.step()

    You can also refer to test_execution_trace_with_kineto() in tests/profiler/test_profiler.py.
    Note: One can also pass any object satisfying the _ITraceObserver interface.
    '''
    schedule: Incomplete
    record_steps: bool
    on_trace_ready: Incomplete
    step_num: int
    current_action: Incomplete
    step_rec_fn: prof.record_function | None
    action_map: dict[tuple[ProfilerAction, ProfilerAction | None], list[Any]]
    def __init__(self, *, activities: Iterable[ProfilerActivity] | None = None, schedule: Callable[[int], ProfilerAction] | None = None, on_trace_ready: Callable[..., Any] | None = None, record_shapes: bool = False, profile_memory: bool = False, with_stack: bool = False, with_flops: bool = False, with_modules: bool = False, experimental_config: _ExperimentalConfig | None = None, execution_trace_observer: _ITraceObserver | None = None, acc_events: bool = False, use_cuda: bool | None = None, custom_trace_id_callback: Callable[[], str] | None = None) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: types.TracebackType | None) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def step(self) -> None:
        """
        Signals the profiler that the next profiling step has started.
        """
    custom_trace_id_callback: Incomplete
    def set_custom_trace_id_callback(self, callback) -> None:
        """
        Sets a callback to be called when a new trace ID is generated.
        """
    def get_trace_id(self):
        """
        Returns the current trace ID.
        """
    def _trace_ready(self) -> None: ...
    def _transit_action(self, prev_action, current_action) -> None: ...
    def _stats(self) -> prof._ProfilerStats | None: ...

class ExecutionTraceObserver(_ITraceObserver):
    """Execution Trace Observer

    Each process can have a single ExecutionTraceObserver instance. The observer
    can be added to record function callbacks via calling register_callback()
    explicitly. Without calling unregister_callback(), repeated calls to
    register_callback() will not add additional observers to record function
    callbacks. Once an ExecutionTraceObserver is created, the start() and stop()
    methods control when the event data is recorded.

    Deleting or calling unregister_callback() will remove the observer from the
    record function callbacks, finalize the output file, and will stop
    incurring any overheads.
    """
    _registered: bool
    _execution_trace_running: bool
    extra_resources_collection: bool
    resources_dir: str
    output_file_path: str
    output_file_path_observer: str
    def __init__(self) -> None:
        """
        Initializes the default states.
        """
    def __del__(self) -> None:
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
    @staticmethod
    def build_execution_trace_obs_from_env() -> ExecutionTraceObserver | None:
        """
        Returns an ExecutionTraceObserver instance if the environment variable
        ENABLE_PYTORCH_EXECUTION_TRACE is set to 1, otherwise returns None.

        Configures the observer to also collect extra resources if the environment variable
        ``ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS=1``. These are resources such as generated kernels,
        index tensor data etc. that are required to make the Execution Trace replayable.
        """
    def set_extra_resource_collection(self, val) -> None:
        """
        Collects extra resources such as generated kernels, index tensor data, and any other
        metadata that is required to complete the Execution Trace content.

        The caller should call this method with val=True after calling register_callback() if they want
        to collect the extra resources.
        """
    def register_callback(self, output_file_path: str) -> Self:
        """
        Adds ET observer to record function callbacks. The data will be
        written to output_file_path.
        """
    def get_resources_dir(self, can_create: bool = False) -> str | None:
        """
        Generates the resources directory for the generated kernels,
        or index tensor data or any other metadata that is required
        to complete the Execution Trace content.

        The directory is created right where the ET file is being output.

        Only works if the observer has called set_extra_resource_collection(val=True).

        Returns None if the observer is not configured with extra resource collection.
        """
    @staticmethod
    def get_resources_dir_for_et_path(trace_path, create_dir: bool = False) -> str | None: ...
    def unregister_callback(self) -> None:
        """
        Removes ET observer from record function callbacks.
        """
    @property
    def is_registered(self):
        """
        Returns True if the execution trace observer is registered, otherwise False.
        """
    def is_running(self):
        """
        Returns True if the observer is running, otherwise False.
        """
    def start(self) -> None:
        """
        Starts to capture.
        """
    def stop(self) -> None:
        """
        Stops to capture.
        """
    def cleanup(self) -> None:
        """
        Calls unregister_callback() to make sure to finalize outputs.
        """
    def get_output_file_path(self) -> str | None:
        """
        Returns the output file name or None.
        """
    def _record_pg_config(self) -> None: ...
