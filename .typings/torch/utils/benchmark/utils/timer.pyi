import enum
from _typeshed import Incomplete
from torch.utils.benchmark.utils import common
from torch.utils.benchmark.utils._stubs import TimeitModuleType, TimerClass
from torch.utils.benchmark.utils.valgrind_wrapper import timer_interface as valgrind_timer_interface
from typing import Any, Callable, NoReturn, overload

__all__ = ['Timer', 'timer', 'Language']

def timer() -> float: ...

timer: Incomplete

class Language(enum.Enum):
    PYTHON = 0
    CPP = 1

class CPPTimer:
    _stmt: str
    _setup: str
    _global_setup: str
    _timeit_module: TimeitModuleType | None
    def __init__(self, stmt: str, setup: str, global_setup: str, timer: Callable[[], float], globals: dict[str, Any]) -> None: ...
    def timeit(self, number: int) -> float: ...

class Timer:
    '''Helper class for measuring execution time of PyTorch statements.

    For a full tutorial on how to use this class, see:
    https://pytorch.org/tutorials/recipes/recipes/benchmark.html

    The PyTorch Timer is based on `timeit.Timer` (and in fact uses
    `timeit.Timer` internally), but with several key differences:

    1) Runtime aware:
        Timer will perform warmups (important as some elements of PyTorch are
        lazily initialized), set threadpool size so that comparisons are
        apples-to-apples, and synchronize asynchronous CUDA functions when
        necessary.

    2) Focus on replicates:
        When measuring code, and particularly complex kernels / models,
        run-to-run variation is a significant confounding factor. It is
        expected that all measurements should include replicates to quantify
        noise and allow median computation, which is more robust than mean.
        To that effect, this class deviates from the `timeit` API by
        conceptually merging `timeit.Timer.repeat` and `timeit.Timer.autorange`.
        (Exact algorithms are discussed in method docstrings.) The `timeit`
        method is replicated for cases where an adaptive strategy is not
        desired.

    3) Optional metadata:
        When defining a Timer, one can optionally specify `label`, `sub_label`,
        `description`, and `env`. (Defined later) These fields are included in
        the representation of result object and by the `Compare` class to group
        and display results for comparison.

    4) Instruction counts
        In addition to wall times, Timer can run a statement under Callgrind
        and report instructions executed.

    Directly analogous to `timeit.Timer` constructor arguments:

        `stmt`, `setup`, `timer`, `globals`

    PyTorch Timer specific constructor arguments:

        `label`, `sub_label`, `description`, `env`, `num_threads`

    Args:
        stmt: Code snippet to be run in a loop and timed.

        setup: Optional setup code. Used to define variables used in `stmt`

        global_setup: (C++ only)
            Code which is placed at the top level of the file for things like
            `#include` statements.

        timer:
            Callable which returns the current time. If PyTorch was built
            without CUDA or there is no GPU present, this defaults to
            `timeit.default_timer`; otherwise it will synchronize CUDA before
            measuring the time.

        globals:
            A dict which defines the global variables when `stmt` is being
            executed. This is the other method for providing variables which
            `stmt` needs.

        label:
            String which summarizes `stmt`. For instance, if `stmt` is
            "torch.nn.functional.relu(torch.add(x, 1, out=out))"
            one might set label to "ReLU(x + 1)" to improve readability.

        sub_label:
            Provide supplemental information to disambiguate measurements
            with identical stmt or label. For instance, in our example
            above sub_label might be "float" or "int", so that it is easy
            to differentiate:
            "ReLU(x + 1): (float)"

            "ReLU(x + 1): (int)"
            when printing Measurements or summarizing using `Compare`.

        description:
            String to distinguish measurements with identical label and
            sub_label. The principal use of `description` is to signal to
            `Compare` the columns of data. For instance one might set it
            based on the input size  to create a table of the form: ::

                                        | n=1 | n=4 | ...
                                        ------------- ...
                ReLU(x + 1): (float)    | ... | ... | ...
                ReLU(x + 1): (int)      | ... | ... | ...


            using `Compare`. It is also included when printing a Measurement.

        env:
            This tag indicates that otherwise identical tasks were run in
            different environments, and are therefore not equivalent, for
            instance when A/B testing a change to a kernel. `Compare` will
            treat Measurements with different `env` specification as distinct
            when merging replicate runs.

        num_threads:
            The size of the PyTorch threadpool when executing `stmt`. Single
            threaded performance is important as both a key inference workload
            and a good indicator of intrinsic algorithmic efficiency, so the
            default is set to one. This is in contrast to the default PyTorch
            threadpool size which tries to utilize all cores.
    '''
    _timer_cls: type[TimerClass]
    _globals: Incomplete
    _language: Language
    _timer: Incomplete
    _task_spec: Incomplete
    def __init__(self, stmt: str = 'pass', setup: str = 'pass', global_setup: str = '', timer: Callable[[], float] = ..., globals: dict[str, Any] | None = None, label: str | None = None, sub_label: str | None = None, description: str | None = None, env: str | None = None, num_threads: int = 1, language: Language | str = ...) -> None: ...
    def _timeit(self, number: int) -> float: ...
    def timeit(self, number: int = 1000000) -> common.Measurement:
        """Mirrors the semantics of timeit.Timer.timeit().

        Execute the main statement (`stmt`) `number` times.
        https://docs.python.org/3/library/timeit.html#timeit.Timer.timeit
        """
    def repeat(self, repeat: int = -1, number: int = -1) -> None: ...
    def autorange(self, callback: Callable[[int, float], NoReturn] | None = None) -> None: ...
    def _threaded_measurement_loop(self, number: int, time_hook: Callable[[], float], stop_hook: Callable[[list[float]], bool], min_run_time: float, max_run_time: float | None = None, callback: Callable[[int, float], NoReturn] | None = None) -> list[float]: ...
    def _estimate_block_size(self, min_run_time: float) -> int: ...
    def blocked_autorange(self, callback: Callable[[int, float], NoReturn] | None = None, min_run_time: float = 0.2) -> common.Measurement:
        """Measure many replicates while keeping timer overhead to a minimum.

        At a high level, blocked_autorange executes the following pseudo-code::

            `setup`

            total_time = 0
            while total_time < min_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                total_time += (timer() - start)

        Note the variable `block_size` in the inner loop. The choice of block
        size is important to measurement quality, and must balance two
        competing objectives:

            1) A small block size results in more replicates and generally
               better statistics.

            2) A large block size better amortizes the cost of `timer`
               invocation, and results in a less biased measurement. This is
               important because CUDA synchronization time is non-trivial
               (order single to low double digit microseconds) and would
               otherwise bias the measurement.

        blocked_autorange sets block_size by running a warmup period,
        increasing block size until timer overhead is less than 0.1% of
        the overall computation. This value is then used for the main
        measurement loop.

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
    def adaptive_autorange(self, threshold: float = 0.1, *, min_run_time: float = 0.01, max_run_time: float = 10.0, callback: Callable[[int, float], NoReturn] | None = None) -> common.Measurement:
        """Similar to `blocked_autorange` but also checks for variablility in measurements
        and repeats until iqr/median is smaller than `threshold` or `max_run_time` is reached.


        At a high level, adaptive_autorange executes the following pseudo-code::

            `setup`

            times = []
            while times.sum < max_run_time
                start = timer()
                for _ in range(block_size):
                    `stmt`
                times.append(timer() - start)

                enough_data = len(times)>3 and times.sum > min_run_time
                small_iqr=times.iqr/times.mean<threshold

                if enough_data and small_iqr:
                    break

        Args:
            threshold: value of iqr/median threshold for stopping

            min_run_time: total runtime needed before checking `threshold`

            max_run_time: total runtime  for all measurements regardless of `threshold`

        Returns:
            A `Measurement` object that contains measured runtimes and
            repetition counts, and can be used to compute statistics.
            (mean, median, etc.)
        """
    @overload
    def collect_callgrind(self, number: int, *, repeats: None, collect_baseline: bool, retain_out_file: bool) -> valgrind_timer_interface.CallgrindStats: ...
    @overload
    def collect_callgrind(self, number: int, *, repeats: int, collect_baseline: bool, retain_out_file: bool) -> tuple[valgrind_timer_interface.CallgrindStats, ...]: ...
