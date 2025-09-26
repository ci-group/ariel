import torch
from _typeshed import Incomplete
from collections.abc import KeysView, Sequence
from enum import Enum
from torch._inductor.custom_graph_pass import CustomGraphPass as CustomGraphPass
from torch._inductor.scheduler import BaseSchedulerNode as BaseSchedulerNode
from torch.utils._config_module import ConfigModule as ConfigModule, _ConfigEntry as _ConfigEntry
from torch.utils._ordered_set import OrderedSet as OrderedSet
from types import FrameType
from typing import Any, Callable, TypeVar

log: Incomplete

def is_type(type_hint, comp_type) -> bool:
    """
    Determines if type_hint is comp_type. There are some type annotations that this doesn't work for.
    I think it's because some Type annotations are Type Objects and some are Special Forms, but not sure.
    There's definite room for improvement to make this more general for someone who deeply understands
    Python types.
    """
def is_optional_type(type_hint) -> bool:
    """
    Special case of is_type.
    """
def is_callable_type(type_hint) -> bool:
    """
    Special Case of is_type.
    """

class DummyPass(CustomGraphPass):
    """
    A Dummy pass to be used by ConfigFuzzer
    """
    def __call__(self, graph: torch.fx.graph.Graph) -> None: ...
    def uuid(self) -> Any | None: ...
T = TypeVar('T')

class TypeExemplars:
    """
    This class returns examples of a Type, given its class name.
    """
    TYPE_EXEMPLARS: dict[str, Any]
    @staticmethod
    def example(t: type[T]) -> T | None:
        """
        Return an example of a class.
        """
    @staticmethod
    def contains(t: type[T]) -> bool: ...

def check_halide_import() -> bool:
    """checks if we have halide available"""

CUDA_BACKEND: Incomplete

class Status(Enum):
    """
    The Status return value enum for Config Fuzzer
    """
    SKIPPED = 'skipped'
    PASSED = 'passed'
    FAILED_COMPILE = 'failed_compile'
    FAILED_RUN_COMPILE_EXCEPTION = 'failed_run_compile_exception'
    FAILED_RUN_EAGER_EXCEPTION = 'failed_run_eager_exception'
    FAILED_RUN_RETURN = 'failed_run_return'
    def failing(self) -> bool:
        """
        Convenience method to check whether these status represent failure.
        """

TYPE_OVERRIDES: dict[str, list[Any]]
SamplingType = Callable[[str, type[Any], Any], Any]

class SamplingMethod(Enum):
    """
    This class handles the process of assigning concrete values to type annotations. So a type annotation of
    ```python
    foo: Optional[int] = None
    ```
    Will be assigned an int if the dispatch function gets TOGGLE, or a 50/50 split between an int and None if it gets
    RANDOM.
    """
    TOGGLE = 'TOGGLE'
    RANDOM = 'RANDOM'
    @staticmethod
    def _generate_value_for_type(random_sample: bool, field_name: str, type_hint: type[Any], default: Any) -> Any:
        """
        Generates a value of a type based on the setting.
        """
    @staticmethod
    def dispatch(sm: SamplingMethod) -> SamplingType:
        """
        Returns a function that will generate values from a type, based on the SamplingMethod passed in.
        """

class Default:
    """
    Singleton default object that will cause the ConfigFuzzer to always use the default value set in the config.
    """

DEFAULT: Incomplete
ComboType = tuple[str, ...]

class ResultType:
    """
    The mapping of the combo strings to the result status after running the config fuzzer.
    """
    _vals: dict[ComboType, Status]
    def __repr__(self) -> str: ...
    def __init__(self) -> None: ...
    def __len__(self) -> int: ...
    def num_ran(self) -> int:
        """
        Returns how many combos actually ran (weren't skipped).
        """
    def set(self, combo: ComboType, status: Status) -> None: ...
    def lookup(self, combo: ComboType) -> Status | None: ...
    def keys(self) -> KeysView[ComboType]: ...
ConfigType = dict[str, Any]
FactoryOutputType = Callable[[], bool]
FactoryType = Callable[[], FactoryOutputType]
MODULE_DEFAULTS: dict[str, ConfigType]

class ConfigFuzzer:
    '''
    This tool makes it easy to search through config state-space with a minimal reproduction or test, either for
      debugging or just bug hunting.
    It has two entry points:
     - bisect, which randomly flips configs and tries to find the minimal reproduction upon failure.
     - fuzz_n_tuple, which tries every combination of n configs. This grows quickly as a function of n, so beware.
    bisect is recommended, but fuzz_n_tuple can give you peace of mind that a new config will compose with
      every other config.

    The main interface is a function factory that will return Callables to be torch.compiled. This function factory
      should return a test function when it\'s called. Said test function returns a boolean, which determines whether
      the ConfigFuzzer considers it a successful run or not. Throwing an exception from within the function will be
      considered a failure as well.

    # Example usage:

    ```python
    import torch._inductor.config as cfg


    def create_simple_test_model_gpu() -> FactoryOutputType:
        batch_size = 32
        seq_length = 50
        hidden_size = 768

        def test_fn() -> bool:
            inp = torch.randn(batch_size, seq_length, hidden_size, device="cuda")
            weight = torch.randn(hidden_size, hidden_size, device="cuda")
            matmul_output = inp @ weight
            final_output = torch.nn.LayerNorm(hidden_size, device="cuda")(matmul_output)
            return True

        return test_fn


    fuzzer = ConfigFuzzer(cfg, create_simple_test_model_gpu, seed=2)

    # Test every pair of configs:
    results = fuzzer.fuzz_n_tuple(n, max_combinations=10000000)

    visualize_results(n, results)

    # Test random configs with bisection:
    ret = fuzzer.bisect(num_attempts=10)

    # reproduce a failing config
    fuzzer.reproduce(
        [{"triton.autotune_pointwise": ..., "coordinate_descent_tuning": ...}]
    )
    ```

    The list of known failures on inductor config are:
    cpp_wrapper, triton_debug_sync_graph
    cpp_wrapper, triton_debug_sync_kernel
    cpp_wrapper, disable_cpp_codegen
    combo_kernels, benchmark_combo_kernel, profile_bandwidth, profile_bandwidth_regex
    trace.enabled, trace.save_real_tensors
    '''
    sample: SamplingType
    default: ConfigType
    seed: Incomplete
    test_timeout: Incomplete
    detailed_results: dict[ComboType, dict[str, Any]]
    config_module: Incomplete
    test_model_fn_factory: Incomplete
    fields: dict[str, _ConfigEntry]
    def __init__(self, config_module: ConfigModule, test_model_fn_factory: FactoryType, seed: int, default: ConfigType | None = None, sm: SamplingMethod = ..., test_timeout: int = 3600) -> None:
        """
        Args:
            config_module: The module containing the configs to fuzz
            test_model_fn_factory: Function that returns a test model, which runs and returns True if successful, or
              the outputs if they should be compared with eager
            seed: Randomness seed.
            default: Default values for the config. Inductor has preset based on know failures.
            sm: How type value samples are generated, default TOGGLE.
            test_timeout: max time a test can take.
        """
    def __repr__(self) -> str: ...
    def _set_config(self, field_name: str, value: Any) -> None:
        """Set a config value in the module."""
    def _reset_configs(self) -> None:
        """Reset all configs to their default values."""
    def new_config(self) -> ConfigType:
        """creates a new config from the default"""
    def reproduce(self, configs: Sequence[ConfigType]) -> ResultType:
        """entrypoint to reproduce any failure"""
    def _reproduce_single_helper(self, conf: ConfigType, results: ResultType) -> None: ...
    def reproduce_single(self, config: ConfigType) -> ResultType: ...
    def _fuzz_helper(self, results: ResultType, combo: ComboType) -> Status: ...
    def fuzz_n_tuple(self, n: int, max_combinations: int = 1000) -> ResultType:
        """
        Test every combination of n configs.

        returns a dict of this shape: {(config-1, config-2... config-n): status}
        """
    def save_state(self, filename: str = 'fuzzer_state.pkl') -> None:
        """Save the current fuzzer state to a file"""
    results: Incomplete
    def load_state(self, filename: str = 'fuzzer_state.pkl') -> None:
        """Load fuzzer state from a file"""
    def timeout_handler(self, signum: int, frame: FrameType | None) -> None: ...
    def test_config(self, results: ResultType, config: ConfigType) -> Status:
        """
        Tests a config by calling the function produced by the factory function.
        """
    def bisect(self, num_attempts: int = 100, p: float = 0.5) -> list[ConfigType]:
        """
        Test configs and bisect to minimal failing configuration.
        """
    def _bisect_failing_config(self, results: ResultType, failing_config: ConfigType) -> ConfigType | None: ...
    def _bisect_failing_config_helper(self, results: ResultType, failing_config: list[tuple[str, Any]]) -> ConfigType | None:
        """
        Bisect a failing configuration to find minimal set of configs that cause failure.

        Splits it into halves, then fourths, then tries dropping configs one-by-one.
        """

def visualize_results(n: int, results: ResultType, filename: str = 'results.html') -> None:
    """
    Creates an HTML document representing the results of running the fuzzer with fuzz_n_tuple, with n = 2.
    """
