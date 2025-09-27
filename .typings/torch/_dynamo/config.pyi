from torch.utils._config_typing import *
from _typeshed import Incomplete
from torch._environment import is_fbcode as is_fbcode
from torch.utils._config_module import Config as Config, get_tristate_env as get_tristate_env, install_config_module as install_config_module
from typing import Any, Callable, Literal

log_file_name: str | None
verbose: Incomplete
verify_correctness: bool
minimum_call_count: int
dead_code_elimination: bool
recompile_limit: int
accumulated_recompile_limit: int
skip_code_recursive_on_recompile_limit_hit: bool
fail_on_recompile_limit_hit: bool
cache_size_limit: int
accumulated_cache_size_limit: int
skip_code_recursive_on_cache_limit_hit: bool
fail_on_cache_limit_hit: bool
specialize_int: bool
specialize_float: bool
dynamic_shapes: bool
use_lazy_graph_module: Incomplete
assume_static_by_default: bool
automatic_dynamic_shapes: bool
automatic_dynamic_shapes_mark_as: Literal['dynamic', 'unbacked']
force_parameter_static_shapes: bool
force_nn_module_property_static_shapes: bool
allow_ignore_mark_dynamic: bool
guard_nn_modules: bool
guard_nn_modules_using_dict_tags: bool
prepare_freezing: Incomplete
traceable_tensor_subclasses: set[type[Any]]
nontraceable_tensor_subclasses: set[type[Any]]
suppress_errors: Incomplete
replay_record_enabled: Incomplete
rewrite_assert_with_torch_assert: bool
disable: Incomplete
cprofile: Incomplete
skipfiles_inline_module_allowlist: dict[Any, Any]
allowed_functions_module_string_ignorelist: Incomplete
repro_after: Incomplete
repro_level: Incomplete
repro_forward_only: Incomplete
repro_tolerance: float
repro_ignore_non_fp: Incomplete
same_two_models_use_fp64: bool
capture_scalar_outputs: Incomplete
capture_dynamic_output_shape_ops: Incomplete
prefer_deferred_runtime_asserts_over_guards: bool
allow_complex_guards_as_runtime_asserts: bool
force_unspec_int_unbacked_size_like_on_torchrec_kjt: bool
allow_unspec_int_on_nn_module: bool
optimize_ddp: bool | Literal['ddp_optimizer', 'python_reducer', 'python_reducer_without_compiled_forward', 'no_optimization']
do_not_emit_runtime_asserts: bool
skip_torchrec: bool
dont_skip_tracing: bool
optimize_ddp_lazy_compile: bool
skip_fsdp_guards: bool
skip_fsdp_hooks: bool
skip_nnmodule_hook_guards: bool
skip_no_tensor_aliasing_guards_on_parameters: bool
skip_tensor_guards_with_matching_dict_tags: bool
raise_on_ctx_manager_usage: bool
raise_on_unsafe_aot_autograd: bool
error_on_nested_jit_trace: bool
error_on_nested_fx_trace: bool
allow_rnn: bool
capture_sparse_compute: Incomplete
error_on_recompile: bool
report_guard_failures: bool
base_dir: Incomplete
trace_numpy: bool
numpy_default_float: str
numpy_default_complex: str
numpy_default_int: str
use_numpy_random_stream: bool
enable_cpp_guard_manager: bool
enable_cpp_symbolic_shape_guards: bool
enable_trace_contextlib: bool
enable_trace_unittest: bool
enable_faithful_generator_behavior: bool
inline_inbuilt_nn_modules: Incomplete
install_free_tensors: bool
enable_cpp_framelocals_guard_eval: bool
use_graph_deduplication: bool
track_nodes_for_deduplication: bool
graph_deduplication_lint: bool
issue_3_13_0_warning: bool
allow_empty_graphs: bool
record_compile_time_instruction_count: bool

def default_debug_dir_root(): ...

debug_dir_root: Incomplete
_save_config_ignore: Incomplete
cudagraph_backend_keep_input_mutation: bool
cudagraph_backend_support_input_mutation: bool
only_allow_pt2_compliant_ops: bool
capture_autograd_function: bool
capture_func_transforms: bool
log_compilation_metrics: bool
reorderable_logging_functions: set[Callable[[Any], None]]
ignore_logger_methods: set[Callable[..., Any]]
inject_BUILD_SET_unimplemented_TESTING_ONLY: bool
_autograd_backward_strict_mode_banned_ops: Incomplete
_autograd_backward_strict_mode_conditional_banned_ops: Incomplete
fake_tensor_cache_enabled: Incomplete
fake_tensor_cache_crosscheck_enabled: Incomplete
fake_tensor_disable_inference_mode: bool
compiled_autograd: bool
compiled_autograd_kwargs_override: dict[str, Any]
enable_compiler_collectives: Incomplete
automatic_dynamic_local_pgo: bool
automatic_dynamic_remote_pgo: bool | None
_unsafe_skip_fsdp_module_guards: Incomplete
run_gc_after_compile: Incomplete
wrap_top_frame: bool
record_runtime_overhead: bool
_custom_ops_profile: Any | None

def _make_closure_patcher(**changes) -> None: ...
