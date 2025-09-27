from torch.utils._config_typing import *
import torch
import torch._inductor
import torch._inductor.custom_graph_pass
from _typeshed import Incomplete
from torch._environment import is_fbcode as is_fbcode
from torch.utils._config_module import Config as Config, get_tristate_env as get_tristate_env, install_config_module as install_config_module
from typing import Any, Callable, Literal

inplace_padding: Incomplete
can_inplace_pad_graph_input: bool

def fx_graph_remote_cache_default() -> bool | None: ...
def vec_isa_ok_default() -> bool | None: ...
def autotune_remote_cache_default() -> bool | None: ...
def bundled_autotune_remote_cache_default() -> bool | None: ...
def bundle_triton_into_fx_graph_cache_default() -> bool | None: ...
def static_cuda_launcher_default() -> bool: ...
def prologue_fusion_enabled() -> bool: ...

enable_auto_functionalized_v2: Incomplete
debug: bool
disable_progress: bool
verbose_progress: bool
precompilation_timeout_seconds: int
fx_graph_cache: bool
fx_graph_remote_cache: bool | None
bundle_triton_into_fx_graph_cache: bool | None
non_blocking_remote_cache_write: bool
autotune_local_cache: bool
autotune_remote_cache: bool | None
bundled_autotune_remote_cache: bool | None
force_disable_caches: bool
unsafe_skip_cache_dynamic_shape_guards: bool
unsafe_marked_cacheable_functions: dict[str, str]
sleep_sec_TESTING_ONLY: int | None
triton_kernel_default_layout_constraint: Literal['needs_fixed_stride_order', 'flexible_layout']
cpp_wrapper: bool
cpp_wrapper_build_separate: bool
cpp_cache_precompile_headers: bool
online_softmax: Incomplete
dce: bool
static_weight_shapes: bool
size_asserts: Incomplete
nan_asserts: Incomplete
scalar_asserts: Incomplete
alignment_asserts: Incomplete
pick_loop_orders: bool
inplace_buffers: bool
allow_buffer_reuse: bool
memory_planning: Incomplete
use_fast_math: Incomplete
bfloat16_atomic_adds_enabled: bool
memory_pool: Literal['none', 'intermediates', 'outputs', 'combined']
benchmark_harness: bool
epilogue_fusion: bool
prologue_fusion: Incomplete
epilogue_fusion_first: bool
pattern_matcher: bool
b2b_gemm_pass: bool
post_grad_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType
post_grad_custom_post_pass: torch._inductor.custom_graph_pass.CustomGraphPassType
joint_custom_pre_pass: Callable[[torch.fx.Graph], None] | None
joint_custom_post_pass: Callable[[torch.fx.Graph], None] | None
pre_grad_custom_pass: Callable[[torch.fx.graph.Graph], None] | None
_pre_fusion_custom_pass: Callable[[list['torch._inductor.scheduler.BaseSchedulerNode']], list['torch._inductor.scheduler.BaseSchedulerNode']] | None
_post_fusion_custom_pass: Callable[[list['torch._inductor.scheduler.BaseSchedulerNode']], list['torch._inductor.scheduler.BaseSchedulerNode']] | None
split_cat_fx_passes: bool
efficient_conv_bn_eval_fx_passes: bool
is_predispatch: bool
group_fusion: bool
batch_fusion: bool
pre_grad_fusion_options: dict[str, dict[str, Any]]
post_grad_fusion_options: dict[str, dict[str, Any]]
reorder_for_locality: bool
dynamic_scale_rblock: Incomplete
force_fuse_int_mm_with_mul: bool
use_mixed_mm: bool
fx_passes_numeric_check: dict[str, Any]
mixed_mm_choice: Literal['default', 'triton', 'aten', 'heuristic']
reorder_for_compute_comm_overlap: bool
reorder_for_compute_comm_overlap_passes: list[str | Callable[[list['torch._inductor.scheduler.BaseSchedulerNode']], list['torch._inductor.scheduler.BaseSchedulerNode']]]
reorder_prefetch_limit: int | None
reorder_for_peak_memory: bool
estimate_op_runtime: str
intra_node_bw: int
inter_node_bw: int
use_experimental_benchmarker: bool
max_autotune: Incomplete
max_autotune_pointwise: Incomplete
max_autotune_gemm: Incomplete
disable_decompose_k: Incomplete
autotune_num_choices_displayed: int | None
graph_partition: bool
force_same_precision: Incomplete
max_autotune_gemm_backends: Incomplete
max_autotune_conv_backends: Incomplete
max_autotune_gemm_search_space: Literal['DEFAULT', 'EXHAUSTIVE']
max_autotune_flex_search_space: Literal['DEFAULT', 'EXHAUSTIVE']
autotune_fallback_to_aten: bool
unbacked_symint_fallback: int
search_autotune_cache: Incomplete
save_args: Incomplete
autotune_in_subproc: Incomplete
max_autotune_subproc_result_timeout_seconds: float
max_autotune_subproc_graceful_timeout_seconds: float
max_autotune_subproc_terminate_timeout_seconds: float
autotune_multi_device: Incomplete
coordinate_descent_tuning: Incomplete
coordinate_descent_check_all_directions: Incomplete
coordinate_descent_search_radius: Incomplete
autoheuristic_collect: Incomplete
autoheuristic_use: Incomplete

def run_autoheuristic(name: str) -> bool: ...
def collect_autoheuristic(name: str) -> bool: ...
def use_autoheuristic(name: str) -> bool: ...

autoheuristic_log_path: Incomplete
layout_opt_default: Incomplete
layout_optimization: Incomplete
force_layout_optimization: Incomplete
keep_output_stride: Incomplete
warn_mix_layout: Incomplete
realize_reads_threshold: int
realize_opcount_threshold: int
realize_acc_reads_threshold: int
fallback_random: bool
implicit_fallbacks: bool
assume_unaligned_fallback_output: Incomplete
aggressive_fusion: bool
debug_fusion: bool
benchmark_fusion: bool
enabled_metric_tables: Incomplete
loop_ordering_after_fusion: bool
score_fusion_memory_threshold: int
benchmark_epilogue_fusion: Incomplete
max_epilogue_benchmarked_choices: int
max_fusion_size: int
max_fusion_buffer_group_pairwise_attempts: int
max_pointwise_cat_inputs: int
force_pointwise_cat: bool
unroll_reductions_threshold: int
comment_origin: bool
conv_1x1_as_mm: bool
split_reductions: bool
min_num_split: Incomplete
benchmark_kernel: Incomplete
constant_and_index_propagation: bool
always_keep_tensor_constants: bool
assert_indirect_indexing: bool
compute_all_bounds: bool
combo_kernels: bool
benchmark_combo_kernel: bool
combo_kernels_autotune: int
combo_kernel_allow_mixed_sizes: int
combo_kernel_foreach_dynamic_shapes: bool
joint_graph_constant_folding: bool
debug_index_asserts: bool
emulate_precision_casts: Incomplete
is_nightly_or_source: Incomplete
developer_warnings: Incomplete
optimize_scatter_upon_const_tensor: Incomplete
add_pre_grad_passes: str | None
remove_pre_grad_passes: str | None

def decide_worker_start_method() -> str: ...

worker_start_method: str
worker_suppress_logging: bool
_fuse_ddp_communication: bool
_fuse_ddp_bucket_size: int
_fuse_ddp_communication_passes: list[Callable[..., None] | str]
_micro_pipeline_tp: bool

class _collective:
    auto_select: bool
    one_shot_all_reduce_threshold_bytes: int

def parallel_compile_enabled_internally() -> bool:
    """
    TODO: Remove when parallel compiled is fully enabled internally. For rollout, use a
    knob to enable / disable. The justknob should not be performed at import, however.
    So for fbcode, we assign compile_threads to 'None' below and initialize lazily in
    async_compile.py.
    """
def decide_compile_threads() -> int:
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform
    3. decide by the number of CPU cores
    """

compile_threads: int | None
use_static_cuda_launcher: bool
static_launch_user_defined_triton_kernels: bool
strict_static_cuda_launcher: bool
global_cache_dir: str | None
kernel_name_max_ops: int
shape_padding: Incomplete
comprehensive_padding: Incomplete
pad_channels_last: bool
disable_padding_cpu: bool
padding_alignment_bytes: int
padding_stride_threshold: int
pad_outputs: bool
bw_outputs_user_visible: bool
force_shape_pad: bool
permute_fusion: Incomplete
profiler_mark_wrapper_call: bool
generate_intermediate_hooks: bool
debug_ir_traceback: bool
_raise_error_for_testing: bool
_profile_var: Incomplete
profile_bandwidth: Incomplete
profile_bandwidth_regex: Incomplete
profile_bandwidth_output: str | None
profile_bandwidth_with_do_bench_using_profiling: Incomplete
disable_cpp_codegen: bool
freezing: bool
freezing_discard_parameters: bool
decompose_mem_bound_mm: bool
assume_aligned_inputs: bool
unsafe_ignore_unsupported_triton_autotune_args: bool
check_stack_no_cycles_TESTING_ONLY: bool
always_complex_memory_overlap_TESTING_ONLY: bool
enable_linear_binary_folding: Incomplete
annotate_training: bool
enable_caching_generated_triton_templates: bool

class cpp:
    threads: int
    no_redundant_loops: Incomplete
    dynamic_threads: Incomplete
    simdlen: int | None
    min_chunk_size: Incomplete
    cxx: tuple[Literal[None], str]
    enable_kernel_profile: Incomplete
    weight_prepack: Incomplete
    inject_relu_bug_TESTING_ONLY: str | None
    inject_log1p_bug_TESTING_ONLY: str | None
    vec_isa_ok: bool | None
    descriptive_names: Literal['torch', 'original_aten', 'inductor_node']
    max_horizontal_fusion_size: Incomplete
    fallback_scatter_reduce_sum: Incomplete
    enable_unsafe_math_opt_flag: Incomplete
    enable_floating_point_contract_flag: Incomplete
    enable_tiling_heuristics: Incomplete
    enable_grouped_gemm_template: bool
    gemm_max_k_slices: Incomplete
    gemm_cache_blocking: Incomplete
    gemm_thread_factors: Incomplete
    enable_loop_tail_vec: bool
    enable_concat_linear: bool
    use_decompose_tanh: Incomplete
    use_small_dequant_buffer: bool

class triton:
    cudagraphs: Incomplete
    cudagraph_trees: bool
    cudagraph_skip_dynamic_graphs: bool
    cudagraph_capture_sizes: tuple[int | tuple[int, ...]] | None
    slow_path_cudagraph_asserts: bool
    cudagraph_trees_history_recording: bool
    cudagraph_support_input_mutation: Incomplete
    cudagraph_unexpected_rerecord_limit: int
    cudagraph_dynamic_shape_warn_limit: int | None
    force_cudagraph_sync: bool
    force_cudagraphs_warmup: bool
    fast_path_cudagraph_asserts: bool
    skip_cudagraph_warmup: bool
    debug_sync_graph: bool
    debug_sync_kernel: bool
    dense_indexing: bool
    coalesce_tiling_analysis: bool
    max_tiles: int | None
    prefer_nd_tiling: bool
    autotune_pointwise: bool
    autotune_cublasLt: bool
    autotune_at_compile_time: bool | None
    autotune_with_sample_inputs: bool
    tile_reductions: bool
    tiling_prevents_pointwise_fusion: bool
    tiling_prevents_reduction_fusion: bool
    unique_kernel_names: Incomplete
    unique_user_kernel_names: Incomplete
    descriptive_names: Literal['torch', 'original_aten', 'inductor_node']
    persistent_reductions: Incomplete
    cooperative_reductions: Incomplete
    force_cooperative_reductions: bool
    multi_kernel: Literal[0, 1, 2, 3]
    divisible_by_16: Incomplete
    min_split_scan_rblock: int
    store_cubin: bool
    spill_threshold: int
    use_block_ptr: bool
    inject_relu_bug_TESTING_ONLY: str | None
    codegen_upcast_to_fp32: bool
    enable_persistent_tma_matmul: Incomplete
    skip_l1_cache: Incomplete
    disallow_failing_autotune_kernels_TESTING_ONLY: bool

class aot_inductor:
    """
    Settings for Ahead-Of-Time Inductor Compilation
    """
    output_path: str
    debug_compile: Incomplete
    compile_wrapper_opt_level: Incomplete
    debug_intermediate_value_printer: Literal['0', '1', '2', '3']
    filtered_kernel_names: Incomplete
    serialized_in_spec: str
    serialized_out_spec: str
    use_runtime_constant_folding: bool
    force_mmap_weights: bool
    package: bool
    package_cpp_only: bool
    metadata: dict[str, str]
    raise_error_on_ignored_optimization: bool
    dump_aoti_minifier: bool
    repro_level: int
    presets: dict[str, Any]
    allow_stack_allocation: bool
    use_minimal_arrayref_interface: bool
    package_constants_in_so: bool
    package_constants_on_disk: bool
    precompile_headers: bool
    embed_kernel_binary: bool
    emit_multi_arch_kernel: bool
    model_name_for_generated_files: str | None
    custom_ops_to_c_shims: dict[torch._ops.OpOverload, list[str]]
    custom_op_libs: list[str] | None

class cuda:
    """Settings for cuda backend, today this consists of cutlass"""
    arch: str | None
    version: str | None
    compile_opt_level: Literal['-O0', '-O1', '-O2', '-O3', '-OS']
    enable_cuda_lto: bool
    enable_ptxas_info: bool
    enable_debug_info: bool
    use_fast_math: bool
    cutlass_dir: Incomplete
    cutlass_max_profiling_configs: int | None
    cutlass_max_profiling_swizzle_options: list[int]
    cutlass_epilogue_fusion_enabled: Incomplete
    cutlass_tma_only: bool
    cuda_cxx: str | None
    cutlass_backend_min_gemm_size: int
    generate_test_runner: bool
    cutlass_op_allowlist_regex: str | None
    cutlass_op_denylist_regex: str | None
    cutlass_instantiation_level: str
    cutlass_presets: str | None
    cutlass_hash_with_compile_cmd: bool
    cutlass_prescreening: bool
    cutlass_enabled_ops: str
    use_binary_remote_cache: bool
    upload_to_binary_remote_cache: bool
    binary_remote_cache_force_write: bool

class rocm:
    arch: list[str]
    ck_supported_arch: list[str]
    compile_opt_level: Literal['-O0', '-O1', '-O2', '-O3', '-Os', '-Oz', '-Omin', '-Ofast', '-Omax']
    is_debug: bool
    save_temps: bool
    use_fast_math: bool
    flush_denormals: bool
    print_kernel_resource_usage: bool
    rocm_home: str | None
    ck_dir: Incomplete
    generate_test_runner: bool
    n_max_profiling_configs: int | None
    ck_max_profiling_configs: int | None
    ck_tile_max_profiling_configs: int | None
    use_preselected_instances: bool
    kBatch_sweep: list[int] | None
    split_k_threshold: int

cpu_backend: Literal['cpp', 'triton', 'halide']
cuda_backend: Literal['triton', 'halide']

class halide:
    cpu_target: str
    gpu_target: str
    scheduler_cuda: Literal['Anderson2021', 'Li2018', 'Adams2019', 'Mullapudi2016']
    scheduler_cpu: Literal['Anderson2021', 'Li2018', 'Adams2019', 'Mullapudi2016']
    asserts: bool
    debug: bool
    scan_kernels: bool

class trace:
    enabled: Incomplete
    save_real_tensors: Incomplete
    debug_dir: str | None
    debug_log: bool
    info_log: bool
    fx_graph: bool
    fx_graph_transformed: bool
    ir_pre_fusion: bool
    ir_post_fusion: bool
    output_code: bool
    graph_diagram: Incomplete
    draw_orig_fx_graph: Incomplete
    dot_graph_shape: Incomplete
    log_url_for_graph_xform: Incomplete
    compile_profile: bool
    upload_tar: Callable[[str], None] | None
    log_autotuning_results: Incomplete
    log_inductor_triton_kernel_to_post_grad_node_info: bool

_save_config_ignore: list[str]
_cache_config_ignore_prefix: list[str]
external_matmul: list[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]]

class test_configs:
    force_extern_kernel_in_multi_template: bool
    max_mm_configs: int | None
    runtime_triton_dtype_assert: bool
    static_cpp_dtype_assert: bool
    autotune_choice_name_regex: str | None
    autotune_choice_desc_regex: str | None
    graphsafe_rng_func_ignores_fallback_random: bool
