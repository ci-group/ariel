from torch.utils._config_typing import *
from _typeshed import Incomplete
from torch._environment import is_fbcode as is_fbcode
from torch.utils._config_module import Config as Config, install_config_module as install_config_module
from typing import Literal

functionalize_rng_ops: bool
fake_tensor_allow_meta: Incomplete
debug_assert: bool
debug_partitioner: Incomplete
decompose_custom_triton_ops: bool
static_weight_shapes: bool
treat_parameters_as_free_to_save: bool
cse: bool
enable_autograd_cache: bool
autograd_cache_allow_custom_autograd_functions: bool
bundled_autograd_cache: bool

def remote_autograd_cache_default() -> bool | None: ...

enable_remote_autograd_cache: Incomplete
view_replay_for_aliased_outputs: Incomplete
max_dist_from_bw: int
ban_recompute_used_far_apart: bool
ban_recompute_long_fusible_chains: bool
ban_recompute_materialized_backward: bool
ban_recompute_not_in_allowlist: bool
ban_recompute_reductions: bool
recompute_views: bool
activation_memory_budget: float
activation_memory_budget_runtime_estimator: str
activation_memory_budget_solver: str
visualize_memory_budget_pareto: Incomplete
memory_budget_pareto_dir: Incomplete
aggressive_recomputation: bool
fake_tensor_allow_unsafe_data_ptr_access: bool
unlift_effect_tokens: bool
custom_op_default_layout_constraint: Literal['needs_exact_strides', 'needs_fixed_stride_order', 'flexible_layout']
fake_tensor_crossref: bool
fake_tensor_propagate_real_tensors: bool
donated_buffer: Incomplete
torch_compile_graph_format: Incomplete
generate_fake_kernels_from_real_mismatches: bool
graphsafe_rng_functionalization: bool
strict_autograd_cache: bool
unsafe_allow_optimization_of_collectives: bool
disable_guess_zero_tangent_for_mutated_input_subclass: bool
guess_tangent_strides_as_outputs: bool
_broadcast_rank0_decision: bool
saved_tensors_hooks_filtering_mode: str
