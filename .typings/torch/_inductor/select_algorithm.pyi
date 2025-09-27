import contextlib
import dataclasses
import functools
import sympy
import torch
from . import config as config, ir as ir, lowering as lowering
from ..utils._sympy.functions import CeilDiv as CeilDiv
from .autotune_process import TensorMeta as TensorMeta, TritonBenchmarkRequest as TritonBenchmarkRequest, TritonCPUBenchmarkRequest as TritonCPUBenchmarkRequest, TritonGPUBenchmarkRequest as TritonGPUBenchmarkRequest
from .codecache import PersistentCache as PersistentCache, PyCodeCache as PyCodeCache, code_hash as code_hash
from .codegen.common import CSEVariable as CSEVariable, IndentedBuffer as IndentedBuffer, KernelTemplate as KernelTemplate, OpOverrides as OpOverrides, WorkspaceArg as WorkspaceArg, WorkspaceZeroMode as WorkspaceZeroMode
from .codegen.simd_kernel_features import SIMDKernelFeatures as SIMDKernelFeatures
from .codegen.subgraph import SubgraphChoiceCaller as SubgraphChoiceCaller
from .codegen.triton import TritonKernel as TritonKernel, TritonScheduling as TritonScheduling, gen_common_triton_imports as gen_common_triton_imports, texpr as texpr
from .codegen.triton_utils import config_of as config_of, equal_1_arg_indices as equal_1_arg_indices, signature_to_meta as signature_to_meta
from .codegen.wrapper import pexpr as pexpr
from .exc import CUDACompileError as CUDACompileError
from .ir import ChoiceCaller as ChoiceCaller, PrimitiveInfoType as PrimitiveInfoType
from .ops_handler import StoreMode as StoreMode
from .runtime.benchmarking import benchmarker as benchmarker
from .runtime.hints import DeviceProperties as DeviceProperties
from .runtime.triton_compat import HAS_WARP_SPEC as HAS_WARP_SPEC
from .runtime.triton_heuristics import FixedGrid as FixedGrid
from .utils import FakeIndentedBuffer as FakeIndentedBuffer, Placeholder as Placeholder, ceildiv as ceildiv, do_bench_using_profiling as do_bench_using_profiling, get_dtype_size as get_dtype_size, is_gpu as is_gpu, restore_stdout_stderr as restore_stdout_stderr, sympy_dot as sympy_dot, sympy_index_symbol as sympy_index_symbol, sympy_product as sympy_product, triton_type as triton_type, triton_type_to_torch as triton_type_to_torch, unique as unique
from .virtualized import V as V
from _typeshed import Incomplete
from collections.abc import Sequence
from torch._dynamo.device_interface import get_interface_for_device as get_interface_for_device
from torch._dynamo.testing import rand_strided as rand_strided
from torch._dynamo.utils import counters as counters, dynamo_timed as dynamo_timed, identity as identity, preserve_rng_state as preserve_rng_state
from torch._inductor.codegen.simd import IterationRangesRoot as IterationRangesRoot
from torch._inductor.utils import clear_on_fresh_cache as clear_on_fresh_cache
from torch.utils._filelock import FileLock as FileLock
from torch.utils._ordered_set import OrderedSet as OrderedSet
from types import ModuleType
from typing import Any, Callable, NamedTuple
from typing_extensions import Self

log: Incomplete
VERIFY: dict[str, Any]
PRINT_AUTOTUNE: bool
DEBUG: bool

class KernelNamespace: ...

extern_kernels: Incomplete

@dataclasses.dataclass
class BenchmarkTensors:
    """Represents a set of inputs and outputs for autotuning with a template"""
    input_tensors: list[torch.Tensor]
    output_tensor: torch.Tensor | None
    def unpack(self): ...

@dataclasses.dataclass
class AutotuneArgs:
    """During autotuning, we need to pass the same inputs to all choices.
    Note:
        Since we typically have a mix of external choices and triton choices, we create
        two lists of inputs for the same underlying buffers:
        - External inputs (for aten kernels): Include offset for sliced tensors
        - Triton inputs: Use base pointer for sliced tensors, without offset
    """
    triton: BenchmarkTensors
    extern: BenchmarkTensors
    expected: torch.Tensor | None = ...
    def get_benchmark_tensors(self, extern: bool = False) -> BenchmarkTensors:
        """Returns the inputs and output tensors for a given choice."""
    @classmethod
    def from_choice_args(cls, example_inputs: list[torch.Tensor], example_inputs_extern: list[torch.Tensor], out: torch.Tensor, out_extern: torch.Tensor, expected: torch.Tensor | None = None) -> Self:
        """Factory method to create AutotuneInputs from separate inputs/outputs"""
    def verify(self, **kwargs) -> None:
        """Verify the correctness of the benchmarking results"""

class PartialRender:
    """
    Some parts of a template need to be generated at the end, but
    inserted into the template at the start.  This allows doing a bunch
    of replacements after the initial render.
    """
    code: Incomplete
    replacement_hooks: Incomplete
    def __init__(self, code, replacement_hooks) -> None: ...
    def finalize_hook(self, hook_key: str, strict: bool = True) -> None: ...
    def finalize_all(self) -> str: ...

@dataclasses.dataclass()
class SubgraphInfo:
    body: IndentedBuffer
    template_mask: str | None = ...
    template_out: str | None = ...
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: V.WrapperHandler | None = ...
    range_trees: list['IterationRangesRoot'] | None = ...
    numels = ...
    only_copy_if_non_none_fields = ...
    def __post_init__(self) -> None: ...
    def to_dict(self): ...

class ModificationWrapper(V.WrapperHandler):
    """Handles placeholder substitutions during subgraph processing."""
    name: Incomplete
    kernel: Incomplete
    fixed_inputs: Incomplete
    mask: Incomplete
    def __init__(self, kernel, subgraph_number: int, fixed_inputs: dict[str, Any], mask: str | None) -> None: ...
    def load(self, name: str, index: sympy.Expr):
        """Handle loading from tensor or fixed input."""
    def indirect_indexing(self, index_var: str, size, check, wrap_neg: bool = True):
        """Convert index variable to symbolic form."""
    def store(self, name: str, index: sympy.Expr, value: CSEVariable, mode: StoreMode = None) -> str:
        """Currently only supports stores for atomic adds coming from scatter nodes
        This is used by flex_attention's backwards grad for captured buffers, see
        zeros_and_scatter lowering
        """
    def _add_kernel_input(self, name: str):
        """Add name as input to kernel and return input ref."""
    def _process_indexing(self, index):
        """Process and rename indexing, adding symbols as kernel inputs."""
RecordedEventsType = list[tuple[str, list[Any], dict[str, Any]]]

class TritonTemplateKernel(TritonKernel):
    input_nodes: Incomplete
    output_node: Incomplete
    named_input_nodes: Incomplete
    defines: Incomplete
    kernel_name: Incomplete
    use_jit: Incomplete
    num_stages: Incomplete
    num_warps: Incomplete
    num_consumer_groups: Incomplete
    num_buffers_warp_spec: Incomplete
    grid_fn: Incomplete
    meta: Incomplete
    call_sizes: Incomplete
    prefix_args: Incomplete
    suffix_args: Incomplete
    epilogue_fn: Incomplete
    render_hooks: Incomplete
    triton_meta: dict[str, object] | None
    subgraphs: list[ir.ComputedBuffer] | None
    workspace_arg: Incomplete
    subgraph_bodies: dict[str, SubgraphInfo]
    prologue_supported_inputs: OrderedSet[str]
    prologue_fused_inputs: OrderedSet[str]
    prologue_fused_inputs_preserve_zero: OrderedSet[str]
    body: IndentedBuffer
    compute: IndentedBuffer
    indexing_code: IndentedBuffer
    loads: IndentedBuffer
    stores: IndentedBuffer
    template_mask: str | None
    template_out: str | None
    ops_handler: V.WrapperHandler | None
    cached_replay_events: RecordedEventsType | None
    frozen_layouts_cnt: int
    prologue_loads_all_inputs: Incomplete
    def __init__(self, kernel_name, input_nodes, output_node, defines, num_stages, num_warps, grid_fn, meta, call_sizes, num_consumer_groups: int = 0, num_buffers_warp_spec: int = 0, use_jit: bool = False, prefix_args: int = 0, suffix_args: int = 0, epilogue_fn=..., subgraphs: list[ir.ComputedBuffer] | None = None, workspace_arg: WorkspaceArg | None = None, prologue_loads_all_inputs: bool = False) -> None: ...
    def input_dependent_preserved_state(self) -> str: ...
    def record_input_dependent_tracked_event(self) -> Callable[..., Any]: ...
    def replay_cached_events(self, events: RecordedEventsType) -> None: ...
    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str): ...
    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str): ...
    def need_numel_args(self): ...
    def estimate_kernel_num_bytes(self):
        """
        Estimate the total number of bytes this kernel takes.
        For in/out nodes, sizes are counted twice: once for reading and
        once for writing.
        """
    def jit_lines(self): ...
    def gen_argdefs(self): ...
    def gen_defines(self): ...
    def def_kernel(self, *argnames):
        """
        Hook called from template code to generate function def and
        needed args.
        """
    def size(self, name: str, index: int):
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
    def stride(self, name, index=None):
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """
    def _get_subgraph(self, subgraph_number: int): ...
    def _handle_scatter_graph(self, scatter_graph):
        """Handle processing for a single scatter graph.

        Args:
            scatter_graph: The scatter graph to process
        """
    def modification(self, subgraph_number: int, output_name: str | None, mask: str | None = None, **fixed_inputs) -> str:
        """This creates a modification function for a subgraph.
        To use this inside a template, the first argument should specify which subgraph to codegen for

        Args:
            subgraph_number (int): The index of the subgraph in self.subgraphs
            output_name (Optional[str]): The name of the output variable to store the result in
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
        """
    range_trees: Incomplete
    numels: Incomplete
    template_indices: Incomplete
    def load_input(self, input_name: str, output_name: str, indices: list[Any] | tuple[Any], mask: str | None = None, other: float | int | None = 0.0, indent_width: int = 4):
        """Loads an input and applies any necessary preprocessing or masking.

        Args:
            input_name (str): The name of the input to load.
            indices (Union[List, Tuple]): The index for each dimension of the input.
            val (str): The name of the variable to store the loaded value.
            mask (Optional[str]): An optional mask to use for the load operation.
            other (Optional[Union[float, int]]): The value to use for masked elements. Default is 0.0.
            indent_width (int): The number of spaces to use for indentation.
        """
    def store_output(self, indices: list[Any] | tuple[Any], val: str, mask: str | None = None, indent_width: int = 4):
        """Stores the final output and appends any epilogue fusions if the buffer hasn't been optimized away.

        Args:
            indices (Union[List, Tuple]): The index for each dimension of the output. The dot product of
                these indices and output strides must match `val`.
            val (str): The value to store.
            mask (Optional[str]): An optional mask to use for the store operation. If provided, this mask
                will be applied to the store.
            indent_width (int): The number of spaces to use for indentation. This is used when the call to
                store_output is indented in the kernel definition.
        """
    def render(self, template, kwargs, record_input_dependent_tracked_event: bool = False): ...
    def make_load(self, name, indices, mask):
        """
        Optional helper called from template code to generate the code
        needed to load from an tensor.
        """
    def indexing(self, index: sympy.Expr, *, dense_indexing: bool = False, copy_shape=None, override_mask=None, block_ptr: bool = False):
        """
        Override the default indexing to use our custom mask and force
        dense indexing.
        """
    def codegen_range_tree(self) -> None: ...
    def call_kernel(self, name: str, node: ir.IRNode | None = None): ...
    def kernel_benchmark_extra_args(self) -> list[str]: ...

@functools.cache
def _jinja2_env(): ...

class GenerateAndLoadResult(NamedTuple):
    """
    Return type of TritonTemplate.generate_and_load.
    """
    mod: ModuleType
    extra: str
    input_call_args: tuple[str, ...]
    prologue_supported_inputs: OrderedSet[str]
    kernel_args_sizevars_keys: tuple[sympy.Expr]
    kernel_options: dict[str, Any]

class GeneratedCodeCacheEntry(NamedTuple):
    code: str
    extra: str
    events: list[Any]

class GeneratedCodeCache:
    """
    Cache for generated code. The cache key is a string representation of the input nodes,
    number of stages, number of warps, and call sizes. The cache value is a tuple of the
    generated code, extra code, and events.
    """
    _cache: dict[str, GeneratedCodeCacheEntry]
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def __repr__(self) -> str: ...
    def make_key(self, input_nodes: tuple[ir.IRNode], num_stages: int, num_warps: int, call_sizes: list[sympy.core.symbol.Symbol], prefix_args: int, suffix_args: int, epilogue_fn: Callable[..., Any] | None, epilogue_fn_hash: str | None, subgraphs: list[ir.Buffer] | None, workspace_arg: WorkspaceArg | None, layout: ir.Layout, num_consumer_groups: int, num_buffers_warp_spec: int, kwargs: dict[str, Any]) -> str | None: ...
    def get_entry(self, cache_key: str | None) -> GeneratedCodeCacheEntry | None: ...
    def put_entry(self, cache_key: str | None, code: str, extra: str, events: list[Any]) -> None: ...

class TritonTemplate(KernelTemplate):
    """
    A Triton template is a template that can be used to generate a Triton kernel.
    """
    kernel_type: type[Any]
    index_counter: Incomplete
    all_templates: dict[str, 'TritonTemplate']
    grid: Incomplete
    template: Incomplete
    debug: Incomplete
    _cache_codegen_enabled_for_template: Incomplete
    _generated_code_cache: GeneratedCodeCache
    prologue_loads_all_inputs: Incomplete
    def __init__(self, name: str, grid: Any, source: str, debug: bool = False, cache_codegen_enabled_for_template: bool = False, prologue_loads_all_inputs: bool = False) -> None: ...
    test_cache: bool
    def maybe_append_choice(self, choices: list[Any], **kwargs: Any) -> NotImplementedError | None:
        """
        Maybe generates a new ChoiceCaller and appends it into existing choices.
        Returns None if success, otherwise returns the error.

        choices: A list of ChoiceCallers.
        kwargs: Additional kwargs to be passed to self.generate() to generate a new ChoiceCaller.
        """
    def generate_and_load(self, input_nodes: tuple[ir.IRNode], num_stages: int, num_warps: int, call_sizes: list[sympy.core.symbol.Symbol], prefix_args: int, suffix_args: int, epilogue_fn: Callable[..., Any] | None, epilogue_fn_hash: str | None, subgraphs: list[ir.Buffer] | None, workspace_arg: WorkspaceArg | None, num_consumer_groups: int, num_buffers_warp_spec: int, layout: ir.Layout, kwargs: dict[str, Any], generate_with_caching) -> GenerateAndLoadResult | None:
        """Generate the python code and load it into the current process"""
    def generate(self, input_nodes: tuple[ir.IRNode], layout: ir.Layout, num_stages: int, num_warps: int, num_consumer_groups: int = 0, num_buffers_warp_spec: int = 0, prefix_args: int = 0, suffix_args: int = 0, epilogue_fn: Callable[..., Any] | None = ..., epilogue_fn_hash: str | None = None, subgraphs: list[ir.Buffer] | None = None, mutated_inputs: list[ir.IRNode] | None = None, call_sizes: list[sympy.core.symbol.Symbol] | None = None, workspace_arg: WorkspaceArg | None = None, generate_with_caching: bool = False, **kwargs):
        """This function generates a TritonTemplateCaller

        Args:
            input_nodes: List of input nodes
            layout: Output layout
            num_stages: Number of stages for triton launch
            num_warps: Number of warps for triton launch
            prefix_args: Number of input nodes to be passed as arguments
            suffix_args: Number of input nodes to be passed as arguments
            epilogue_fn: Optional epilogue function to be called on the output
            subgraphs: Optional subgraphs to be passed as arguments, these will be inlined
                into the triton template string
            mutated_inputs: Optional list of input nodes that are mutated by the kernel, this is helpful
                if you need to return multiple outputs. You can pass them as inputs and mark them as
                being mutated by the kernel.
        """

class ExternKernelChoice:
    name: Incomplete
    cpp_kernel_name: Incomplete
    has_out_variant: Incomplete
    op_overload: Incomplete
    use_fallback_kernel: Incomplete
    kernel_creator: Incomplete
    def __init__(self, kernel, cpp_kernel=None, *, name=None, has_out_variant: bool = True, op_overload=None, use_fallback_kernel: bool = False, kernel_creator=None) -> None: ...
    def to_callable(self): ...
    def call_name(self): ...
    @functools.cache
    def hash_key(self): ...
    ordered_kwargs_for_cpp_kernel: Incomplete
    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=(), **kwargs): ...

class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    make_kernel_render: Incomplete
    bmreq: TritonBenchmarkRequest
    log_info: dict[str, Any]
    mutated_inputs: Incomplete
    workspace_arg: Incomplete
    allowed_prologue_inps: Incomplete
    def __init__(self, name, input_nodes, layout, make_kernel_render, description, bmreq, log_info: dict[str, PrimitiveInfoType | list[PrimitiveInfoType]] | None = None, mutated_inputs=None, workspace_arg: WorkspaceArg | None = None, allowed_prologue_inps: OrderedSet[str] | None = None) -> None: ...
    def benchmark(self, *args, out): ...
    def precompile(self) -> None: ...
    def __str__(self) -> str: ...
    def call_name(self): ...
    def hash_key(self): ...
    def output_node(self): ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def get_make_kernel_render(self): ...
    def autoheuristic_id(self): ...

class ExternKernelCaller(ChoiceCaller):
    choice: Incomplete
    kwargs: Incomplete
    has_out_variant: Incomplete
    def __init__(self, choice: ExternKernelChoice, input_nodes, layout, kwargs=None, *, has_out_variant: bool = True) -> None: ...
    def __str__(self) -> str: ...
    def benchmark(self, *args, out): ...
    def to_callable(self): ...
    def hash_key(self): ...
    def output_node(self): ...
    def info_dict(self) -> dict[str, PrimitiveInfoType | list[PrimitiveInfoType]]:
        """Information returned here is logged to the autotune log file when that is enabled."""
    def autoheuristic_id(self): ...

@functools.cache
def get_mm_log_filename() -> str | None: ...
def append_to_log(filename, data) -> None: ...

class DataProcessorChoiceCallerWrapper:
    _wrapped: Incomplete
    _preprocessor: Incomplete
    _postprocessor: Incomplete
    def __init__(self, wrapped, preprocessor, postprocessor) -> None: ...
    def __getattr__(self, name): ...
    def benchmark(self, *args, out) -> float: ...
    def output_node(self) -> ir.TensorBox: ...
    def __repr__(self) -> str: ...

class DataProcessorTemplateWrapper:
    """
    A wrapper class for a kernel template.

    This class together with `DataProcessorChoiceCallerWrapper` provides a convenient way to
    preprocess and postprocess data before and after using the wrapped template. A typical
    usage is to reorder or filter the input nodes in order to match the expected input of other
    kernel choices like a ATen kernel. A more complicated usage is to prepack the weights.
    See the example from :mod:`cpp_gemm_template` for more details.
    """
    _preprocessor: Incomplete
    _postprocessor: Incomplete
    _wrapped: Incomplete
    def __init__(self, wrapped_template_cls, preprocessor, postprocessor, **kwargs) -> None: ...
    def __getattr__(self, name): ...
    def maybe_append_choice(self, choices, **kwargs): ...
    def generate(self, **kwargs): ...
    def __repr__(self) -> str: ...

class ErrorFromChoice(RuntimeError):
    choice: Incomplete
    def __init__(self, msg, choice: ChoiceCaller, inputs_str) -> None: ...

class NoValidChoicesError(RuntimeError): ...

@functools.cache
def get_num_workers() -> int: ...
def create_inputs_key(input_nodes) -> str: ...
def create_precompile_key(name: str, inputs_key: str, choices: list[ChoiceCaller]) -> str: ...
FeedbackFunction = Callable[[dict[ChoiceCaller, float], str, list[Any], list[ChoiceCaller], Callable[[], dict[ChoiceCaller, float]]], None]

class AlgorithmSelectorCache(PersistentCache):
    """
    A persistent cache for algorithm selection results used in autotuning of GEMMs
    and convolutions.

    This classes includes precompilation and benchmarking of the kernels.

    The cache is keyed by input characteristics (sizes, strides, dtypes, etc.) but
    doesn't depend on the output layout.
    """
    precompile_cache: dict[str, Callable[[], None]]
    feedback_saver_fns: list[FeedbackFunction]
    prescreening_cache: dict[str, OrderedSet[str]]
    def __init__(self, *args, **kwargs) -> None: ...
    def cache_clear(self) -> None: ...
    def __call__(self, name, choices: list[ChoiceCaller], input_nodes, layout, input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None = None, precompilation_timeout_seconds: int = ..., return_multi_template: bool = False): ...
    def make_precompile_fn(self, choices, name: str, inputs_key: str, precompilation_timeout_seconds: int | None = ...) -> Callable[[], None]:
        """
        Returns a function that precompiles the given choices.
        """
    @classmethod
    def get_inputs(cls, choices: Sequence[ChoiceCaller], input_nodes: list[ir.IRNode], layout: ir.Layout, input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None) -> AutotuneArgs:
        """
        Factory method to create AutotuneArgs from a list of ChoiceCallers.
        """
    @classmethod
    def benchmark_choice(cls, choice: ChoiceCaller, autotune_args: AutotuneArgs) -> float: ...
    @classmethod
    def benchmark_choices(cls, choices: Sequence[ChoiceCaller], autotune_args: AutotuneArgs) -> dict[ChoiceCaller, float]: ...
    @classmethod
    def benchmark_in_current_process(cls, choices: Sequence[ChoiceCaller], input_nodes: list[ir.IRNode], layout: ir.Layout, input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None) -> dict[ChoiceCaller, float]: ...
    @classmethod
    def benchmark_in_sub_process(cls, choices: Sequence[ChoiceCaller], input_nodes: list[ir.IRNode], layout: ir.Layout, input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None): ...
    @classmethod
    def make_benchmark_fn(cls, choices: Sequence[ChoiceCaller], input_nodes: list[ir.IRNode], layout: ir.Layout, input_gen_fns: dict[int, Callable[[ir.Buffer], torch.Tensor]] | None): ...
    @staticmethod
    def prescreen_choices(choices: list[ChoiceCaller], name: str, inputs_key: str, prescreen_cache: dict[str, OrderedSet[str]]) -> list[ChoiceCaller]:
        """
        Figure out what choices need to be prescreened before autotuning with runtime
        params.

        Prescreening is a process of reducing the number of autotuning for choices with
        runtime params via a two stage autotuning process. First, we fix a set of runtime
        params (here we use swizzle=2) and run autotuning to get a set of candidates.
        Then, we run autotuning again with the candidates and the full set of runtime
        params.

        Since have the concept of runtime params, we need to differentiate between
        choice's hash_key and choice's kernel_hash_key. The former includes information
        like runtime params, while the latter does not. prescreen_cache, if exists, stores
        the set of hash_key that should win the prescreening.

        Right now, only CUTLASS choices have runtime params.
        """
    @staticmethod
    def prune_choices_postscreen(choices: list[ChoiceCaller], candidate_timings: dict[ChoiceCaller, float], name: str, inputs_key: str, prescreen_cache: dict[str, OrderedSet[str]]) -> list[ChoiceCaller]:
        """
        Prune the choices after prescreening.
        """
    @staticmethod
    def log_results(name: str, input_nodes: list[ir.IRNode], timings: dict[ChoiceCaller, float], elapse: float, precompile_elapse: float, prescreening_elapse: float | None = None): ...
    @staticmethod
    def benchmark_example_value(node):
        """
        Convert an ir.Buffer into a concrete torch.Tensor we can use for
        benchmarking.
        """
    @staticmethod
    def generate_example_value(size, stride, device, dtype, extra_size, allocation_size=None): ...
    @staticmethod
    def key_of(node):
        """
        Extract the pieces of an ir.Buffer that we should invalidate cached
        autotuning results on.
        """
    def add_feedback_saver(self, fn: FeedbackFunction): ...

_ALGORITHM_SELECTOR_CACHE: AlgorithmSelectorCache | None

def autotune_select_algorithm(*args, **kwargs): ...
def add_feedback_saver(fn: FeedbackFunction): ...
def realize_inputs(*args): ...

class SymbolicGridFn:
    '''
    Wrapper around a grid function that allows either int or sympy inputs.

        @SymbolicGridFn
        def grid(x, meta, *, cdiv):
            return cdiv(x, meta["BLOCK_X"])
    '''
    fn: Incomplete
    kwargs_int: Incomplete
    kwargs_sym: Incomplete
    def __init__(self, fn: Callable[..., tuple[Any, Any, Any]]) -> None: ...
    def __call__(self, *args, **kwargs) -> tuple[int, int, int]: ...
    def sympy_call(self, *args, **kwargs): ...
