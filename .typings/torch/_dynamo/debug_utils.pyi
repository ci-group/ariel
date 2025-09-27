import functools
import torch
from . import config as config
from .utils import clone_inputs as clone_inputs, get_debug_dir as get_debug_dir
from _typeshed import Incomplete
from torch import Tensor as Tensor
from torch._dynamo.testing import rand_strided as rand_strided
from torch._prims_common import is_float_dtype as is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef as StorageWeakRef
from torch.utils._content_store import ContentStoreReader as ContentStoreReader, ContentStoreWriter as ContentStoreWriter
from typing import Any, Callable, TypeVar

log: Incomplete
T = TypeVar('T')
inductor_config: Incomplete
use_buck: Incomplete
extra_deps: Incomplete
extra_imports: str
cur_target: Incomplete
BUCK_CMD_PREFIX: Incomplete

class BuckTargetWriter:
    target: Incomplete
    path: Incomplete
    cmd_line_path: Incomplete
    def __init__(self, filename) -> None: ...
    def build(self): ...
    def write(self, print_msg: bool = True): ...

def minifier_dir(): ...

MAX_CONSTANT_NUMEL_INLINE: int

class NNModuleToString:
    safe_reprs: Incomplete
    @staticmethod
    def can_convert_to_string(gm): ...
    @staticmethod
    def convert(gm): ...

@functools.cache
def _cuda_system_info_comment(): ...
def generate_env_vars_string(*, stable_output: bool = False):
    """
    Generate a string configuration for environment variables related to Dynamo, Inductor, and Triton.
    """
def generate_config_string(*, stable_output: bool = False): ...
def get_minifier_repro_path(): ...
def helper_for_dump_minify(contents) -> None: ...

class AccuracyError(Exception): ...

def clone_inputs_retaining_gradness(example_inputs):
    """
    This clone inputs is different from utils clone_input. In case of minifier,
    all the tensors are leaf tensors while creating a new graph. So, we set the
    requires_grad field w/o checking the leafness of the tensor.
    """
def run_fwd_maybe_bwd(gm, args, only_fwd: bool = False, disable_clone: bool = False):
    """
    Runs a forward and possibly backward iteration for a given mod and args.

    When disable_clone is True, we will use args as-is without cloning.
    This is higher fidelity but we may destroy the args in the process.
    """
def same_two_models(gm, opt_gm, example_inputs, only_fwd: bool = False, *, require_fp64: bool = False, ignore_non_fp: bool = False):
    """
    Check two models have same accuracy.

    require_fp64: if True, raise an error if we unable to calculate the fp64 reference
    ignore_non_fp: if True, do not compare outputs which are not floating point.  This
        is mostly useful for the minifier (which wants to avoid quantizing floating point
        error into integer/boolean error)
    """
def cast_dtype_args_to_fp64(model): ...
def cast_to(dtype, model, inputs): ...
def cast_to_fp64(model, inputs): ...
def backend_accuracy_fails(gm, example_inputs, compiler_fn, only_fwd: bool = False, *, require_fp64: bool = False, ignore_non_fp: bool = False): ...
def _stride_or_default(stride: torch._prims_common.StrideType | None, *, shape: torch._prims_common.ShapeType) -> torch._prims_common.StrideType: ...
def _mk_defaulter(d: T) -> Callable[[T | None], T]: ...

_dtype_or_default: Incomplete
_device_or_default: Incomplete
_storage_offset_or_default: Incomplete
_requires_grad_or_default: Incomplete
_is_leaf_or_default: Incomplete

class NopInputReader:
    total: int
    def __init__(self) -> None: ...
    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None) -> None: ...
    def tensor(self, *args, **kwargs) -> None: ...
    def symint(self, *args, **kwargs) -> None: ...

class InputReader:
    store: Incomplete
    args: Incomplete
    pbar: Incomplete
    def __init__(self, save_dir=None, *, pbar=None) -> None: ...
    def storage(self, storage_hash, nbytes, *, device=None, dtype_hint=None): ...
    def tensor(self, storage, shape, stride=None, *, storage_offset=None, dtype=None, requires_grad=None, is_leaf=None, **metadata): ...
    def symint(self, val): ...

class InputWriter:
    _lines: Incomplete
    storage_counter: Incomplete
    save_dir: Incomplete
    store: Incomplete
    seen_storages: Incomplete
    def __init__(self, save_dir, *, stable_hash: bool = False) -> None: ...
    def lines(self): ...
    def storage(self, untyped_storage, *, dtype_hint=None, device_hint=None) -> str: ...
    def tensor(self, name, t) -> None: ...
    def unsupported(self, name, arg) -> None: ...
    def const(self, name) -> None: ...
    def symint(self, name, val) -> None: ...

def aot_graph_input_parser(func: Callable[[list[Tensor]], list[Tensor]], device: str = 'cuda', sym_shapes: dict[str, int] | None = None, default_sym_shape: int | None = None) -> dict[str, Any]:
    '''
    Takes in a function which has been printed with print_readable() and constructs kwargs to run it.

    Handles Tensor inputs, Symints, and a graph module which might have tensor constants.

    Consider a function `forward` defined as follows:

    def forward(self, primals_1: "f32[1001, 6]", primals_2: "f32[s0]", primals_3: "Sym(s0)",):
        _tensor_constant0: "i64[4190]" = self._tensor_constant0
        # Further implementation

    kwargs = aot_graph_input_parser(forward)
    forward(**kwargs)
    '''
def profile_to_file(filename: str) -> Callable[[T], T]:
    """
    Decorator to cProfile a given function and save the result to disk on process exit.

    Args:
        filename: filename to save profile to
    """
