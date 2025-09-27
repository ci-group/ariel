import torch.fx as fx
from .. import config as config
from ..backends.registry import lookup_backend as lookup_backend, register_debug_backend as register_debug_backend
from ..debug_utils import clone_inputs_retaining_gradness as clone_inputs_retaining_gradness
from _typeshed import Incomplete
from torch._dynamo.backends.registry import CompiledFn as CompiledFn
from torch._dynamo.debug_utils import AccuracyError as AccuracyError, BUCK_CMD_PREFIX as BUCK_CMD_PREFIX, BuckTargetWriter as BuckTargetWriter, InputReader as InputReader, InputWriter as InputWriter, NNModuleToString as NNModuleToString, NopInputReader as NopInputReader, backend_accuracy_fails as backend_accuracy_fails, extra_imports as extra_imports, generate_config_string as generate_config_string, generate_env_vars_string as generate_env_vars_string, helper_for_dump_minify as helper_for_dump_minify, minifier_dir as minifier_dir, run_fwd_maybe_bwd as run_fwd_maybe_bwd, same_two_models as same_two_models
from torch.fx.experimental.symbolic_shapes import fx_placeholder_targets as fx_placeholder_targets
from torch.hub import tqdm as tqdm

log: Incomplete
inductor_config: Incomplete
use_buck: Incomplete

def _accuracy_fails(gm, example_inputs, compiler_fn): ...

class WrapBackendDebug:
    _torchdynamo_orig_callable: Incomplete
    _compiler_name: Incomplete
    __name__: Incomplete
    get_compiler_config: Incomplete
    def __init__(self, unconfigured_compiler_fn, compiler_name: str) -> None: ...
    def __call__(self, gm, example_inputs, **kwargs): ...

def wrap_backend_debug(unconfigured_compiler_fn, compiler_name: str):
    """
    A minifier decorator that wraps the TorchDynamo produced Fx graph modules.
    As opposed to wrap_compiler_debug, this wrapper intercepts at the
    TorchDynamo produced Fx Graph Module. This makes it backend-agnostic to some
    level, e.g., it is useful for minifying issues related to Aot Autograd
    tracing.  If an error is found, we minify and save the minified repro in
    repro.tar.gz.
    """
def generate_dynamo_fx_repro_string(gm, args, compiler_name, check_accuracy: bool = False, *, stable_output: bool = False, save_dir=None, command: str = 'run'):
    """
    Generate a repro string for backend-agnostic minified version.
    """
def dump_backend_repro_as_file(gm, args, compiler_name, check_accuracy: bool = False) -> None:
    """
    Saves the repro to a repro.py file
    """
def dump_backend_state(gm, args, compiler_name, check_accuracy: bool = False):
    """
    Dumps the dynamo graph to repro the issue.
    1) It tries to convert Fx GraphModule to a string. If we can, it writes to a
    repro.py file.
    2) If we can't convert Fx GraphModule to a string, we use to_folder to save
    the module and save a tar file.
    """
def dump_to_minify_after_dynamo(gm, args, compiler_name) -> None: ...
@register_debug_backend
def dynamo_minifier_backend(gm: fx.GraphModule, example_inputs, compiler_name: CompiledFn): ...
@register_debug_backend
def dynamo_accuracy_minifier_backend(gm, example_inputs, compiler_name): ...
def backend_fails(gm, example_inputs, compiler_fn, orig_failure):
    """
    Minifier uses this function to identify if the minified graph module fails
    with the same error.

    One caveat is that minifier can potentially go into a wrong direction when
    the resulting graph module fails for a different reason. To avoid this, we
    save the string for the original exception and check similarity between new
    and old exception. They can be somewhat different in some cases, when the
    exception string depends on the failing node information. So, we have a
    loose similarity metric to guide the minifier path.
    """
def run_load_args(options, mod, load_args): ...
def repro_minify(options, mod, load_args) -> None: ...
def repro_run(options, mod, load_args) -> None: ...
def run_repro(mod, load_args, *, command: str = 'run', accuracy: bool | str = '', save_dir=None, autocast: bool = False, backend: str = 'inductor', **kwargs): ...
