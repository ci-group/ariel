import torch
from _typeshed import Incomplete
from dataclasses import dataclass, field
from torch import fx as fx
from torch._dynamo.output_graph import GraphCompileReason as GraphCompileReason
from torch._dynamo.utils import deepcopy_to_fake_tensor as deepcopy_to_fake_tensor, detect_fake_mode as detect_fake_mode
from torch._logging import trace_structured as trace_structured
from torch.fx.node import Node as Node
from typing import Any

log: Incomplete
ddp_graph_log: Incomplete

def args_str(args): ...

@dataclass
class Bucket:
    size: int = ...
    params: list[str] = field(default_factory=list)
    nodes: list[fx.Node] = field(default_factory=list)
    param_ids: list = field(default_factory=list)
    opcount_increased_to_capture_external_output: int = ...
    paramsize_before_opcount_increase: int = ...

def bucket_has_external_output(bucket: Bucket) -> bool: ...
def pretty_print_buckets(buckets: list[Bucket], bucket_bytes_cap: int): ...
def has_higher_order_op(gm): ...
def propagate_metadata(orig_gm, split_gm) -> None: ...
def propagate_dynamo_source(orig_gm, split_gm) -> None: ...

class SubmodCompiler(torch.fx.interpreter.Interpreter):
    compiler: Incomplete
    fake_mode: Incomplete
    def __init__(self, module, compiler, fake_mode) -> None: ...
    submod: Incomplete
    unwrap_singleton_tuple: Incomplete
    def compile_submod(self, input_mod, args, kwargs):
        """
        Compile the submodule,
        using a wrapper to make sure its output is always a tuple,
        which is required by AotAutograd based compilers
        """
    tc: Incomplete
    def run_node(self, n: Node) -> Any: ...

class DDPOptimizer:
    """Note [DDPOptimizer]
    DDPOptimizer applies when dynamo compiles models wrapped in DistributedDataParallel (DDP),
    breaking the dynamo graph into chunks to compile separately, with the breaks aligning to
    the boundaries of gradient-allreduce buckets chosen by DDP.

    Background/Motivation
     - DDP uses allreduce collectives to synchronize partial gradients computed on different workers
     - DDP groups gradient allreduces into 'buckets' to optimize communication efficiency of all-reduce
     - Parameters grouped into buckets are assumed to be adjacent in time, so they become ready
       at around the same time during backward and thus can share the same allreduce efficiently
     - Allreduces must overlap with backward compute for optimal training performance
     - DDP schedules allreduces using 'hooks' fired from the c++ autograd engine in pytorch, which
       operates when individual grads become 'ready'
     - Dynamo+AOTAutograd produces a single fused graph that runs 'atomically' from the perspective of the
       autograd engine, such that all gradients become 'ready' at the same time.  Hooks fire after the whole
       fused backward function executes, preventing any overlap of compute and communication

    Algorithm
     - DDPOptimizer starts off with an FX graph traced by dynamo which represents forward.  It can traverse
       this graph in reverse order to determine the true order that gradients will become ready during backward.
     - Parameter sizes are counted in reverse order, up to a bucket size limit, at which point a new bucket is started
       and a graph break introduced
     - Each of the subgraphs is compiled by the compiler provided to dynamo by the user, and then fused back together
       into an outer module that is returned to the user

    Notes
     - It would be better to enforce (by adding an API to DDP) that the bucket splits chosen here are used by DDP,
       and that DDP does not need to detect or optimize bucket order by observing execution at runtime, as it does
       in eager.
     - If Dynamo can't capture a whole graph for the portion of the model wrapped by DDP, this algorithm will currently
       produce splits that do not necessarily align with the buckets used by DDP.  This should result in performance
       degradation approaching the baseline case where graph-splits are not used, but not worse.
     - If the backend compiler fails to compile a single subgraph, it will execute eagerly despite the rest of the
       subgraphs being compiled
     - DDP has a 'parameters_and_buffers_to_ignore' field, which DDPOptimizer attempts to honor by reading markers
       left by DDP on individual parameters.  In cases where other transformations, such as reparameterization, are
       also used, the ignore markers could be lost.  If DDPOptimizer fails to ignore a parameter ignored by DDP,
       it is not catastrophic but could impact performance by choosing sub-optimal bucket splits.
     - DDPOptimizer always ignores all buffers, regardless of their ignore flag, since buffers do not require gradients,
       and therefore aren't allreduced by DDP.  (They are broadcast during forward, but this is not covered by
       DDPOptimizer)

    Debugging
     - Generally, it is easiest to debug DDPOptimizer in a single process program, using pdb.
     - In many cases, the log messages are helpful (they show bucket size assignments)-
       just set TORCH_LOGS env to include any of 'dynamo', 'distributed', or 'dist_ddp'.
     - See `benchmarks/dynamo/distributed.py` for a simple harness that will run a toy model or a torchbench model
       in a single process (or with torchrun, in multiple processes)

    Args:
        bucket_bytes_cap (int): Controls the size of buckets, in bytes, used to determine graphbreaks.  Should be
            set to match the equivalent parameter on the original DDP module.

        backend_compile_fn (callable): A dynamo compiler function, to be invoked to compile each subgraph.

        first_bucket_cap (int): Controls the size of the first bucket.  Should match DDP's first bucket cap.  DDP
            special-cases the first bucket size since it is sometimes optimal to start a small allreduce early.

    """
    first_bucket_cap: Incomplete
    bucket_bytes_cap: Incomplete
    backend_compile_fn: Incomplete
    def __init__(self, bucket_bytes_cap: int, backend_compile_fn, first_bucket_cap: int | None = None) -> None: ...
    def _ignore_parameter(self, parameter): ...
    def add_param(self, bucket, param, name) -> None: ...
    def add_module_params_to_bucket(self, mod, bucket, processed_modules, prefix) -> None: ...
    def add_param_args(self, bucket, node) -> None: ...
    buckets: Incomplete
    def compile_fn(self, gm: fx.GraphModule, example_inputs: list[torch.Tensor]):
        """
        Implements graph splitting, first determining a set of of buckets by counting
        parameter sizes in reverse graph order, then invoking the user/backend compiler
        to compile each subgraph. Finally, stiches compiled graphs into one graphmodule
        and returns its callable.
        """
