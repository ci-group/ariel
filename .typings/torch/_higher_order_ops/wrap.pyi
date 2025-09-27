from _typeshed import Incomplete
from torch._logging import warning_once as warning_once
from torch._ops import HigherOrderOperator as HigherOrderOperator
from torch.types import _dtype as _dtype

log: Incomplete
uid: Incomplete

class Wrap(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, func, *args, **kwargs): ...

wrap: Incomplete

class WrapWithSetGradEnabled(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, enable_grad, wrapped_func, *args, **kwargs): ...

wrap_with_set_grad_enabled: Incomplete

class WrapWithAutocast(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, device_type: str, dtype: _dtype | None, enabled: bool, cache_enabled: bool | None, wrapped_func, *args, **kwargs): ...

wrap_with_autocast: Incomplete

class DynamoBypassingWrapper(HigherOrderOperator):
    def __init__(self) -> None: ...
    def __call__(self, wrapper_fn_or_key, inner_fn, *args, **kwargs): ...

dynamo_bypassing_wrapper: Incomplete

class WrapActivationCheckpoint(HigherOrderOperator):
    """
    This operator is used to wrap torch.utils.checkpoint. This avoids
    TorchDynamo to look into saved tensor hooks and directly passes the control
    to AOT Autograd, which is ok with tracing saved tensor hooks. As a result of
    AOT tracing torch.utils.checkpoint code, we have a backward graph with
    recomputed forward nodes.

    However, we might deprecate this operator soon. The difficulty arises in the
    functionalization of rng ops. Today, there are two different
    functionalization of rng ops - one at AOT autograd and other at Inductor.
    And they are difficult to map to each other. The rng states also complicate
    pattern matching in Inductor. Due to the ease of implementation, we are
    currently inclined towards functionalization at Inductor level, which means
    that duplication/recomputation is done as a compiler pass in the
    partitioners. See TagActivationCheckpoint for more information.
    """
    def __init__(self) -> None: ...
    def __call__(self, function, *args, **kwargs): ...

wrap_activation_checkpoint: Incomplete

class TagActivationCheckpoint(HigherOrderOperator):
    '''
    This operator is supposed to be used only with torch.compile stack. This
    accepts a Fx graph module which needs to be checkpointed. This operator adds
    "recomputable" tag to the nodes of the Fx graph that should be recomputed.

    The goal is to:
    1. Avoid using Dynamo to trace through saved tensor hooks.
    2. For selective checkpointing case, let AOTAutograd trace through
       saved tensor hooks but has special logic with TorchDispatchMode to override
       the usual saved_tensor_hooks fn logic in order to tag the nodes.
    3. Rely on the partitioners to actually duplicate the nodes.
    This sits well in the torch.compile stack, because by the time graph
    reaches partitioner, inductor has already run its functionalization of rng
    ops (by setting fixed seed for each random op, see `replace_random_passes`).
    Therefore, the duplication of nodes, by design, respects the rng states in
    the forward and recomputed forward in backward.
    '''
    def __init__(self) -> None: ...
    @staticmethod
    def divide_kwargs(kwargs):
        """
        checkpoint fn can have mixed kwargs between checkpointed fn and
        checkpoint fn itself. For example
        >> def gn(x, y, z=None):
        >>     a = torch.matmul(x, y)
        >>     if z is not None:
        >>         return torch.matmul(a, z)
        >>     return a
        >> def fn(x, y, z):
        >>     return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))
        In the above case, z belongs to checkpointed function gn, but
        use_reentrant belongs to the checkpoint function. This function splits
        the kwargs into checkpoint_kwargs and gmod_kwargs (or
        checkpointed_fn_kwargs).
        We do sorting to ensure same graph from run to run for better
        debuggability. It is not required for correctness.
        """
    def tag_nodes(self, gmod, is_sac): ...
    def __call__(self, gmod, *args, **kwargs): ...

tag_activation_checkpoint: Incomplete
