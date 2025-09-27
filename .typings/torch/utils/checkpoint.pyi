from typing import *
import contextlib
import enum
import torch
import weakref
from _typeshed import Incomplete
from torch.utils._python_dispatch import TorchDispatchMode
from typing import Any
from weakref import ReferenceType

__all__ = ['checkpoint', 'checkpoint_sequential', 'CheckpointError', 'CheckpointFunction', 'check_backward_validity', 'detach_variable', 'get_device_states', 'set_device_states', 'noop_context_fn', 'set_checkpoint_early_stop', 'DefaultDeviceType', 'set_checkpoint_debug_enabled', 'CheckpointPolicy', 'SelectiveCheckpointContext', 'create_selective_checkpoint_contexts', 'SAC_IGNORED_OPS']

@contextlib.contextmanager
def set_checkpoint_debug_enabled(enabled: Optional[bool]):
    """
    Context manager that sets whether checkpoint should print additional debug
    information when running. See the ``debug`` flag for
    :func:`~torch.utils.checkpoint.checkpoint` for more information. Note that
    when set, this context manager overrides the value of ``debug`` passed to
    checkpoint. To defer to the local setting, pass ``None`` to this context.

    Args:
        enabled (bool): Whether checkpoint should print debug information.
            Default is 'None'.
    """
def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]: ...
def check_backward_validity(inputs: Iterable[Any]) -> None: ...

class DefaultDeviceType:
    """
    A class that manages the default device type for checkpointing.

    If no non-CPU tensors are present, the default device type will
    be used. The default value is 'cuda'. The device type is used in
    the checkpointing process when determining which device states
    to save and restore for recomputation.
    """
    _default_device_type: str
    @staticmethod
    def set_device_type(device: str = 'cuda'):
        """
        Set the default device type for checkpointing.

        Args:
            device (str): The device type to be set as default. Default is 'cuda'.
        """
    @staticmethod
    def get_device_type() -> str:
        """
        Get the current default device type for checkpointing.

        Returns:
            str: The current default device type.
        """

def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]: ...
def set_device_states(devices, states, *, device_type=None) -> None:
    """Sets random number generator states for the specified devices.

    Args:
        devices: Device ids to set states for.
        states: States to set.
        device_type: ``device_type`` of the devices to set states for. Default
            is the device returned by a call to ``DefaultDeviceType.get_device_type()``,
            which is ``cuda`` if not changed by calling ``DefaultDeviceType::set_device_type()``.
    """

class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args): ...
    @staticmethod
    def backward(ctx, *args): ...

def noop_context_fn(): ...
@torch._disable_dynamo
def checkpoint(function, *args, use_reentrant: Optional[bool] = None, context_fn: Callable[[], Tuple[ContextManager, ContextManager]] = ..., determinism_check: str = ..., debug: bool = False, **kwargs):
    '''Checkpoint a model or part of the model.

    Activation checkpointing is a technique that trades compute for memory.
    Instead of keeping tensors needed for backward alive until they are used in
    gradient computation during backward, forward computation in checkpointed
    regions omits saving tensors for backward and recomputes them during the
    backward pass. Activation checkpointing can be applied to any part of a
    model.

    There are currently two checkpointing implementations available, determined
    by the :attr:`use_reentrant` parameter. It is recommended that you use
    ``use_reentrant=False``. Please refer the note below for a discussion of
    their differences.

    .. warning::

        If the :attr:`function` invocation during the backward pass differs
        from the forward pass, e.g., due to a global variable, the checkpointed
        version may not be equivalent, potentially causing an
        error being raised or leading to silently incorrect gradients.

    .. warning::

        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True`` variant, please refer to the
        note below for important considerations and potential limitations.

    .. note::

        The reentrant variant of checkpoint (``use_reentrant=True``) and
        the non-reentrant variant of checkpoint (``use_reentrant=False``)
        differ in the following ways:

        * Non-reentrant checkpoint stops recomputation as soon as all needed
          intermediate activations have been recomputed. This feature is enabled
          by default, but can be disabled with :func:`set_checkpoint_early_stop`.
          Reentrant checkpoint always recomputes :attr:`function` in its
          entirety during the backward pass.

        * The reentrant variant does not record the autograd graph during the
          forward pass, as it runs with the forward pass under
          :func:`torch.no_grad`. The non-reentrant version does record the
          autograd graph, allowing one to perform backward on the graph within
          checkpointed regions.

        * The reentrant checkpoint only supports the
          :func:`torch.autograd.backward` API for the backward pass without its
          `inputs` argument, while the non-reentrant version supports all ways
          of performing the backward pass.

        * At least one input and output must have ``requires_grad=True`` for the
          reentrant variant. If this condition is unmet, the checkpointed part
          of the model will not have gradients. The non-reentrant version does
          not have this requirement.

        * The reentrant version does not consider tensors in nested structures
          (e.g., custom objects, lists, dicts, etc) as participating in
          autograd, while the non-reentrant version does.

        * The reentrant checkpoint does not support checkpointed regions with
          detached tensors from the computational graph, whereas the
          non-reentrant version does. For the reentrant variant, if the
          checkpointed segment contains tensors detached using ``detach()`` or
          with :func:`torch.no_grad`, the backward pass will raise an error.
          This is because ``checkpoint`` makes all the outputs require gradients
          and this causes issues when a tensor is defined to have no gradient in
          the model. To avoid this, detach the tensors outside of the
          ``checkpoint`` function.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint. Note that under torch.compile,
            this flag doesn\'t take effect and we always preserve RNG state.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.
        context_fn(Callable, optional): A callable returning a tuple of two
            context managers. The function and its recomputation will be run
            under the first and second context managers respectively.
            This argument is only supported if ``use_reentrant=False``.
        determinism_check(str, optional): A string specifying the determinism
            check to perform. By default it is set to ``"default"`` which
            compares the shapes, dtypes, and devices of the recomputed tensors
            against those the saved tensors. To turn off this check, specify
            ``"none"``. Currently these are the only two supported values.
            Please open an issue if you would like to see more determinism
            checks. This argument is only supported if ``use_reentrant=False``,
            if ``use_reentrant=True``, the determinism check is always disabled.
        debug(bool, optional): If ``True``, error messages will also include
            a trace of the operators ran during the original forward computation
            as well as the recomputation. This argument is only supported if
            ``use_reentrant=False``.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    '''
def checkpoint_sequential(functions, segments, input, use_reentrant=None, **kwargs):
    '''Checkpoint a sequential model to save memory.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will not store
    the intermediate activations. The inputs of each checkpointed segment will
    be saved for re-running the segment in the backward pass.

    .. warning::
        The ``use_reentrant`` parameter should be passed explicitly. In version
        2.4 we will raise an exception if ``use_reentrant`` is not passed.
        If you are using the ``use_reentrant=True` variant, please see
        :func:`~torch.utils.checkpoint.checkpoint` for
        the important considerations and limitations of this variant. It is
        recommended that you use ``use_reentrant=False``.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional):  Omit stashing and restoring
            the RNG state during each checkpoint.
            Default: ``True``
        use_reentrant(bool):
            specify whether to use the activation checkpoint variant that
            requires reentrant autograd. This parameter should be passed
            explicitly. In version 2.5 we will raise an exception if
            ``use_reentrant`` is not passed. If ``use_reentrant=False``,
            ``checkpoint`` will use an implementation that does not require
            reentrant autograd. This allows ``checkpoint`` to support additional
            functionality, such as working as expected with
            ``torch.autograd.grad`` and support for keyword arguments input into
            the checkpointed function.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> # xdoctest: +SKIP("stub")
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    '''
@contextlib.contextmanager
def set_checkpoint_early_stop(enable: bool):
    '''Context manager that sets whether checkpoint should stop recomputation early.

    By default, non-reentrant checkpoint stops recomputation as soon as it
    has computed all needed Tensors. This context manager can be used to disable
    that feature if it is problematic for your specific application.

    This context manager only needs to be active when forward is run. It does
    not need to be active during backward.

    Example::

    >>> # xdoctest: +SKIP(failing)
    >>> message = "saved tensors default hooks are disabled"
    >>> with set_checkpoint_early_stop(False):
    ...     # Any checkpoint under this context manager will respect this
    ...     # context manager, even if its backward is performed outside.
    ...     out = checkpoint(fn, inputs)
    ...
    >>> out.backward()
    '''

class _Handle: ...

class _Holder:
    handles: Dict[int, Optional[_Handle]]
    def __init__(self) -> None: ...

class _NoopSaveInputs(torch.autograd.Function):
    @staticmethod
    def forward(*args): ...
    @staticmethod
    def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None: ...
    @staticmethod
    def backward(ctx, *grad_outputs) -> None: ...

class _CheckpointFrame:
    recompute_fn: Incomplete
    input_saver: Incomplete
    weak_holders: List[ReferenceType]
    recomputed: DefaultDict[int, weakref.WeakKeyDictionary[_Handle, torch.Tensor]]
    recomp_counter: DefaultDict[int, int]
    is_recomputed: DefaultDict[int, bool]
    early_stop: Incomplete
    metadata_fn: Incomplete
    unpack_error_cb: Incomplete
    x_metadatas: Incomplete
    forward_completed: bool
    ignore_saved_mismatch: bool
    def __init__(self, recompute_fn, early_stop, unpack_error_cb, metadata_fn) -> None: ...
    def check_recomputed_tensors_match(self, gid) -> None: ...

class CheckpointError(RuntimeError): ...
class _StopRecomputationError(Exception): ...

class _recomputation_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, target_frame_ref: ReferenceType, gid: int) -> None: ...

class _checkpoint_hook(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, frame) -> None: ...

class _VersionWrapper:
    val: Union[torch.Tensor, Any]
    version: Optional[int]
    def __init__(self, val) -> None: ...
    def get_val(self, allow_cache_entry_mutation): ...

class SelectiveCheckpointContext:
    """
    Context passed to policy function during selective checkpointing.

    This class is used to pass relevant metadata to the policy function during
    selective checkpointing. The metadata includes whether the current invocation
    of the policy function is during recomputation or not.

    Example:
        >>> # xdoctest: +SKIP(stub)
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    print(ctx.is_recompute)
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
    is_recompute: Incomplete
    def __init__(self, *, is_recompute) -> None: ...

class CheckpointPolicy(enum.Enum):
    """
    Enum for specifying the policy for checkpointing during backpropagation.

    The following policies are supported:

    - ``{MUST,PREFER}_SAVE``: The operation's output will be saved during the forward
      pass and will not be recomputed during the backward pass
    - ``{MUST,PREFER}_RECOMPUTE``: The operation's output will not be saved during the
      forward pass and will be recomputed during the backward pass

    Use ``MUST_*`` over ``PREFER_*`` to indicate that the policy should not be overridden
    by other subsystems like `torch.compile`.

    .. note::
        A policy function that always returns ``PREFER_RECOMPUTE`` is
        equivalent to vanilla checkpointing.

        A policy function that returns ``PREFER_SAVE`` every op is
        NOT equivalent to not using checkpointing. Using such a policy would
        save additional tensors not limited to ones that are actually needed for
        gradient computation.
    """
    MUST_SAVE = 0
    PREFER_SAVE = 1
    MUST_RECOMPUTE = 2
    PREFER_RECOMPUTE = 3

SAC_IGNORED_OPS: Incomplete

class _CachingTorchDispatchMode(TorchDispatchMode):
    policy_fn: Incomplete
    storage: Incomplete
    def __init__(self, policy_fn, storage) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...

class _CachedTorchDispatchMode(TorchDispatchMode):
    policy_fn: Incomplete
    storage: Incomplete
    allow_cache_entry_mutation: Incomplete
    def __init__(self, policy_fn, storage, allow_cache_entry_mutation) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...

def create_selective_checkpoint_contexts(policy_fn_or_list, allow_cache_entry_mutation: bool = False):
    """
    Helper to avoid recomputing certain ops during activation checkpointing.

    Use this with `torch.utils.checkpoint.checkpoint` to control which
    operations are recomputed during the backward pass.

    Args:
        policy_fn_or_list (Callable or List):
          - If a policy function is provided, it should accept a
            :class:`SelectiveCheckpointContext`, the :class:`OpOverload`, args and
            kwargs to the op, and return a :class:`CheckpointPolicy` enum value
            indicating whether the execution of the op should be recomputed or not.
          - If a list of operations is provided, it is equivalent to a policy
            returning `CheckpointPolicy.MUST_SAVE` for the specified
            operations and `CheckpointPolicy.PREFER_RECOMPUTE` for all other
            operations.
        allow_cache_entry_mutation (bool, optional): By default, an error is
            raised if any tensors cached by selective activation checkpoint are
            mutated in order to ensure correctness. If set to `True`, this check
            is disabled.
    Returns:
        A tuple of two context managers.

    Example:
        >>> # xdoctest: +REQUIRES(LINUX)
        >>> import functools
        >>>
        >>> x = torch.rand(10, 10, requires_grad=True)
        >>> y = torch.rand(10, 10, requires_grad=True)
        >>>
        >>> ops_to_save = [
        >>>    torch.ops.aten.mm.default,
        >>> ]
        >>>
        >>> def policy_fn(ctx, op, *args, **kwargs):
        >>>    if op in ops_to_save:
        >>>        return CheckpointPolicy.MUST_SAVE
        >>>    else:
        >>>        return CheckpointPolicy.PREFER_RECOMPUTE
        >>>
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)
        >>>
        >>> # or equivalently
        >>> context_fn = functools.partial(create_selective_checkpoint_contexts, ops_to_save)
        >>>
        >>> def fn(x, y):
        >>>     return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y
        >>>
        >>> out = torch.utils.checkpoint.checkpoint(
        >>>     fn, x, y,
        >>>     use_reentrant=False,
        >>>     context_fn=context_fn,
        >>> )
    """
