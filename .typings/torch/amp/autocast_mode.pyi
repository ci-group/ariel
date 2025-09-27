from _typeshed import Incomplete
from torch.types import _dtype
from typing import Any

__all__ = ['autocast_decorator', 'autocast', 'is_autocast_available', 'custom_fwd', 'custom_bwd']

def is_autocast_available(device_type: str) -> bool:
    """
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu', 'mtia', 'maia', 'xpu', and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
def autocast_decorator(autocast_instance, func): ...

class autocast:
    '''
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.
    See the :ref:`Autocast Op Reference<autocast-op-reference>` for details.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with torch.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage (along with gradient scaling)
    in more complex scenarios (e.g., gradient penalty, multiple models/losses, custom autograd functions).

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...
            @torch.autocast(device_type="cuda")
            def forward(self, input):
                ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).
    If a Tensor from the autocast region is already ``float32``, the cast is a no-op,
    and incurs no additional overhead.
    CUDA Example::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with torch.autocast(device_type="cuda"):
            # torch.mm is on autocast\'s list of ops that should run in float16.
            # Inputs are float32, but the op runs in float16 and produces float16 output.
            # No manual casts are required.
            e_float16 = torch.mm(a_float32, b_float32)
            # Also handles mixed input types
            f_float16 = torch.mm(d_float32, e_float16)

        # After exiting autocast, calls f_float16.float() to use with d_float32
        g_float32 = torch.mm(d_float32, f_float16.float())

    CPU Training Example::

        # Creates model and optimizer in default precision
        model = Net()
        optimizer = optim.SGD(model.parameters(), ...)

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()

                # Runs the forward pass with autocasting.
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    output = model(input)
                    loss = loss_fn(output, target)

                loss.backward()
                optimizer.step()


    CPU Inference Example::

        # Creates model in default precision
        model = Net().eval()

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            for input in data:
                # Runs the forward pass with autocasting.
                output = model(input)

    CPU Inference Example with Jit Trace::

        class TestModel(nn.Module):
            def __init__(self, input_size, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(input_size, num_classes)
            def forward(self, x):
                return self.fc1(x)

        input_size = 2
        num_classes = 2
        model = TestModel(input_size, num_classes).eval()

        # For now, we suggest to disable the Jit Autocast Pass,
        # As the issue: https://github.com/pytorch/pytorch/issues/75956
        torch._C._jit_set_autocast_mode(False)

        with torch.cpu.amp.autocast(cache_enabled=False):
            model = torch.jit.trace(model, torch.randn(1, input_size))
        model = torch.jit.freeze(model)
        # Models Run
        for _ in range(3):
            model(torch.randn(1, input_size))

    Type mismatch errors *in* an autocast-enabled region are a bug; if this is what you observe,
    please file an issue.

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.  Disabling autocast gives you explicit control over
    the execution type.  In the subregion, inputs from the surrounding region
    should be cast to ``dtype`` before use::

        # Creates some tensors in default dtype (here assumed to be float32)
        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")
        c_float32 = torch.rand((8, 8), device="cuda")
        d_float32 = torch.rand((8, 8), device="cuda")

        with torch.autocast(device_type="cuda"):
            e_float16 = torch.mm(a_float32, b_float32)
            with torch.autocast(device_type="cuda", enabled=False):
                # Calls e_float16.float() to ensure float32 execution
                # (necessary because e_float16 was created in an autocasted region)
                f_float32 = torch.mm(c_float32, e_float16.float())

            # No manual casts are required when re-entering the autocast-enabled region.
            # torch.mm again runs in float16 and produces float16 output, regardless of input types.
            g_float16 = torch.mm(d_float32, f_float32)

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.  This affects :class:`torch.nn.DataParallel` and
    :class:`torch.nn.parallel.DistributedDataParallel` when used with more than one GPU per process
    (see :ref:`Working with Multiple GPUs<amp-multigpu>`).

    Args:
        device_type(str, required):  Device type to use. Possible values are: \'cuda\', \'cpu\', \'mtia\', \'maia\', \'xpu\', and \'hpu\'.
                                     The type is the same as the `type` attribute of a :class:`torch.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(torch_dtype, optional):  Data type for ops run in autocast. It uses the default value
            (``torch.float16`` for CUDA and ``torch.bfloat16`` for CPU), given by
            :func:`~torch.get_autocast_dtype`, if :attr:`dtype` is ``None``.
            Default: ``None``
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    '''
    _enabled: Incomplete
    device: Incomplete
    fast_dtype: Incomplete
    custom_backend_name: Incomplete
    custom_device_mod: Incomplete
    _cache_enabled: Incomplete
    def __init__(self, device_type: str, dtype: _dtype | None = None, enabled: bool = True, cache_enabled: bool | None = None) -> None: ...
    prev_cache_enabled: Incomplete
    prev: Incomplete
    prev_fastdtype: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any): ...
    def __call__(self, func): ...

def custom_fwd(fwd=None, *, device_type: str, cast_inputs: _dtype | None = None):
    """
    Create a helper decorator for ``forward`` methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`torch.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`torch.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.
    See the :ref:`example page<amp-custom-examples>` for more detail.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu', 'mtia', 'maia', 'xpu' and so on.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
