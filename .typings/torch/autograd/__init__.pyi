import torch
from .anomaly_mode import detect_anomaly as detect_anomaly, set_detect_anomaly as set_detect_anomaly
from .function import Function as Function, NestedIOFunction as NestedIOFunction
from .grad_mode import enable_grad as enable_grad, inference_mode as inference_mode, no_grad as no_grad, set_grad_enabled as set_grad_enabled, set_multithreading_enabled as set_multithreading_enabled
from .gradcheck import gradcheck as gradcheck, gradgradcheck as gradgradcheck
from .variable import Variable as Variable
from collections.abc import Sequence
from torch.types import _TensorOrTensors, _TensorOrTensorsOrGradEdge, _size

__all__ = ['Variable', 'Function', 'backward', 'grad_mode', 'NestedIOFunction', 'detect_anomaly', 'enable_grad', 'grad', 'gradcheck', 'gradgradcheck', 'inference_mode', 'no_grad', 'set_detect_anomaly', 'set_grad_enabled', 'set_multithreading_enabled', 'variable']

_OptionalTensor = torch.Tensor | None
_ShapeorNestedShape = _size | Sequence[_size] | torch.Tensor

def backward(tensors: _TensorOrTensorsOrGradEdge, grad_tensors: _TensorOrTensors | None = None, retain_graph: bool | None = None, create_graph: bool = False, grad_variables: _TensorOrTensors | None = None, inputs: _TensorOrTensorsOrGradEdge | None = None) -> None:
    '''Compute the sum of gradients of given tensors with respect to graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don\'t need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    ``.grad`` attributes or set them to ``None`` before calling it.
    See :ref:`Default gradient layouts<default-grad-layouts>`
    for details on the memory layout of accumulated gradients.

    .. note::
        Using this method with ``create_graph=True`` will create a reference cycle
        between the parameter and its gradient which can cause a memory leak.
        We recommend using ``autograd.grad`` when creating the graph to avoid this.
        If you have to use this function, make sure to reset the ``.grad`` fields of your
        parameters to ``None`` after use to break the cycle and avoid the leak.

    .. note::

        If you run any forward ops, create ``grad_tensors``, and/or call ``backward``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        When ``inputs`` are provided and a given input is not a leaf,
        the current implementation will call its grad_fn (even though it is not strictly needed to get this gradients).
        It is an implementation detail on which the user should not rely.
        See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

    Args:
        tensors (Sequence[Tensor] or Tensor or Sequence[GradientEdge] or GradientEdge): Tensors of which
            the derivative will be computed.
        grad_tensors (Sequence[Tensor or None] or Tensor, optional): The "vector" in
            the Jacobian-vector product, usually gradients w.r.t. each element of
            corresponding tensors. None values can be specified for scalar Tensors or
            ones that don\'t require grad. If a None value would be acceptable for all
            grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
        inputs (Sequence[Tensor] or Tensor or Sequence[GradientEdge], optional): Inputs w.r.t. which the gradient
            be will accumulated into ``.grad``. All other Tensors will be ignored. If
            not provided, the gradient is accumulated into all the leaf Tensors that
            were used to compute the :attr:`tensors`.
    '''
def grad(outputs: _TensorOrTensorsOrGradEdge, inputs: _TensorOrTensorsOrGradEdge, grad_outputs: _TensorOrTensors | None = None, retain_graph: bool | None = None, create_graph: bool = False, only_inputs: bool = True, allow_unused: bool | None = None, is_grads_batched: bool = False, materialize_grads: bool = False) -> tuple[torch.Tensor, ...]:
    '''Compute and return the sum of gradients of outputs with respect to the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in vector-Jacobian product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn\'t require_grad,
    then the gradient can be ``None``).

    .. note::

        If you run any forward ops, create ``grad_outputs``, and/or call ``grad``
        in a user-specified CUDA stream context, see
        :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

    .. note::

        ``only_inputs`` argument is deprecated and is ignored now (defaults to ``True``).
        To accumulate gradient for other parts of the graph, please use
        ``torch.autograd.backward``.

    Args:
        outputs (sequence of Tensor or GradientEdge): outputs of the differentiated function.
        inputs (sequence of Tensor or GradientEdge): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the vector-Jacobian product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don\'t require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (Optional[bool], optional): If ``False``, specifying inputs
            that were not used when computing outputs (and therefore their grad is
            always zero) is an error. Defaults to the value of ``materialize_grads``.
        is_grads_batched (bool, optional): If ``True``, the first dimension of each
            tensor in ``grad_outputs`` will be interpreted as the batch dimension.
            Instead of computing a single vector-Jacobian product, we compute a
            batch of vector-Jacobian products for each "vector" in the batch.
            We use the vmap prototype feature as the backend to vectorize calls
            to the autograd engine so that this computation can be performed in a
            single call. This should lead to performance improvements when compared
            to manually looping and performing backward multiple times. Note that
            due to this feature being experimental, there may be performance
            cliffs. Please use ``torch._C._debug_only_display_vmap_fallback_warnings(True)``
            to show any performance warnings and file an issue on github if warnings exist
            for your use case. Defaults to ``False``.
        materialize_grads (bool, optional): If ``True``, set the gradient for unused inputs
            to zero instead of None. This is useful when computing higher-order derivatives.
            If ``materialize_grads`` is ``True`` and ``allow_unused`` is ``False``, an error
            will be raised. Defaults to ``False``.

    '''
def variable(*args, **kwargs) -> None: ...
is_multithreading_enabled = torch._C._is_multithreading_enabled
is_view_replay_enabled = torch._C._is_view_replay_enabled

# Names in __all__ with no definition:
#   grad_mode
