import torch
from torch.types import _TensorOrTensors
from typing import Callable

__all__ = ['gradcheck', 'gradgradcheck', 'GradcheckError', 'get_numerical_jacobian', 'get_analytical_jacobian', 'get_numerical_jacobian_wrt_specific_input']

class GradcheckError(RuntimeError):
    """Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""

def get_numerical_jacobian(fn, inputs, target=None, eps: float = 0.001, grad_out: float = 1.0):
    """Compute the numerical Jacobian for a given fn and its inputs.

    This is a Deprecated API.

    Args:
        fn: the function to compute the Jacobian for (must take inputs as a tuple)
        inputs: input to `fn`
        target: the Tensors wrt whom Jacobians are calculated (default=`input`)
        eps: the magnitude of the perturbation during finite differencing
             (default=`1e-3`)
        grad_out: defaults to 1.0.

    Returns:
        A list of Jacobians of `fn` (restricted to its first output) with respect to
        each input or target, if provided.

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad: bool = False) -> tuple[torch.Tensor, ...]: ...
def get_analytical_jacobian(inputs, output, nondet_tol: float = 0.0, grad_out: float = 1.0): ...
def gradcheck(func: Callable[..., _TensorOrTensors], inputs: _TensorOrTensors, *, eps: float = 1e-06, atol: float = 1e-05, rtol: float = 0.001, raise_exception: bool = True, nondet_tol: float = 0.0, check_undefined_grad: bool = True, check_grad_dtypes: bool = False, check_batched_grad: bool = False, check_batched_forward_grad: bool = False, check_forward_ad: bool = False, check_backward_ad: bool = True, fast_mode: bool = False, masked: bool | None = None) -> bool:
    """Check gradients computed via small finite differences against analytical
    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    For most of the complex functions we consider for optimization purposes, no notion of
    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of
    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient
    computation is done under the assumption that the overall function has a real-valued
    output, we treat functions with complex output in a special way. For these functions,
    gradcheck is applied to two real-valued functions corresponding to taking the real
    components of the complex outputs for the first, and taking the imaginary components
    of the complex outputs for the second. For more details, check out
    :ref:`complex_autograd-doc`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. note::
        Gradcheck may fail when evaluated on non-differentiable points
        because the numerically computed gradients via finite differencing may differ
        those computed analytically (not necessarily because either is incorrect).
        For more context, see :ref:`non-differentiable-func-grad`.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.Tensor.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.
        check_undefined_grad (bool, optional): if ``True``, check if undefined output grads
            are supported and treated as zeros, for ``Tensor`` outputs.
        check_batched_grad (bool, optional): if ``True``, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.
        check_batched_forward_grad (bool, optional): if ``True``, checks if we can compute
            batched forward gradients using forward ad and prototype vmap support. Defaults to ``False``.
        check_forward_ad (bool, optional): if ``True``, check that the gradients computed with forward
            mode AD match the numerical ones. Defaults to ``False``.
        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on
            backward mode AD to be implemented. Defaults to ``True``.
        fast_mode (bool, optional): Fast mode for gradcheck and gradgradcheck is currently only
            implemented for R to R functions. If none of the inputs and outputs are complex
            a faster implementation of gradcheck that no longer computes the entire jacobian
            is run; otherwise, we fall back to the slow implementation.
        masked (bool, optional): if ``True``, the gradients of unspecified elements of
            sparse tensors are ignored. Defaults to ``False``.
    Returns:
        ``True`` if all differences satisfy allclose condition

    """
def gradgradcheck(func: Callable[..., _TensorOrTensors], inputs: _TensorOrTensors, grad_outputs: _TensorOrTensors | None = None, *, eps: float = 1e-06, atol: float = 1e-05, rtol: float = 0.001, gen_non_contig_grad_outputs: bool = False, raise_exception: bool = True, nondet_tol: float = 0.0, check_undefined_grad: bool = True, check_grad_dtypes: bool = False, check_batched_grad: bool = False, check_fwd_over_rev: bool = False, check_rev_over_rev: bool = True, fast_mode: bool = False, masked: bool = False) -> bool:
    """Check gradients of gradients computed via small finite differences
    against analytical gradients wrt tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point or complex type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.Tensor.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.
        check_undefined_grad (bool, optional): if True, check if undefined output grads
            are supported and treated as zeros
        check_batched_grad (bool, optional): if True, check if we can compute
            batched gradients using prototype vmap support. Defaults to False.
        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that
            no longer computes the entire jacobian.
        masked (bool, optional): if True, the gradients of unspecified elements of
            sparse tensors are ignored (default, False).
    Returns:
        True if all differences satisfy allclose condition
    """
