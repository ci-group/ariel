import torch
import torch._C as _C
from _typeshed import Incomplete
from typing import Any

__all__ = ['FunctionCtx', 'BackwardCFunction', 'FunctionMeta', 'Function', 'once_differentiable', 'InplaceFunction', 'NestedIOFunction']

class FunctionCtx:
    to_save: Incomplete
    def save_for_backward(self, *tensors: torch.Tensor):
        """Save given tensors for a future call to :func:`~Function.backward`.

        ``save_for_backward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and only with tensors.

        All tensors intended to be used in the backward pass should be saved
        with ``save_for_backward`` (as opposed to directly on ``ctx``) to prevent
        incorrect gradients and memory leaks, and enable the application of saved
        tensor hooks. See :class:`torch.autograd.graph.saved_tensors_hooks`.
        See :ref:`extending-autograd` for more details.

        Note that if intermediary tensors, tensors that are neither inputs
        nor outputs of :func:`forward`, are saved for backward, your custom Function
        may not support double backward.
        Custom Functions that do not support double backward should decorate their
        :func:`backward` method with ``@once_differentiable`` so that performing
        double backward raises an error. If you'd like to support double backward,
        you can either recompute intermediaries based on the inputs during backward
        or return the intermediaries as the outputs of the custom Function. See the
        `double backward tutorial <https://pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html>`_
        for more details.

        In :func:`backward`, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``. This is a no-op.

        See :ref:`extending-autograd` for more details on how to use this method.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
            >>>         w = x * z
            >>>         out = x * y + y * z + w * y
            >>>         ctx.save_for_backward(x, y, w, out)
            >>>         ctx.z = z  # z is not a tensor
            >>>         return out
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, grad_out):
            >>>         x, y, w, out = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         gx = grad_out * (y + y * z)
            >>>         gy = grad_out * (x + z + w)
            >>>         gz = None
            >>>         return gx, gy, gz
            >>>
            >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double)
            >>> b = torch.tensor(2., requires_grad=True, dtype=torch.double)
            >>> c = 4
            >>> d = Func.apply(a, b, c)

        """
    saved_for_forward: Incomplete
    def save_for_forward(self, *tensors: torch.Tensor):
        """Save given tensors for a future call to :func:`~Function.jvp`.

        ``save_for_forward`` should be called at most once, in either the
        :func:`setup_context` or :func:`forward` methods, and all arguments
        should be tensors.

        In :func:`jvp`, saved objects can be accessed through the :attr:`saved_tensors`
        attribute.

        Arguments can also be ``None``. This is a no-op.

        See :ref:`extending-autograd` for more details on how to use this method.

        Example::

            >>> # xdoctest: +SKIP
            >>> class Func(torch.autograd.Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x: torch.Tensor, y: torch.Tensor, z: int):
            >>>         ctx.save_for_backward(x, y)
            >>>         ctx.save_for_forward(x, y)
            >>>         ctx.z = z
            >>>         return x * y * z
            >>>
            >>>     @staticmethod
            >>>     def jvp(ctx, x_t, y_t, _):
            >>>         x, y = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         return z * (y * x_t + x * y_t)
            >>>
            >>>     @staticmethod
            >>>     def vjp(ctx, grad_out):
            >>>         x, y = ctx.saved_tensors
            >>>         z = ctx.z
            >>>         return z * grad_out * y, z * grad_out * x, None
            >>>
            >>>     a = torch.tensor(1., requires_grad=True, dtype=torch.double)
            >>>     t = torch.tensor(1., dtype=torch.double)
            >>>     b = torch.tensor(2., requires_grad=True, dtype=torch.double)
            >>>     c = 4
            >>>
            >>>     with fwAD.dual_level():
            >>>         a_dual = fwAD.make_dual(a, t)
            >>>         d = Func.apply(a_dual, b, c)

        """
    dirty_tensors: Incomplete
    def mark_dirty(self, *args: torch.Tensor):
        """Mark given tensors as modified in an in-place operation.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be inputs.

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.

        Examples::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class Inplace(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         x_npy = x.numpy() # x_npy shares storage with x
            >>>         x_npy += 1
            >>>         ctx.mark_dirty(x)
            >>>         return x
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, grad_output):
            >>>         return grad_output
            >>>
            >>> a = torch.tensor(1., requires_grad=True, dtype=torch.double).clone()
            >>> b = a * a
            >>> Inplace.apply(a)  # This would lead to wrong gradients!
            >>>                   # but the engine would not know unless we mark_dirty
            >>> # xdoctest: +SKIP
            >>> b.backward() # RuntimeError: one of the variables needed for gradient
            >>>              # computation has been modified by an inplace operation

        """
    def mark_shared_storage(self, *pairs) -> None: ...
    non_differentiable: Incomplete
    def mark_non_differentiable(self, *args: torch.Tensor):
        """Mark outputs as non-differentiable.

        This should be called at most once, in either the :func:`setup_context`
        or :func:`forward` methods, and all arguments should be tensor outputs.

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a sort. See example::
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         sorted, idx = x.sort()
            >>>         ctx.mark_non_differentiable(idx)
            >>>         ctx.save_for_backward(x, idx)
            >>>         return sorted, idx
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):  # still need to accept g2
            >>>         x, idx = ctx.saved_tensors
            >>>         grad_input = torch.zeros_like(x)
            >>>         grad_input.index_add_(0, idx, g1)
            >>>         return grad_input

        """
    materialize_grads: Incomplete
    def set_materialize_grads(self, value: bool):
        """Set whether to materialize grad tensors. Default is ``True``.

        This should be called only from either the :func:`setup_context` or
        :func:`forward` methods.

        If ``True``, undefined grad tensors will be expanded to tensors full of zeros
        prior to calling the :func:`backward` and :func:`jvp` methods.

        Example::

            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
            >>> class SimpleFunc(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         return x.clone(), x.clone()
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):
            >>>         return g1 + g2  # No check for None necessary
            >>>
            >>> # We modify SimpleFunc to handle non-materialized grad outputs
            >>> class Func(Function):
            >>>     @staticmethod
            >>>     def forward(ctx, x):
            >>>         ctx.set_materialize_grads(False)
            >>>         ctx.save_for_backward(x)
            >>>         return x.clone(), x.clone()
            >>>
            >>>     @staticmethod
            >>>     @once_differentiable
            >>>     def backward(ctx, g1, g2):
            >>>         x, = ctx.saved_tensors
            >>>         grad_input = torch.zeros_like(x)
            >>>         if g1 is not None:  # We must check for None now
            >>>             grad_input += g1
            >>>         if g2 is not None:
            >>>             grad_input += g2
            >>>         return grad_input
            >>>
            >>> a = torch.tensor(1., requires_grad=True)
            >>> b, _ = Func.apply(a)  # induces g2 to be undefined

        """
_ContextMethodMixin = FunctionCtx

class _HookMixin:
    @staticmethod
    def _register_hook(backward_hooks, hook): ...

class BackwardCFunction(_C._FunctionBase, FunctionCtx, _HookMixin):
    """
    This class is used for internal autograd work. Do not use.
    """
    def apply(self, *args):
        """
        Apply method used when executing this Node during the backward
        """
    def apply_jvp(self, *args):
        """
        Apply method used when executing forward mode AD during the forward
        """
    def _compiled_autograd_key(self): ...

class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """
    def __init__(cls, name, bases, attrs) -> None: ...

class _SingleLevelFunction(_C._FunctionBase, FunctionCtx, _HookMixin, metaclass=FunctionMeta):
    @staticmethod
    def forward(*args: Any, **kwargs: Any) -> Any:
        """Define the forward of the custom autograd Function.

        This function is to be overridden by all subclasses.
        There are two ways to define forward:

        Usage 1 (Combined forward and ctx)::

            @staticmethod
            def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
                pass

        - It must accept a context ctx as the first argument, followed by any
          number of arguments (tensors or other types).
        - See :ref:`combining-forward-context` for more details

        Usage 2 (Separate forward and ctx)::

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                pass

            @staticmethod
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> None:
                pass

        - The forward no longer accepts a ctx argument.
        - Instead, you must also override the :meth:`torch.autograd.Function.setup_context`
          staticmethod to handle setting up the ``ctx`` object.
          ``output`` is the output of the forward, ``inputs`` are a Tuple of inputs
          to the forward.
        - See :ref:`extending-autograd` for more details

        The context can be used to store arbitrary data that can be then
        retrieved during the backward pass. Tensors should not be stored
        directly on `ctx` (though this is not currently enforced for
        backward compatibility). Instead, tensors should be saved either with
        :func:`ctx.save_for_backward` if they are intended to be used in
        ``backward`` (equivalently, ``vjp``) or :func:`ctx.save_for_forward`
        if they are intended to be used for in ``jvp``.
        """
    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        """There are two ways to define the forward pass of an autograd.Function.

        Either:

        1. Override forward with the signature ``forward(ctx, *args, **kwargs)``.
           ``setup_context`` is not overridden. Setting up the ctx for backward
           happens inside the ``forward``.
        2. Override forward with the signature ``forward(*args, **kwargs)`` and
           override ``setup_context``. Setting up the ctx for backward happens
           inside ``setup_context`` (as opposed to inside the ``forward``)

        See :meth:`torch.autograd.Function.forward` and :ref:`extending-autograd` for more details.
        """
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Define a formula for differentiating the operation with backward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        (Defining this function is equivalent to defining the ``vjp`` function.)

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs as the :func:`forward` returned (None will be passed in
        for non tensor outputs of the forward function),
        and it should return as many tensors, as there were inputs to
        :func:`forward`. Each argument is the gradient w.r.t the given output,
        and each returned value should be the gradient w.r.t. the
        corresponding input. If an input is not a Tensor or is a Tensor not
        requiring grads, you can just pass None as a gradient for that input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computed w.r.t. the
        output.
        """
    vjp = backward
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        """Define a formula for differentiating the operation with forward mode automatic differentiation.

        This function is to be overridden by all subclasses.
        It must accept a context :attr:`ctx` as the first argument, followed by
        as many inputs as the :func:`forward` got (None will be passed in
        for non tensor inputs of the forward function),
        and it should return as many tensors as there were outputs to
        :func:`forward`. Each argument is the gradient w.r.t the given input,
        and each returned value should be the gradient w.r.t. the
        corresponding output. If an output is not a Tensor or the function is not
        differentiable with respect to that output, you can just pass None as a
        gradient for that input.

        You can use the :attr:`ctx` object to pass any value from the forward to this
        functions.
        """

class Function(_SingleLevelFunction):
    """Base class to create custom `autograd.Function`.

    To create a custom `autograd.Function`, subclass this class and implement
    the :meth:`forward` and :meth:`backward` static methods. Then, to use your custom
    op in the forward pass, call the class method ``apply``. Do not call
    :meth:`forward` directly.

    To ensure correctness and best performance, make sure you are calling the
    correct methods on ``ctx`` and validating your backward function using
    :func:`torch.autograd.gradcheck`.

    See :ref:`extending-autograd` for more details on how to use this class.

    Examples::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> class Exp(Function):
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> # Use it by calling the apply method:
        >>> # xdoctest: +SKIP
        >>> output = Exp.apply(input)
    """
    def __init__(self, *args, **kwargs) -> None: ...
    def __call__(self, *args, **kwargs) -> None: ...
    generate_vmap_rule: bool
    @staticmethod
    def vmap(info, in_dims, *args) -> None:
        """Define the behavior for this autograd.Function underneath :func:`torch.vmap`.

        For a :func:`torch.autograd.Function` to support
        :func:`torch.vmap`, you must either override this static method, or set
        ``generate_vmap_rule`` to ``True`` (you may not do both).

        If you choose to override this staticmethod: it must accept

        - an ``info`` object as the first argument. ``info.batch_size``
          specifies the size of the dimension being vmapped over,
          while ``info.randomness`` is the randomness option passed to
          :func:`torch.vmap`.
        - an ``in_dims`` tuple as the second argument.
          For each arg in ``args``, ``in_dims`` has a corresponding
          ``Optional[int]``. It is ``None`` if the arg is not a Tensor or if
          the arg is not being vmapped over, otherwise, it is an integer
          specifying what dimension of the Tensor is being vmapped over.
        - ``*args``, which is the same as the args to :meth:`~Function.forward`.

        The return of the vmap staticmethod is a tuple of ``(output, out_dims)``.
        Similar to ``in_dims``, ``out_dims`` should be of the same structure as
        ``output`` and contain one ``out_dim`` per output that specifies if the
        output has the vmapped dimension and what index it is in.

        Please see :ref:`func-autograd-function` for more details.
        """
    @classmethod
    def apply(cls, *args, **kwargs): ...
    @staticmethod
    def _compiled_autograd_key(ctx): ...

def once_differentiable(fn): ...

class InplaceFunction(Function):
    """
    This class is here only for backward compatibility reasons.
    Use :class:`Function` instead of this for any new use case.
    """
    inplace: Incomplete
    def __init__(self, inplace: bool = False) -> None: ...

class NestedIOFunction(Function):
    """
    This class is here only for backward compatibility reasons.
    Use :class:`Function` instead of this for any new use case.
    """
    _nested_input: Incomplete
    def _do_forward(self, *input): ...
    retain_variables: Incomplete
    def _do_backward(self, gradients, retain_variables): ...
    def backward(self, *gradients: Any) -> Any:
        """
        Shared backward utility.
        """
    __call__ = _do_forward
    _nested_output: Incomplete
    def forward(self, *args: Any) -> Any:
        """
        Shared forward utility.
        """
    to_save: Incomplete
    _to_save_nested: Incomplete
    def save_for_backward(self, *args: Any) -> None:
        """
        See :meth:`Function.save_for_backward`.
        """
    @property
    def saved_tensors(self):
        """
        See :meth:`Function.saved_tensors`.
        """
    dirty_tensors: Incomplete
    def mark_dirty(self, *args: Any, **kwargs: Any) -> None:
        """
        See :meth:`Function.mark_dirty`.
        """
    non_differentiable: Incomplete
    def mark_non_differentiable(self, *args: Any, **kwargs: Any) -> None:
        """
        See :meth:`Function.mark_non_differentiable`.
        """
    def forward_extended(self, *input: Any) -> None:
        """
        User defined forward.
        """
    def backward_extended(self, *grad_output: Any) -> None:
        """
        User defined backward.
        """
