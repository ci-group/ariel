import contextlib
import torch
from .apis import vmap as vmap
from .vmap import doesnt_support_saved_tensors_hooks as doesnt_support_saved_tensors_hooks, get_chunk_sizes as get_chunk_sizes
from _typeshed import Incomplete
from collections.abc import Generator
from torch._C._functorch import _assert_wrapped_functional as _assert_wrapped_functional, _func_decrement_nesting as _func_decrement_nesting, _func_increment_nesting as _func_increment_nesting, _grad_decrement_nesting as _grad_decrement_nesting, _grad_increment_nesting as _grad_increment_nesting, _jvp_decrement_nesting as _jvp_decrement_nesting, _jvp_increment_nesting as _jvp_increment_nesting, _propagate_functional_input_mutation as _propagate_functional_input_mutation, _unwrap_for_grad as _unwrap_for_grad, _unwrap_functional_tensor as _unwrap_functional_tensor, _wrap_for_grad as _wrap_for_grad, _wrap_functional_tensor as _wrap_functional_tensor, get_inplace_requires_grad_allowed as get_inplace_requires_grad_allowed, get_unwrapped as get_unwrapped, is_functorch_wrapped_tensor as is_functorch_wrapped_tensor, set_inplace_requires_grad_allowed as set_inplace_requires_grad_allowed
from torch._functorch.utils import argnums_t as argnums_t, exposed_in as exposed_in
from torch._subclasses.functional_tensor import FunctionalTensor as FunctionalTensor
from torch.fx.experimental import const_fold as const_fold
from torch.fx.experimental.proxy_tensor import make_fx as make_fx
from torch.utils._pytree import tree_flatten as tree_flatten, tree_map as tree_map, tree_map_ as tree_map_, tree_map_only as tree_map_only, tree_unflatten as tree_unflatten, treespec_pprint as treespec_pprint
from typing import Any, Callable

def lazy_dynamo_disallow(func): ...
@contextlib.contextmanager
def enable_inplace_requires_grad(enabled) -> Generator[None]: ...
def _set_tensor_requires_grad(x): ...
def _create_differentiable(inps, level=None): ...
def _undo_create_differentiable(inps, level=None): ...
def _is_differentiable(maybe_tensor): ...
def _any_differentiable(tensor_or_tuple_of_tensors): ...
def _wrap_tensor_for_grad(maybe_tensor, level): ...
def _wrap_all_tensors(tensor_pytree, level): ...
def _as_tuple(val): ...
def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph: bool = False, create_graph: bool = True): ...
def vjp(func: Callable, *primals, has_aux: bool = False):
    '''
    Standing for the vector-Jacobian product, returns a tuple containing the
    results of ``func`` applied to ``primals`` and a function that, when
    given ``cotangents``, computes the reverse-mode Jacobian of ``func`` with
    respect to ``primals`` times ``cotangents``.

    Args:
        func (Callable): A Python function that takes one or more arguments. Must
            return one or more Tensors.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, vjp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the vjp of
        ``func`` with respect to all ``primals`` using the cotangents passed
        to the returned function. If ``has_aux is True``, then instead returns a
        ``(output, vjp_fn, aux)`` tuple.
        The returned ``vjp_fn`` function will return a tuple of each VJP.

    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`

        >>> x = torch.randn([5])
        >>> f = lambda x: x.sin().sum()
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> grad = vjpfunc(torch.tensor(1.))[0]
        >>> assert torch.allclose(grad, torch.func.grad(f)(x))

    However, :func:`vjp` can support functions with multiple outputs by
    passing in the cotangents for each of the outputs

        >>> x = torch.randn([5])
        >>> f = lambda x: (x.sin(), x.cos())
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    :func:`vjp` can even support outputs being Python structs

        >>> x = torch.randn([5])
        >>> f = lambda x: {\'first\': x.sin(), \'second\': x.cos()}
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> cotangents = {\'first\': torch.ones([5]), \'second\': torch.ones([5])}
        >>> vjps = vjpfunc(cotangents)
        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())

    The function returned by :func:`vjp` will compute the partials with
    respect to each of the ``primals``

        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])
        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)
        >>> cotangents = torch.randn([5, 5])
        >>> vjps = vjpfunc(cotangents)
        >>> assert len(vjps) == 2
        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))
        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))

    ``primals`` are the positional arguments for ``f``. All kwargs use their
    default value

        >>> x = torch.randn([5])
        >>> def f(x, scale=4.):
        >>>   return x * scale
        >>>
        >>> (_, vjpfunc) = torch.func.vjp(f, x)
        >>> vjps = vjpfunc(torch.ones_like(x))
        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.))

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``vjp``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:

            >>> # xdoctest: +SKIP(failing)
            >>> with torch.no_grad():
            >>>     vjp(f)(x)

        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``vjp`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    '''
@contextlib.contextmanager
def grad_increment_nesting() -> Generator[Incomplete]: ...
def enter_jvp_nesting(): ...
def exit_jvp_nesting() -> None: ...
@contextlib.contextmanager
def jvp_increment_nesting() -> Generator[Incomplete]: ...
@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(func: Callable, *primals, argnums: argnums_t | None = None, has_aux: bool = False): ...
def _safe_zero_index(x): ...
def error_if_complex(func_name, args, is_input) -> None: ...
def jacrev(func: Callable, argnums: int | tuple[int] = 0, *, has_aux: bool = False, chunk_size: int | None = None, _preallocate_and_copy: bool = False):
    '''
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using reverse mode autodiff

    .. note::
        Using :attr:`chunk_size=1` is equivalent to computing the jacobian
        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are
        not applicable.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        chunk_size (None or int): If None (default), use the maximum chunk size
            (equivalent to doing a single vmap over vjp to compute the jacobian).
            If 1, then compute the jacobian row-by-row with a for-loop.
            If not None, then compute the jacobian :attr:`chunk_size` rows at a time
            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing
            the jacobian, please try to specify a non-None chunk_size.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>> jacobian = jacrev(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacrev
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    :func:`jacrev` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacrev, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacrev(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    Additionally, :func:`jacrev` can be composed with itself to produce
    Hessians

        >>> from torch.func import jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacrev(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacrev` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacrev
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    .. note::
        Using PyTorch ``torch.no_grad`` together with ``jacrev``.
        Case 1: Using ``torch.no_grad`` inside a function:

            >>> def f(x):
            >>>     with torch.no_grad():
            >>>         c = x ** 2
            >>>     return x - c

        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.

        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:

            >>> with torch.no_grad():
            >>>     jacrev(f)(x)

        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the
        outer one. This is because ``jacrev`` is a "function transform": its result
        should not depend on the result of a context manager outside of ``f``.
    '''
def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None) -> Generator[Incomplete]: ...
def _construct_standard_basis_for(tensors, tensor_numels): ...
def _validate_and_wrap_argnum(argnum, num_args): ...
def _check_unique_non_empty(argnums) -> None: ...
def _replace_args(old_args, new_args, argnums): ...
def _validate_and_wrap_argnums(argnums, num_args): ...
def _slice_argnums(args, argnums, as_tuple: bool = True): ...

JVP_NESTING: int

def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None: ...
def assert_non_empty_tensor_output(output: list[Any], api: str) -> None: ...
def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None: ...
def assert_non_empty_list_of_tensors(output: list[torch.Tensor], api: str, argname: str) -> None: ...

jvp_str: str

def safe_unpack_dual(dual, strict): ...
def jvp(func: Callable, primals: Any, tangents: Any, *, strict: bool = False, has_aux: bool = False):
    '''
    Standing for the Jacobian-vector product, returns a tuple containing
    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at
    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. The returned function will also be computing the
            derivative with respect to these arguments
        tangents (Tensors): The "vector" for which Jacobian-vector-product is
            computed. Must be the same structure and sizes as the inputs to
            ``func``.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            other auxiliary objects that will not be differentiated.
            Default: False.

    Returns:
        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``
        evaluated at ``primals`` and the Jacobian-vector product.
        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.

    jvp is useful when you wish to compute gradients of a function R^1 -> R^N

        >>> from torch.func import jvp
        >>> x = torch.randn([])
        >>> f = lambda x: x * torch.tensor([1., 2., 3])
        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))
        >>> assert torch.allclose(value, f(x))
        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))

    :func:`jvp` can support functions with multiple inputs by passing in the
    tangents for each of the inputs

         >>> from torch.func import jvp
         >>> x = torch.randn(5)
         >>> y = torch.randn(5)
         >>> f = lambda x, y: (x * y)
         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))
         >>> assert torch.allclose(output, x + y)

    '''
def _jvp_with_argnums(func: Callable, primals: Any, tangents: Any, argnums: argnums_t | None, *, strict: bool = False, has_aux: bool): ...
def safe_unflatten(tensor, dim, shape): ...
def jacfwd(func: Callable, argnums: argnums_t = 0, has_aux: bool = False, *, randomness: str = 'error'):
    '''
    Computes the Jacobian of ``func`` with respect to the arg(s) at index
    ``argnum`` using forward-mode autodiff

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Jacobian with respect to.
            Default: 0.
        has_aux (bool): Flag indicating that ``func`` returns a
            ``(output, aux)`` tuple where the first element is the output of
            the function to be differentiated and the second element is
            auxiliary objects that will not be differentiated.
            Default: False.
        randomness(str): Flag indicating what type of randomness to use.
            See :func:`vmap` for more detail. Allowed: "different", "same", "error".
            Default: "error"

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Jacobian of ``func`` with respect to the arg(s) at
        ``argnums``. If ``has_aux is True``, then the returned function
        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``
        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use :func:`jacrev`, which has better operator coverage.

    A basic usage with a pointwise, unary operation will give a diagonal array
    as the Jacobian

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>> jacobian = jacfwd(torch.sin)(x)
        >>> expected = torch.diag(torch.cos(x))
        >>> assert torch.allclose(jacobian, expected)

    :func:`jacfwd` can be composed with vmap to produce batched
    Jacobians:

        >>> from torch.func import jacfwd, vmap
        >>> x = torch.randn(64, 5)
        >>> jacobian = vmap(jacfwd(torch.sin))(x)
        >>> assert jacobian.shape == (64, 5, 5)

    If you would like to compute the output of the function as well as the
    jacobian of the function, use the ``has_aux`` flag to return the output
    as an auxiliary object:

        >>> from torch.func import jacfwd
        >>> x = torch.randn(5)
        >>>
        >>> def f(x):
        >>>   return x.sin()
        >>>
        >>> def g(x):
        >>>   result = f(x)
        >>>   return result, result
        >>>
        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)
        >>> assert torch.allclose(f_x, f(x))

    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`
    to produce Hessians

        >>> from torch.func import jacfwd, jacrev
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hessian = jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))

    By default, :func:`jacfwd` computes the Jacobian with respect to the first
    input. However, it can compute the Jacboian with respect to a different
    argument by using ``argnums``:

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=1)(x, y)
        >>> expected = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian, expected)

    Additionally, passing a tuple to ``argnums`` will compute the Jacobian
    with respect to multiple arguments

        >>> from torch.func import jacfwd
        >>> def f(x, y):
        >>>   return x + y ** 2
        >>>
        >>> x, y = torch.randn(5), torch.randn(5)
        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)
        >>> expectedX = torch.diag(torch.ones_like(x))
        >>> expectedY = torch.diag(2 * y)
        >>> assert torch.allclose(jacobian[0], expectedX)
        >>> assert torch.allclose(jacobian[1], expectedY)

    '''
def hessian(func, argnums: int = 0):
    '''
    Computes the Hessian of ``func`` with respect to the arg(s) at index
    ``argnum`` via a forward-over-reverse strategy.

    The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is
    a good default for good performance. It is possible to compute Hessians
    through other compositions of :func:`jacfwd` and :func:`jacrev` like
    ``jacfwd(jacfwd(func))`` or ``jacrev(jacrev(func))``.

    Args:
        func (function): A Python function that takes one or more arguments,
            one of which must be a Tensor, and returns one or more Tensors
        argnums (int or Tuple[int]): Optional, integer or tuple of integers,
            saying which arguments to get the Hessian with respect to.
            Default: 0.

    Returns:
        Returns a function that takes in the same inputs as ``func`` and
        returns the Hessian of ``func`` with respect to the arg(s) at
        ``argnums``.

    .. note::
        You may see this API error out with "forward-mode AD not implemented
        for operator X". If so, please file a bug report and we will prioritize it.
        An alternative is to use ``jacrev(jacrev(func))``, which has better
        operator coverage.

    A basic usage with a R^N -> R^1 function gives a N x N Hessian:

        >>> from torch.func import hessian
        >>> def f(x):
        >>>   return x.sin().sum()
        >>>
        >>> x = torch.randn(5)
        >>> hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)
        >>> assert torch.allclose(hess, torch.diag(-x.sin()))

    '''
@doesnt_support_saved_tensors_hooks
def grad_and_value_impl(func, argnums, has_aux, args, kwargs) -> Callable: ...
def grad_impl(func: Callable, argnums: argnums_t, has_aux: bool, args, kwargs): ...
def _maybe_wrap_functional_tensor(maybe_tensor, level, *, _python_functionalize: bool = False): ...
def _wrap_all_tensors_to_functional(tensor_pytree, level, *, _python_functionalize: bool = False): ...
def _maybe_unwrap_functional_tensor(maybe_tensor, *, reapply_views: bool): ...
def _unwrap_all_tensors_from_functional(tensor_pytree, *, reapply_views: bool): ...
def functionalize(func: Callable, *, remove: str = 'mutations') -> Callable:
    '''
    functionalize is a transform that can be used to remove (intermediate)
    mutations and aliasing from a function, while preserving the function\'s
    semantics.

    ``functionalize(func)`` returns a new function with the same semantics
    as ``func``, but with all intermediate mutations removed.
    Every inplace operation performed on an intermediate tensor:
    ``intermediate.foo_()``
    gets replaced by its out-of-place equivalent:
    ``intermediate_updated = intermediate.foo()``.

    functionalize is useful for shipping a pytorch program off to
    backends or compilers that aren\'t able to easily represent
    mutations or aliasing operators.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        remove (str): An optional string argument, that takes on either
            the value \'mutations\' or \'mutations_and_views\'.
            If \'mutations\' is passed in then all mutating operators
            will be replaced with their non-mutating equivalents.
            If \'mutations_and_views\' is passed in, then additionally, all aliasing
            operators will be replaced with their non-aliasing equivalents.
            Default: \'mutations\'.

    Returns:
        Returns a new "functionalized" function. It takes the same inputs as
        ``func``, and has the same behavior, but any mutations
        (and optionally aliasing) performed on intermediate tensors
        in the function will be removed.

    functionalize will also remove mutations (and views) that were performed on function inputs.
    However to preserve semantics, functionalize will "fix up" the mutations after
    the transform has finished running, by detecting if any tensor inputs "should have"
    been mutated, and copying the new data back to the inputs if necessary.


    Example::

        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.fx.experimental.proxy_tensor import make_fx
        >>> from torch.func import functionalize
        >>>
        >>> # A function that uses mutations and views, but only on intermediate tensors.
        >>> def f(a):
        ...     b = a + 1
        ...     c = b.view(-1)
        ...     c.add_(1)
        ...     return b
        ...
        >>> inpt = torch.randn(2)
        >>>
        >>> out1 = f(inpt)
        >>> out2 = functionalize(f)(inpt)
        >>>
        >>> # semantics are the same (outputs are equivalent)
        >>> print(torch.allclose(out1, out2))
        True
        >>>
        >>> f_traced = make_fx(f)(inpt)
        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove=\'mutations_and_views\'))(inpt)
        >>>
        >>> print(f_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1])
            add_ = torch.ops.aten.add_(view, 1);  view = None
            return add

        >>> print(f_no_mutations_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view = torch.ops.aten.view(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view, 1);  view = None
            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None
            return view_1

        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            add = torch.ops.aten.add(a_1, 1);  a_1 = None
            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None
            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None
            return view_copy_1


        >>> # A function that mutates its input tensor
        >>> def f(a):
        ...     b = a.view(-1)
        ...     b.add_(1)
        ...     return a
        ...
        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove=\'mutations_and_views\'))(inpt)
        >>> #
        >>> # All mutations and views have been removed,
        >>> # but there is an extra copy_ in the graph to correctly apply the mutation to the input
        >>> # after the function has completed.
        >>> print(f_no_mutations_and_views_traced.code)



        def forward(self, a_1):
            view_copy = torch.ops.aten.view_copy(a_1, [-1])
            add = torch.ops.aten.add(view_copy, 1);  view_copy = None
            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None
            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None
            return view_copy_1


    There are a few "failure modes" for functionalize that are worth calling out:
      (1) Like other torch.func transforms, `functionalize()` doesn\'t work with functions
          that directly use `.backward()`. The same is true for torch.autograd.grad.
          If you want to use autograd, you can compute gradients directly
          with `functionalize(grad(f))`.
      (2) Like other torch.func transforms, `functionalize()` doesn\'t work with global state.
          If you call `functionalize(f)` on a function that takes views / mutations of
          non-local state, functionalization will simply no-op and pass the view/mutation
          calls directly to the backend.
          One way to work around this is is to ensure that any non-local state creation
          is wrapped into a larger function, which you then call functionalize on.
      (3) `resize_()` has some limitations: functionalize will only work on programs
          that use resize_()` as long as the tensor being resized is not a view.
      (4) `as_strided()` has some limitations: functionalize will not work on
          `as_strided()` calls that result in tensors with overlapping memory.


    Finally, a helpful mental model for understanding functionalization is that
    most user pytorch programs are writing with the public torch API.
    When executed, torch operators are generally decomposed into
    our internal C++ "ATen" API.
    The logic for functionalization happens entirely at the level of ATen.
    Functionalization knows how to take every aliasing operator in ATen,
    and map it to its non-aliasing equivalent
    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),
    and how to take every mutating operator in ATen,
    and map it to its non-mutating equivalent
    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),
    while tracking aliases and mutations out-of-line to know when to fix things up.
    Information about which ATen operators are aliasing or mutating all comes from
    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.
    '''
def linearize(func: Callable, *primals) -> tuple[Any, Callable]:
    """
    Returns the value of ``func`` at ``primals`` and linear approximation
    at ``primals``.

    Args:
        func (Callable): A Python function that takes one or more arguments.
        primals (Tensors): Positional arguments to ``func`` that must all be
            Tensors. These are the values at which the function is linearly approximated.

    Returns:
        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``
        applied to ``primals`` and a function that computes the jvp of
        ``func`` evaluated at ``primals``.

    linearize is useful if jvp is to be computed multiple times at ``primals``. However,
    to achieve this, linearize saves intermediate computation and has higher memory requirements
    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient
    to compute vmap(jvp) instead of using linearize.

    .. note::
        linearize evaluates ``func`` twice. Please file an issue for an implementation
        with a single evaluation.

    Example::

        >>> import torch
        >>> from torch.func import linearize
        >>> def fn(x):
        ...     return x.sin()
        ...
        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))
        >>> jvp_fn(torch.ones(3, 3))
        tensor([[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]])
        >>>

    """
def debug_unwrap(tensor: torch.Tensor, *, recurse: bool = True) -> torch.Tensor:
    """Unwraps a functorch tensor (e.g. BatchedTensor, GradTrackingTensor) to its underlying tensor.

    This function should only be used in a debug setting (e.g. trying to print the
    value of a Tensor in a debugger). Otherwise, using the result of function
    inside of a function being transformed will lead to undefined behavior.
    """
