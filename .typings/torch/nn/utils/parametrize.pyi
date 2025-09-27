from _typeshed import Incomplete
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from torch import Tensor
from torch.nn.modules.container import Module, ModuleList
from torch.nn.parameter import Parameter

__all__ = ['cached', 'ParametrizationList', 'register_parametrization', 'is_parametrized', 'remove_parametrizations', 'type_before_parametrizations', 'transfer_parametrizations_and_params']

@contextmanager
def cached() -> Generator[None]:
    """Context manager that enables the caching system within parametrizations registered with :func:`register_parametrization`.

    The value of the parametrized objects is computed and cached the first time
    they are required when this context manager is active. The cached values are
    discarded when leaving the context manager.

    This is useful when using a parametrized parameter more than once in the forward pass.
    An example of this is when parametrizing the recurrent kernel of an RNN or when
    sharing weights.

    The simplest way to activate the cache is by wrapping the forward pass of the neural network

    .. code-block:: python

        import torch.nn.utils.parametrize as P

        ...
        with P.cached():
            output = model(inputs)

    in training and evaluation. One may also wrap the parts of the modules that use
    several times the parametrized tensors. For example, the loop of an RNN with a
    parametrized recurrent kernel:

    .. code-block:: python

        with P.cached():
            for x in xs:
                out_rnn = self.rnn_cell(x, out_rnn)
    """

class ParametrizationList(ModuleList):
    """A sequential container that holds and manages the original parameters or buffers of a parametrized :class:`torch.nn.Module`.

    It is the type of ``module.parametrizations[tensor_name]`` when ``module[tensor_name]``
    has been parametrized with :func:`register_parametrization`.

    If the first registered parametrization has a ``right_inverse`` that returns one tensor or
    does not have a ``right_inverse`` (in which case we assume that ``right_inverse`` is the identity),
    it will hold the tensor under the name ``original``.
    If it has a ``right_inverse`` that returns more than one tensor, these will be registered as
    ``original0``, ``original1``, ...

    .. warning::
        This class is used internally by :func:`register_parametrization`. It is documented
        here for completeness. It shall not be instantiated by the user.

    Args:
        modules (sequence): sequence of modules representing the parametrizations
        original (Parameter or Tensor): parameter or buffer that is parametrized
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.
    """
    original: Tensor
    unsafe: bool
    is_tensor: Incomplete
    ntensors: Incomplete
    def __init__(self, modules: Sequence[Module], original: Tensor | Parameter, unsafe: bool = False) -> None: ...
    def right_inverse(self, value: Tensor) -> None:
        """Call the ``right_inverse`` methods of the parametrizations in the inverse registration order.

        Then, it stores the result in ``self.original`` if ``right_inverse`` outputs one tensor
        or in ``self.original0``, ``self.original1``, ... if it outputs several.

        Args:
            value (Tensor): Value to which initialize the module
        """
    def forward(self) -> Tensor: ...

def register_parametrization(module: Module, tensor_name: str, parametrization: Module, *, unsafe: bool = False) -> Module:
    '''Register a parametrization to a tensor in a module.

    Assume that ``tensor_name="weight"`` for simplicity. When accessing ``module.weight``,
    the module will return the parametrized version ``parametrization(module.weight)``.
    If the original tensor requires a gradient, the backward pass will differentiate
    through :attr:`parametrization`, and the optimizer will update the tensor accordingly.

    The first time that a module registers a parametrization, this function will add an attribute
    ``parametrizations`` to the module of type :class:`~ParametrizationList`.

    The list of parametrizations on the tensor ``weight`` will be accessible under
    ``module.parametrizations.weight``.

    The original tensor will be accessible under
    ``module.parametrizations.weight.original``.

    Parametrizations may be concatenated by registering several parametrizations
    on the same attribute.

    The training mode of a registered parametrization is updated on registration
    to match the training mode of the host module

    Parametrized parameters and buffers have an inbuilt caching system that can be activated
    using the context manager :func:`cached`.

    A :attr:`parametrization` may optionally implement a method with signature

    .. code-block:: python

        def right_inverse(self, X: Tensor) -> Union[Tensor, Sequence[Tensor]]

    This method is called on the unparametrized tensor when the first parametrization
    is registered to compute the initial value of the original tensor.
    If this method is not implemented, the original tensor will be just the unparametrized tensor.

    If all the parametrizations registered on a tensor implement `right_inverse` it is possible
    to initialize a parametrized tensor by assigning to it, as shown in the example below.

    It is possible for the first parametrization to depend on several inputs.
    This may be implemented returning a tuple of tensors from ``right_inverse``
    (see the example implementation of a ``RankOne`` parametrization below).

    In this case, the unconstrained tensors are also located under ``module.parametrizations.weight``
    with names ``original0``, ``original1``,...

    .. note::

        If unsafe=False (default) both the forward and right_inverse methods will be called
        once to perform a number of consistency checks.
        If unsafe=True, then right_inverse will be called if the tensor is not parametrized,
        and nothing will be called otherwise.

    .. note::

        In most situations, ``right_inverse`` will be a function such that
        ``forward(right_inverse(X)) == X`` (see
        `right inverse <https://en.wikipedia.org/wiki/Inverse_function#Right_inverses>`_).
        Sometimes, when the parametrization is not surjective, it may be reasonable
        to relax this.

    .. warning::

        If a parametrization depends on several inputs, :func:`~register_parametrization`
        will register a number of new parameters. If such parametrization is registered
        after the optimizer is created, these new parameters will need to be added manually
        to the optimizer. See :meth:`torch.Optimizer.add_param_group`.

    Args:
        module (nn.Module): module on which to register the parametrization
        tensor_name (str): name of the parameter or buffer on which to register
            the parametrization
        parametrization (nn.Module): the parametrization to register
    Keyword args:
        unsafe (bool): a boolean flag that denotes whether the parametrization
            may change the dtype and shape of the tensor. Default: `False`
            Warning: the parametrization is not checked for consistency upon registration.
            Enable this flag at your own risk.

    Raises:
        ValueError: if the module does not have a parameter or a buffer named :attr:`tensor_name`

    Examples:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_LAPACK)
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.utils.parametrize as P
        >>>
        >>> class Symmetric(nn.Module):
        >>>     def forward(self, X):
        >>>         return X.triu() + X.triu(1).T  # Return a symmetric matrix
        >>>
        >>>     def right_inverse(self, A):
        >>>         return A.triu()
        >>>
        >>> m = nn.Linear(5, 5)
        >>> P.register_parametrization(m, "weight", Symmetric())
        >>> print(torch.allclose(m.weight, m.weight.T))  # m.weight is now symmetric
        True
        >>> A = torch.rand(5, 5)
        >>> A = A + A.T  # A is now symmetric
        >>> m.weight = A  # Initialize the weight to be the symmetric matrix A
        >>> print(torch.allclose(m.weight, A))
        True

        >>> class RankOne(nn.Module):
        >>>     def forward(self, x, y):
        >>> # Form a rank 1 matrix multiplying two vectors
        >>>         return x.unsqueeze(-1) @ y.unsqueeze(-2)
        >>>
        >>>     def right_inverse(self, Z):
        >>> # Project Z onto the rank 1 matrices
        >>>         U, S, Vh = torch.linalg.svd(Z, full_matrices=False)
        >>> # Return rescaled singular vectors
        >>>         s0_sqrt = S[0].sqrt().unsqueeze(-1)
        >>>         return U[..., :, 0] * s0_sqrt, Vh[..., 0, :] * s0_sqrt
        >>>
        >>> linear_rank_one = P.register_parametrization(
        ...     nn.Linear(4, 4), "weight", RankOne()
        ... )
        >>> print(torch.linalg.matrix_rank(linear_rank_one.weight).item())
        1

    '''
def is_parametrized(module: Module, tensor_name: str | None = None) -> bool:
    """Determine if a module has a parametrization.

    Args:
        module (nn.Module): module to query
        tensor_name (str, optional): name of the parameter in the module
            Default: ``None``
    Returns:
        ``True`` if :attr:`module` has a parametrization for the parameter named :attr:`tensor_name`,
        or if it has any parametrization when :attr:`tensor_name` is ``None``;
        otherwise ``False``
    """
def remove_parametrizations(module: Module, tensor_name: str, leave_parametrized: bool = True) -> Module:
    """Remove the parametrizations on a tensor in a module.

    - If ``leave_parametrized=True``, ``module[tensor_name]`` will be set to
      its current output. In this case, the parametrization shall not change the ``dtype``
      of the tensor.
    - If ``leave_parametrized=False``, ``module[tensor_name]`` will be set to
      the unparametrised tensor in ``module.parametrizations[tensor_name].original``.
      This is only possible when the parametrization depends on just one tensor.

    Args:
        module (nn.Module): module from which remove the parametrization
        tensor_name (str): name of the parametrization to be removed
        leave_parametrized (bool, optional): leave the attribute :attr:`tensor_name` parametrized.
            Default: ``True``

    Returns:
        Module: module

    Raises:
        ValueError: if ``module[tensor_name]`` is not parametrized
        ValueError: if ``leave_parametrized=False`` and the parametrization depends on several tensors
    """
def type_before_parametrizations(module: Module) -> type:
    """Return the module type before parametrizations were applied and if not, then it returns the module type.

    Args:
        module (nn.Module): module to get type of
    """
def transfer_parametrizations_and_params(from_module: Module, to_module: Module, tensor_name: str | None = None) -> Module:
    """Transfer parametrizations and the parameters they parametrize from :attr:`from_module` to :attr:`to_module`.

    If :attr:`tensor_name` is specified, only transfers the specified parameter, otherwise
    transfers all parametrized parameters. If those parameters do not exist in to_module, it will create them.
    Does nothing if from_module is not parametrized.

    Args:
        from_module (nn.Module): module to transfer from
        to_module (nn.Module): module to transfer to
        tensor_name (str, optional): parameter to transfer

    Returns:
        Module: to_module
    """
