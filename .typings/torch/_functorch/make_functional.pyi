import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Sequence
from torch import Tensor as Tensor
from torch.nn.utils._named_member_accessor import NamedMemberAccessor as NamedMemberAccessor
from typing import Any, Callable, NoReturn

def raise_parameter_tying_error() -> NoReturn: ...
def create_names_map(named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]], tied_named_params: dict[str, Tensor] | Iterable[tuple[str, Tensor]]) -> dict[str, list[str]]:
    """
    named_params is a dictionary of tensors: {'A': A, 'B': B}
    tied_named_params is another dictionary of tensors {'A': A, 'B': B, 'B_tied': B}
    with potentially tied (or 'duplicated') tensors

    This function creates a mapping from the names in named_params to the
    names in tied_named_params: {'A': ['A'], 'B': ['B', 'B_tied']}.
    """
def _extract_members(mod: nn.Module, named_members: Callable[..., Iterable[tuple[str, Tensor]]], subclass: Callable[[Tensor], Tensor]) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]: ...
def extract_weights(mod: nn.Module) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]:
    """
    This function removes all the Parameters from the model and
    return them as a tuple as well as their original attribute names.
    The weights must be re-loaded with `load_weights` before the model
    can be used again.
    Note that this function modifies the model in place and after this
    call, mod.parameters() will be empty.
    """
def extract_buffers(mod: nn.Module) -> tuple[tuple[Tensor, ...], tuple[str, ...], dict[str, list[str]]]: ...
def load_weights(mod: nn.Module, names: Sequence[str], params: Sequence[Tensor], as_params: bool = False) -> None:
    """
    Reload a set of weights so that `mod` can be used again to perform a forward pass.
    Note that the `params` are regular Tensors (that can have history) and so are left
    as Tensors. This means that mod.parameters() will still be empty after this call.
    """
def _swap_state(mod: nn.Module, names_map: dict[str, list[str]], elems: Iterable[Tensor]) -> list[Tensor]: ...
def load_buffers(mod: nn.Module, names: Sequence[str], buffers: Sequence[Tensor], as_params: bool = False) -> None: ...
def load_state(model: nn.Module, weights: Sequence[Tensor], weight_names: Sequence[str], buffers: Sequence[Tensor] = (), buffer_names: Sequence[str] = ()) -> nn.Module:
    """load_state(model, weights, weight_names, buffers=(), buffer_names=()) -> model

    load_state takes `weights` and `buffers` and assigns them to the model.
    This is the inverse operation of `make_functional_deprecated_v1`.
    """
def make_functional_deprecated_v1(model: nn.Module):
    """make_functional_deprecated_v1(model) -> weights, func, weight_names

    Given an nn.Module, make_functional_deprecated_v1 extracts the state (weights)
    and returns a functional version of the model, `func`. This makes
    it so that it is possible use transforms over the parameters of
    `model`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, func, _ = make_functional_deprecated_v1(model)
    func(weights, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, _, func = make_functional_deprecated_v1(model)
    grad_weights = grad(func)(weights, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """
def make_functional_with_buffers_deprecated_v1(model: nn.Module):
    """make_functional_with_buffers_deprecated_v1(model) -> weights, buffers, func, weight_names, buffer_names

    Given an nn.Module, make_functional_with_buffers_deprecated_v1 extracts the state (weights and buffers)
    and returns a functional version of the model, `func`.

    `func` can be invoked as follows:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    ```

    And here is an example of applying the grad transform:
    ```
    x = torch.randn(4, 3)
    model = nn.Linear(3, 3)
    weights, buffers, func, _, _ = make_functional_with_buffers_deprecated_v1(model)
    func(weights, buffers, (x,))
    grad_weights = grad(func)(weights, buffers, (x,))
    ```

    To put the state back into a model, use `load_state`.
    """

class FunctionalModuleWithBuffers(nn.Module):
    """
    This is the callable object returned by :func:`make_functional_with_buffers`.
    """
    stateless_model: Incomplete
    param_names: Incomplete
    buffer_names: Incomplete
    all_names_map: Incomplete
    def __init__(self, stateless_model: nn.Module, param_names: tuple[str, ...], buffer_names: tuple[str, ...], param_names_map: dict[str, list[str]], buffer_names_map: dict[str, list[str]]) -> None: ...
    @staticmethod
    def _create_from(model: nn.Module, disable_autograd_tracking: bool = False) -> tuple['FunctionalModuleWithBuffers', tuple[Tensor, ...], tuple[Tensor, ...]]: ...
    def forward(self, params: Iterable[Tensor], buffers: Iterable[Tensor], *args, **kwargs) -> Any: ...

class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """
    stateless_model: Incomplete
    param_names: Incomplete
    names_map: Incomplete
    def __init__(self, stateless_model: nn.Module, param_names: tuple[str, ...], names_map: dict[str, list[str]]) -> None: ...
    @staticmethod
    def _create_from(model: nn.Module, disable_autograd_tracking: bool = False) -> tuple['FunctionalModule', tuple[Tensor, ...]]: ...
    def forward(self, params: Iterable[Tensor], *args, **kwargs) -> Any: ...

def make_functional(model: nn.Module, disable_autograd_tracking: bool = False) -> tuple[FunctionalModule, tuple[Tensor, ...]]:
    """make_functional(model, disable_autograd_tracking=False) -> func, params

    Given a ``torch.nn.Module``, :func:`make_functional` extracts the state
    (params) and returns a functional version of the model, ``func``. This
    makes it so that it is possible use transforms over the parameters of
    ``model``.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)
        func(params, x)

    And here is an example of applying the grad transform over the parameters
    of a model.

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params = make_functional(model)

        def compute_loss(params, x, t):
            y = func(params, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, x, t)

    If the model has any buffers, please use :func:`make_functional_with_buffers` instead.

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
def make_functional_with_buffers(model: nn.Module, disable_autograd_tracking: bool = False) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """make_functional_with_buffers(model, disable_autograd_tracking=False) -> func, params, buffers

    Given a ``torch.nn.Module``, make_functional_with_buffers extracts the
    state (params and buffers) and returns a functional version of the model
    ``func`` that can be invoked like a function.

    ``func`` can be invoked as follows:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers

        x = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)
        func(params, buffers, x)

    And here is an example of applying the grad transform over the parameters
    of a model:

    .. code-block:: python

        import torch
        import torch.nn as nn
        from functorch import make_functional_with_buffers, grad

        x = torch.randn(4, 3)
        t = torch.randn(4, 3)
        model = nn.Linear(3, 3)
        func, params, buffers = make_functional_with_buffers(model)

        def compute_loss(params, buffers, x, t):
            y = func(params, buffers, x)
            return nn.functional.mse_loss(y, t)

        grad_weights = grad(compute_loss)(params, buffers, x, t)

    Args:
        model (torch.nn.Module): Input model.
        disable_autograd_tracking (bool): Flag to disable gradients tracking for output parameters.
            The returned params are unrelated to the set of params from the original model. If False (default),
            the params will have ``requires_grad=True`` on them (aka they will be trackable with regular
            PyTorch autograd), matching the requires_grad-ness of the params from the original model.
            Otherwise, the returned params will have ``requires_grad=False``. Default, False.
            If you plan on using regular PyTorch autograd (e.g., if you want to call ``.backward()`` or
            ``torch.autograd.grad()``, then set ``disable_autograd_tracking=False``.
            Otherwise, if you're only planning on using functorch's gradient transforms,
            then please set ``disable_autograd_tracking=True`` to avoid unnecessarily tracking
            history with PyTorch autograd.

    """
def transpose_stack(tuple_of_tuple_of_tensors: tuple[tuple[Tensor, ...], ...]) -> tuple[Tensor, ...]: ...
def combine_state_for_ensemble(models: Sequence[nn.Module]) -> tuple[FunctionalModuleWithBuffers, tuple[Tensor, ...], tuple[Tensor, ...]]:
    """combine_state_for_ensemble(models) -> func, params, buffers

    Prepares a list of torch.nn.Modules for ensembling with :func:`vmap`.

    Given a list of ``M`` ``nn.Modules`` of the same class, stacks all of their
    parameters and buffers together to make ``params`` and ``buffers``.
    Each parameter and buffer in the result will have an additional dimension
    of size ``M``.

    :func:`combine_state_for_ensemble` also returns ``func``, a functional
    version of one of the models in :attr:`models`. One cannot directly run
    ``func(params, buffers, *args, **kwargs)`` directly, you probably want to
    use ``vmap(func, ...)(params, buffers, *args, **kwargs)``

    Here's an example of how to ensemble over a very simple model:

    .. code-block:: python

        num_models = 5
        batch_size = 64
        in_features, out_features = 3, 3
        models = [torch.nn.Linear(in_features, out_features) for i in range(num_models)]
        data = torch.randn(batch_size, 3)

        fmodel, params, buffers = combine_state_for_ensemble(models)
        output = vmap(fmodel, (0, 0, None))(params, buffers, data)

        assert output.shape == (num_models, batch_size, out_features)

    .. warning::
        All of the modules being stacked together must be the same (except for
        the values of their parameters/buffers). For example, they should be in the
        same mode (training vs eval).

        This API is subject to change -- we're investigating better ways to
        create ensembles and would love your feedback how to improve this.
    """
def functional_init(model_class: type[nn.Module], ensemble_shape: tuple[()] | tuple[int] = (), device: torch.types.Device = 'cpu'): ...
def functional_init_with_buffers(model_class: type[nn.Module], ensemble_shape: tuple[()] | tuple[int] = (), device: torch.types.Device = 'cpu'): ...
