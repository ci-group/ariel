import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Collection, Mapping
from torch import optim as optim
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from typing import Any, Callable, overload

__all__: list[str]
logger: Incomplete

class _NamedOptimizer(optim.Optimizer):
    '''
    ``_NamedOptimizer`` takes a dict of parameters and exposes ``state_dict`` by parameter key.

    We replace the original key (number) in an optim to the
    fully qualified name (FQN) string. User can initialize the optim as they
    initialize a PyTorch optim, the only difference is that they also need to
    pass in the FQN of each parameters.

    Args:
        named_parameters (Mapping[str, Union[torch.Tensor, ShardedTensor]]):
            Mapping from FQN to parameter.
        optimizer_class (optim.Optimizer):
            The class of optimizer to instantiate.
        param_groups (Collection[Mapping[str, Any]]):
            `param_groups` to pass to optimizer if specified.
            The key of the inner map needs to be FQNs.
            Default: None
        module (nn.Module): the module whose parameters to updated
            by the optimizer.
        args: arguments to pass to the optimizer constructor.
        kwargs: arguments to pass to the optimizer constructor.

    Example::
        >>> # xdoctest: +SKIP("distributed")
        >>> from torch import optim
        >>> from torch.distributed.optim import _NamedOptimizer
        >>>
        >>> # Define the named optimizer.
        >>> m = Model(...)
        >>> named_optim = _NamedOptimizer(m.named_parameters(), optim.SGD)
        >>> # Forward pass + backward pass.
        >>> named_optim.step()
        >>> ...
        >>> # Call state_dict for the named optimizer returns a FQN state_dict.
        >>> named_optim.state_dict()

    Warning: This API is still in development and subject to change.

    TODO: Add tutorial for _NamedOptimizer.
    TODO: Add documentation in the docstring for the public attributes
          like self.param_groups and self.named_parameters.
    '''
    param_groups: Collection[Mapping[str, Any]]
    named_parameters: Incomplete
    _optimizer: Incomplete
    module: Incomplete
    ordered_param_keys: Incomplete
    def __init__(self, named_parameters: Mapping[str, torch.Tensor | ShardedTensor], optimizer_class: optim.Optimizer, param_groups: Collection[Mapping[str, Any]] | None = None, module: nn.Module | None = None, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> None: ...
    def _param_groups_check(self) -> None: ...
    def state_dict(self) -> dict[str, Any]:
        """
        Return the ``state_dict`` of the optimizer.

        Instead of using number to index
        parameters, we will use module fully qualified name (FQN) as the key.
        """
    @overload
    def step(self, closure: None = None) -> None: ...
    @overload
    def step(self, closure: Callable[[], float]) -> float: ...
    @property
    def state(self) -> Mapping[torch.Tensor, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Define the default behavior to load a state_dict for ``_NamedOptimizer``.

        Sample Code
        ```
            my_model = MyModule()
            optimizer = _NamedOptimizer(my_model.named_parameters(), Adagrad)
            ...

            optim_state_dict = optimizer.state_dict()
            ...
            ...

            optimizer.load_state_dict(optim_state_dict)
            ...
        ```
        Args:
            state_dict (dict[str, Any]) : A ``state_dict`` to load into the optimizer.
                Note that this state dict update is performed in place.

        .. note:: PyTorch is using lazy init to initialize the optim states.
            So it is possible that there is no optim state when user call
            ``load_state_dict`` and for ``_NamedOptimizer`` we make it stricter
            that users can only call ``load_state_dict`` after the state is initialized.
            By doing this, we can validate the optim ``state_dict`` to be loaded.
        """
    def add_param_group(self, param_group: Mapping[str, Any]) -> None:
        """
        Add a param group to the :class:`_NamedOptimizer` s `param_groups`.

        Warning: This API is still in development and subject to change.
        """
    def init_state(self) -> None:
        """
        Run a dummy optimizer step, which allows to initialize optimizer state because we do lazy init for most optimizers.

        This allows doing in-place loading of optimizer state from a checkpoint.
        """
    def _pre_load_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]: ...
    def _post_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]: ...

def _gen_param_group_key(param_keys: list[str]) -> str:
    """Concatenate all param keys as a unique identifier for one param group."""
