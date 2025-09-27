import torch
from _typeshed import Incomplete
from collections.abc import Iterable as Iterable
from typing import no_type_check

__all__: list[str]
param_to_optim_hook_handle_map: Incomplete
param_to_acc_grad_map: Incomplete

@no_type_check
def _apply_optimizer_in_backward(optimizer_class, params, optimizer_kwargs, register_hook: bool = True) -> None:
    '''
    Upon ``backward()``, the optimizer specified for each parameter will fire after
    the gradient has been accumulated into the parameter.

    Note - gradients for these parameters will be set to None after ``backward()``.
    This means that any other optimizer not specified via `_apply_optimizer_in_backward`
    over this parameter will be a no-op.

    Args:
        optimizer_class: (Type[torch.optim.Optimizer]): Optimizer to apply to parameter
        params: (Iterator[nn.Parameter]): parameters to apply optimizer state to
        optimizer_kwargs: (Dict[str, Any]): kwargs to pass to optimizer constructor
        register_hook: (bool): whether to register a hook that runs the optimizer
            after gradient for this parameter is accumulated. This is the default
            way that optimizer in backward is implemented, but specific use cases
            (such as DDP) may wish to override this to implement custom behavior.
            (Default = True)

    Example::
        params_generator = model.parameters()
        param_1 = next(params_generator)
        remainder_params = list(params_generator)

        apply_optimizer_in_backward(torch.optim.SGD, [param_1], {"lr": 0.02})
        apply_optimizer_in_backward(torch.optim.Adam, remainder_params, {"lr": 0.04})

        model(...).sum().backward()  # after backward, parameters will already
        # have their registered optimizer(s) applied.

    '''
def _get_in_backward_optimizers(module: torch.nn.Module) -> list[torch.optim.Optimizer]:
    '''
    Return a list of in-backward optimizers applied to ``module``\'s parameters. Note that these
    optimizers are not intended to directly have their ``step`` or ``zero_grad`` methods called
    by the user and are intended to be used for things like checkpointing.

    Args:
        module: (torch.nn.Module): model to retrieve in-backward optimizers for

    Returns:
        List[torch.optim.Optimizer]: the in-backward optimizers.

    Example::
        _apply_optimizer_in_backward(torch.optim.SGD, model.parameters(), {"lr": 0.01})
        optims = _get_optimizers_in_backward(model)
    '''
