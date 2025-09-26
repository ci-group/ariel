import torch.nn as nn
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state as _get_module_fsdp_state, _override_module_mixed_precision as _override_module_mixed_precision
from torch.distributed.fsdp.wrap import _Policy as _Policy, _construct_wrap_fn as _construct_wrap_fn, _or_policy as _or_policy, _post_order_apply as _post_order_apply, _recursive_wrap as _recursive_wrap, _run_mixed_precision_override_policy as _run_mixed_precision_override_policy, _wrap_module_cls_individually as _wrap_module_cls_individually
from typing import Any, Callable

def _auto_wrap(root_module: nn.Module, policy: Callable | _Policy, ignored_modules: set[nn.Module], ignored_params: set[nn.Parameter], root_kwargs: dict[str, Any], fsdp_fn: Callable):
    """
    Auto wraps modules in ``root_module`` 's tree according to ``policy``
    following a post-order traversal.

    Precondition: ``root_kwargs`` should contain all arguments except
    ``module``. This function accepts the kwargs dict directly since it gets
    forwarded into the post-order traversal function.
    """
def _check_nested_wrapping(root_module: nn.Module): ...
def _warn_on_overridden_mixed_precision(overridden_module_classes: set[type[nn.Module]]): ...
def _validate_frozen_params(root_module: nn.Module, modules_to_wrap: set[nn.Module], ignored_params: set[nn.Parameter], use_orig_params: bool):
    """
    This checks that, given ``modules_to_wrap``, each module would manage
    parameters that are uniformly frozen or non-frozen. This uniformity
    requirement is strict for ``use_orig_params=False`` (hard error) and highly
    recommended for ``use_orig_params=True`` (user warning).
    """
def _get_post_order_named_modules(root_module: nn.Module) -> list[tuple[str, nn.Module]]:
    """
    This returns the named modules following a post-order traversal, which is a
    valid reverse topological sort. We achieve this using the reverse of a
    stack-based DFS order instead of reversing ``root_module.named_modules()``
    since the former gives the modules in registration order at each level in
    the module tree (as opposed to the reverse), which allows us to error/warn
    on the first registered module that violates the condition.

    For example, consider the following module structure:
        M(
          S1(),
          S2(
            SS1(),
            SS2(),
          ),
          S3(),
        )
    The reverse DFS order is [S1, SS1, SS2, S2, S3, M], while the reverse
    ``named_modules()`` order is [S3, SS2, SS1, S2, S1, M].
    """
def _get_managed_param_to_fqn(module_to_wrap: nn.Module, ignored_params: set[nn.Parameter], visited_modules: set[nn.Module], root_prefix: str) -> dict[nn.Parameter, str]:
    """
    This returns a dict that maps managed parameter to its FQN for the given
    ``module_to_wrap``. The dict's keys are exactly the parameters that would
    be managed by the module, where this is achieved by calling this function
    on the modules to wrap in reverse topological order, destructively updating
    ``visited_modules``, and not traversing into those modules. The FQNs are
    prefixed from the root (via ``root_prefix``) to be more informative.

    NOTE: This function is meant to be called pre-wrapping and iteratively in
    reverse topological order to cover the full module tree. This differs from
    the ``_get_param_to_fqn()`` function meant to be called post-wrapping and
    on the full module tree in one shot. Given those differences, we do not try
    to unify the two.
    """
