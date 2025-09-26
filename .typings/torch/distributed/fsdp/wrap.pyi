import abc
import contextlib
import torch.nn as nn
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any, Callable

__all__ = ['always_wrap_policy', 'lambda_auto_wrap_policy', 'transformer_auto_wrap_policy', 'size_based_auto_wrap_policy', 'enable_wrap', 'wrap', 'CustomPolicy', 'ModuleWrapPolicy']

def always_wrap_policy(*args, **kwargs) -> bool:
    """
    A simple recursive wrap policy that always returns ``True``. This means
    that every submodule is wrapped by the wrapper class in
    :func:`_recursive_wrap`.
    """

class _Policy(ABC, metaclass=abc.ABCMeta):
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """
    @abstractmethod
    def _run_policy(self, root_module: nn.Module, ignored_modules: set[nn.Module], root_kwargs: dict[str, Any]) -> dict[nn.Module, dict[str, Any]]:
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """

class ModuleWrapPolicy(_Policy):
    """
    This policy applies to every module of the specified module classes,
    passing in the kwargs given to the root.
    """
    _module_classes: Incomplete
    _module_classes_str: Incomplete
    def __init__(self, module_classes: Iterable[type[nn.Module]]) -> None: ...
    def _run_policy(self, root_module: nn.Module, ignored_modules: set[nn.Module], root_kwargs: dict[str, Any]) -> dict[nn.Module, dict[str, Any]]: ...
    def __call__(self, module, recurse, *args, **kwargs): ...
    def __repr__(self) -> str: ...

class CustomPolicy(_Policy):
    '''
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    - If the function returns ``True``, then the module has the API applied
      with the root\'s kwargs.
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root\'s kwargs.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> model = init_transformer_model(...)
        >>> def lambda_fn(module: nn.Module):
        >>>     if module is model.lm_head:
        >>>         return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
        >>>     elif isinstance(module, TransformerBlock):
        >>>         return True
        >>>     return False
        >>> policy = CustomPolicy(lambda_fn)
        >>> fsdp_model = FSDP(model, auto_wrap_policy=policy)
    '''
    _lambda_fn: Incomplete
    def __init__(self, lambda_fn: Callable[[nn.Module], bool | dict[str, Any]]) -> None: ...
    def _run_policy(self, root_module: nn.Module, ignored_modules: set[nn.Module], root_kwargs: dict[str, Any]) -> dict[nn.Module, dict[str, Any]]: ...

def lambda_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, lambda_fn: Callable) -> bool:
    """
    A convenient auto wrap policy to wrap submodules based on an arbitrary user
    function. If `lambda_fn(submodule) == True``, the submodule will be wrapped as
    a `wrapper_cls` unit.

    Return if a module should be wrapped during auto wrapping.

    The first three parameters are required by :func:`_recursive_wrap`.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        lambda_fn (Callable[[nn.Module], bool]): If this returns ``True``, then
            this module will be wrapped.
    """
def transformer_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, transformer_layer_cls: set[type[nn.Module]]) -> bool:
    """
    See :func:`_module_wrap_policy`, where ``transformer_layer_cls`` is the
    same as ``module_classes``. Note that shared parameters must be wrapped in
    the same FSDP instance, so this auto wrap policy can help wrap shared
    embeddings into the same FSDP instance for transformer models.
    """
def size_based_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, min_num_params: int = ..., force_leaf_modules: set[type[nn.Module]] | None = None, exclude_wrap_modules: set[type[nn.Module]] | None = None) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Optional[set[type[nn.Module]]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Optional[set[type[nn.Module]]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
@contextlib.contextmanager
def enable_wrap(*, wrapper_cls: Any, **wrapper_kwargs: Any) -> Generator[None, None, None]:
    """
    Context manager to wrap modules using a wrapper.

    Useful for when you'd like to apply the same configuration arguments to all
    child modules that you wrap. A particularly important use case is wrapping
    large layers so that they get sharded (in-place) during initialization, to
    avoid running out of system memory. Large layers can indicate that they
    should be sharded via the ``wrap`` annotation and this context manager can
    provide the exact configuration for these nested instances.

    Usage::

        with enable_wrap(wrapper_cls, **params):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        wrapper_cls:
            Class that `wrap` annotation will `wrap` modules with, such as
            `FullyShardedDataParallel`.
        **wrapper_kwargs:
            Configuration settings that will be passed to all ``wrap``
            instances inside the context
    """
def wrap(module: nn.Module, **wrap_overrides: Any) -> nn.Module:
    """
    Annotate that a module should be wrapped. Annotated modules will only be
    wrapped if inside of an :func:`enable_wrap` context manager. This allows
    a module to be initialized both with and without a wrapper without code
    change.

    The class that this function wraps the passed in ``nn.Module`` with is the
    passed in ``wrapper_cls`` argument into ``enable_wrap``. Both
    ``enable_wrap`` and ``wrap`` can take in kwargs specifying how to construct
    the ``wrapper_cls`` instance. In the case of duplicate kwargs in
    ``enable_wrap`` and ``wrap``, the argument passed into ``wrap`` will be
    respected.

    Usage::

        with enable_wrap(wrapper_cls=FSDP, **fsdp_config):
            # Wraps layer in FSDP by default if within context
            self.l1 = wrap(torch.nn.Linear(5, 5))

    Args:
        module (nn.Module): module to wrap (if in :func:`enable_wrap` context)
        **wrap_overrides: configuration overrides that will take priority over
            the values provided by the :func:`enable_wrap` context
    """

class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """
    in_autowrap_context: bool
    wrapper_cls: Callable | None
    kwargs: dict[str, Any]
    def __init__(self, **kwargs: dict[str, Any]) -> None: ...
    @staticmethod
    def enable_autowrap_context(kwargs: Any) -> None: ...
    @staticmethod
    def disable_autowrap_context() -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None: ...
