import abc
from _typeshed import Incomplete
from torch import nn
from typing import Any

__all__ = ['BaseSparsifier']

class BaseSparsifier(abc.ABC, metaclass=abc.ABCMeta):
    '''Base class for all sparsifiers.

    Abstract methods that need to be implemented:

    - update_mask: Function to compute a new mask for all keys in the
        `groups`.

    Args:
        - model [nn.Module]: model to configure. The model itself is not saved
            but used for the state_dict saving / loading.
        - config [list]: configuration elements should be a dict map that includes
            `tensor_fqn` of tensors to sparsify
        - defaults [dict]: default configurations will be attached to the
            configuration. Only the keys that don\'t exist in the `config` will
            be updated.

    Example::

        >>> # xdoctest: +SKIP("Can\'t instantiate abstract class BaseSparsifier with abstract method update_mask")
        >>> config = [{\'tensor_fqn\': \'layer1.weight\', \'tensor_fqn\': \'linear2.weight2\', \'sparsity_level\': 0.5}]
        >>> defaults = {\'sparsity_level\': 0.7}
        >>> # model.layer1.weight will have `sparsity_level` = 0.7 (getting default)
        >>> sparsifier = BaseSparsifier(config, defaults)
    '''
    defaults: dict[str, Any]
    state: dict[str, dict]
    groups: list[dict[str, Any]]
    enable_mask_update: bool
    def __init__(self, defaults: dict[str, Any] | None = None) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None: ...
    def __repr__(self) -> str: ...
    def state_dict(self) -> dict[str, Any]:
        '''Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - current state of the sparsification.
        * groups - a list containing all sparsity configuration groups
            with the key \'tensor_fqn\' specifying the path to the sparsified tensor within a model

        TODO: Need a clean way of loading the state of the "prepared" module
        '''
    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True): ...
    config: Incomplete
    def make_config_from_model(self, model: nn.Module, SUPPORTED_MODULES: set[type[nn.Linear]] = ...) -> None: ...
    model: Incomplete
    def prepare(self, model, config) -> None:
        """Prepares a model, by adding the parametrizations.

        Note::

            The model is modified inplace. If you need to preserve the original
            model, use copy.deepcopy.
        """
    def _prepare(self, *args, **kwargs) -> None:
        """Adds mask parametrization to the layer weight"""
    def squash_mask(self, params_to_keep: tuple[str, ...] | None = None, params_to_keep_per_layer: dict[str, tuple[str, ...]] | None = None, *args, **kwargs):
        '''Squashes the sparse masks into the appropriate tensors.

        If either the `params_to_keep` or `params_to_keep_per_layer` is set,
        the module will have a `sparse_params` dict attached to it.

        Args:
            params_to_keep: List of keys to save in the module or a dict
                            representing the modules and keys that will have
                            sparsity parameters saved
            params_to_keep_per_layer: Dict to specify the params that should be
                            saved for specific layers. The keys in the dict
                            should be the module fqn, while the values should
                            be a list of strings with the names of the variables
                            to save in the `sparse_params`

        Examples:
            >>> # xdoctest: +SKIP("locals are undefined")
            >>> # Don\'t save any sparse params
            >>> sparsifier.squash_mask()
            >>> hasattr(model.submodule1, "sparse_params")
            False

            >>> # Keep sparse params per layer
            >>> sparsifier.squash_mask(
            ...     params_to_keep_per_layer={
            ...         "submodule1.linear1": ("foo", "bar"),
            ...         "submodule2.linear42": ("baz",),
            ...     }
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {\'foo\': 42, \'bar\': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {\'baz\': 0.1}

            >>> # Keep sparse params for all layers
            >>> sparsifier.squash_mask(params_to_keep=("foo", "bar"))
            >>> print(model.submodule1.linear1.sparse_params)
            {\'foo\': 42, \'bar\': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {\'foo\': 42, \'bar\': 24}

            >>> # Keep some sparse params for all layers, and specific ones for
            >>> # some other layers
            >>> sparsifier.squash_mask(
            ...     params_to_keep=("foo", "bar"),
            ...     params_to_keep_per_layer={"submodule2.linear42": ("baz",)},
            ... )
            >>> print(model.submodule1.linear1.sparse_params)
            {\'foo\': 42, \'bar\': 24}
            >>> print(model.submodule2.linear42.sparse_params)
            {\'foo\': 42, \'bar\': 24, \'baz\': 0.1}
        '''
    def convert(self, module: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]] | None = None, inplace: bool = False, parameterization: type[nn.Module] = ...):
        """Converts submodules in input module to a different module according to `mapping`
        by calling `from_dense` method on the target module class
        Args:
            module: input module
            mapping: a dictionary that maps from source module type to target
                module type, can be overwritten to allow swapping user defined
                Modules
            inplace: carry out model transformations in-place, the original module
                is mutated
        """
    def step(self, use_path: bool = True) -> None: ...
    @abc.abstractmethod
    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs): ...
