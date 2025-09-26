import abc
from _typeshed import Incomplete
from torch import nn
from torch.ao.pruning.sparsifier import base_sparsifier
from typing import Any

__all__ = ['BaseDataSparsifier']

class _Container(nn.Module): ...

class BaseDataSparsifier(base_sparsifier.BaseSparsifier, metaclass=abc.ABCMeta):
    """
    Base Data Sparsifier class for all Data sparsifiers.
    The abstract class accepts raw torch tensors / embedding / embedding bags (refer to SUPPORTED_TYPES above)
    to prepare for sparsification.
    In this case, mask (and parametrizations) is owned by the class and not by the user.
    Specifically, the container object inside the class maintains the mask and parametrizations of the input data

    Args:
        data_list (list of tuples)
            list of (name, data) tuples to sparsify. Lookup SUPPORTED_TYPES
            for type of data. Internally, a container module handles the data sparsification.

        defaults (dict)
            default configurations will be attached to the
            configuration. Only the keys that don't exist in the `config` will
            be updated.
    Example::
        >>> # xdoctest: +SKIP
        >>> data_list = [('tensor_1', torch.randn(3,3)), ('tensor_2', torch.randn(4,4))]
        >>> defaults = {'sparsity_level': 0.7}
        >>> sparsifier = DerivedDataSparsifier(data_list = data_list, **defaults) # Some sparsifier that inherits BaseDataSparsifier
        >>> new_tensor_to_add = {'name': 'tensor_3', 'data': torch.randn(5,5), 'sparsity_level': 0.3}
        >>> sparsifier.add_data(**new_tensor_to_add)
        >>> # tensor_1 and tensor_2 will have sparsity_level of 0.7 but tensor_3 will have sparsity_level=0.3
    """
    _container: Incomplete
    data_groups: dict[str, dict]
    def __init__(self, data_list: list[tuple[str, Any]] | None = None, **defaults) -> None: ...
    def prepare(self, model, config) -> None: ...
    def _extract_weight(self, data): ...
    def add_data(self, name: str, data, reuse_mask: bool = True, **config):
        """Configures and parametrizes the internal container model with name and data.

        **Note**:
            1. If the data with name already exists, it replaces the data.
            2. While replacing, the old mask is reused when `reuse_mask=True`
            3. If `reuse_mask=True`, then the replacing data needs to have the same shape as that of old data.
            4. By default, the config of the replaced data is used as config for the replacing data, unless something
               is specified in the config dictionary.
        """
    def get_data(self, name: str, return_original: bool = True):
        """Returns weight tensor (or data)
        Args:
            - name: name of the data to be returned
            - return_original returns weight tensor without applying parametrization if True
                else - returns the sparsified version (parametrized)
        """
    def _convert_mask(self, states, sparse_coo: bool = True):
        """Converts the mask to sparse coo or dense tensors depending on the `sparse_coo` argument."""
    def state_dict(self):
        """Returns the state of the optimizer as a :class:`dict`.

        It contains:
        * state - contains name -> mask mapping.
        * data_groups - a list containing all sparsity configuration groups
            with the key name specifying the name of the data
        * container_state_dict - the state dictionary of the internal
            container model used for sparsification
        """
    def _load_container_from_state(self, states, data_groups, container_state_dict) -> None:
        """This restores the state of the container specifically based on the data present in state and data_groups
        If the data was parametrized, then the data would be added to the container and then parametrized,
        else it would just add the attribute the container.
        """
    def load_state_dict(self, state_dict, strict: bool = True) -> None:
        """The load_state_dict() restores the state of the sparsifier based on the state_dict

        Args:
        * state_dict - the dictionary that to which the current sparsifier needs to be restored to
        * strict - If True - the sparsifier is reset and is restored exactly to the state in state_dict.
            If False - the current sparsifier is not reset before loading the state_dict i.e. data added
            before loading the state_dict is not erased.
        """
    def __setstate__(self, state) -> None: ...
    def __getstate__(self): ...
    def __repr__(self) -> str: ...
    def get_mask(self, name: str): ...
    def squash_mask(self, *args, leave_parametrized: bool = True, names=None, **kwargs) -> None:
        """Squashes the sparse masks into the appropriate tensors. Also, accepts list of strings
        to squash mask for. If none, squashes mask for all the keys
        kwargs:
            * names: list of strings to squash mask for
            * sparsified: if true - applies the mask before squashing
                          if false - does not apply the mask before squashing
        """
    def step(self) -> None: ...
    @abc.abstractmethod
    def update_mask(self, name, data, **kwargs): ...
    def _delete_data(self, name) -> None:
        """Detaches some data from the sparsifier.

        Args:
            name (str)
                Name of the data to be removed from the sparsifier

        Note:
            Currently private. Kind of used as a helper function when replacing data of the same name
        """
