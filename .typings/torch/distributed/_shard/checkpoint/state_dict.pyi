import contextlib
import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass, field
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed.tensor import DTensor
from torch.nn.modules.module import _IncompatibleKeys
from typing import Callable, no_type_check

__all__ = ['FQNS_T', 'PrimitiveType', 'ValueType', 'DictValueType', 'ListDictValueType', 'OptimizerStateType', 'StateDictOptions', 'get_model_state_dict', 'get_optimizer_state_dict', 'get_state_dict', 'set_model_state_dict', 'set_optimizer_state_dict', 'set_state_dict']

FQNS_T = set[str]
PrimitiveType = DTensor | ShardedTensor | torch.Tensor | int | float | str
ValueType: Incomplete
DictValueType = dict[str, ValueType]
ListDictValueType = list[DictValueType]
OptimizerStateType = dict[str, DictValueType | ListDictValueType]

@dataclass
class StateDictOptions:
    """
    This dataclass specifies how get_state_dict/set_state_dict will work.

    - ``full_state_dict``: if this is set to True, all the tensors in the
      returned state_dict will be gathered. No ShardedTensor and DTensor
      will be in the returned state_dict.

    - ``cpu_offload``: offload all the tensors to cpu. To prevent CPU OOM, if
      ``full_state_dict`` is also true, then only the rank0 will get the
      state_dict and all other ranks will get empty state_dict.

    - ``ignore_frozen_params``: if the value is True, the returned state_dict
      won't contain any frozen parameters -- the ``requires_grad`` is False.
      The default value is False.

    - ``keep_submodule_prefixes`` (deprecated): when ``submodules`` is not None, this option
      indicates whether to keep the submodule prefixes from the state_dict keys.
      or example, if the submodule is ``module.pretrain`` and the full FQN of
      the parameter is ``pretrain.layer1.weight`` of the param. When this option
      is True, the parameter's key in the returned state_dict will be
      ``pretrain.layer1.weight``. If the options is False, the key will be
      ``layer1.weight``.
      Note that if ``keep_submodule_prefixes`` is False, there may be conflicted
      FQNs, hence there should be only one submodule in ``submodules``.

    - ``strict``: the ``strict`` option when ``set_state_dict`` calls
      model.load_state_dict().

    - ``broadcast_from_rank0``: when the option is True, rank0 should receive a
       full state_dict and will broadcast the tensors in the state_dict/
       optim_state_dict one by one to other ranks. Other ranks will receive
       the tensors and shard according to the local shards in the model and
       optimizer. ``full_state_dict`` must be set to True when using this option.
       This option currently only supports DTensor, not the legacy ShardedTensor.
    """
    full_state_dict: bool = ...
    cpu_offload: bool = ...
    ignore_frozen_params: bool = ...
    keep_submodule_prefixes: bool = ...
    strict: bool = ...
    broadcast_from_rank0: bool = ...
    flatten_optimizer_state_dict: bool = ...
    dsd_fqn_modifiers: str = ...

@dataclass
class _StateDictInfo(StateDictOptions):
    fqn_param_mapping: dict[str | torch.Tensor, FQNS_T | torch.Tensor] = field(default_factory=dict)
    shared_params_mapping: dict[str | torch.Tensor, FQNS_T | torch.Tensor] = field(default_factory=dict)
    submodule_prefixes: set[str] = field(default_factory=set)
    handle_model: bool = ...
    handle_optim: bool = ...
    fsdp_context: Callable = ...
    fsdp_modules: list[nn.Module] = field(default_factory=list)

class _EXTRA_STATE: ...

def get_model_state_dict(model: nn.Module, *, submodules: set[nn.Module] | None = None, options: StateDictOptions | None = None) -> dict[str, ValueType]:
    """
    Return the model state_dict of ``model``.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``model``.

    :rtype: typing.Dict[str, ValueType]
    """
def get_optimizer_state_dict(model: nn.Module, optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer], *, submodules: set[nn.Module] | None = None, options: StateDictOptions | None = None) -> OptimizerStateType:
    """
    Return the combined state_dict for optimizers.

    See ``get_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        The state_dict for ``optimizers``.

    :rtype: OptimizerStateType
    """
def get_state_dict(model: nn.Module, optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer], *, submodules: set[nn.Module] | None = None, options: StateDictOptions | None = None) -> tuple[dict[str, ValueType], OptimizerStateType]:
    """
    Return the model state_dict and optimizers state_dict.

    ``get_state_dict`` can process any module that is parallelized by PyTorch
    FSDP/fully_shard, DDP/replicate, tensor_parallel/parallelize_module, and any
    combination of these parallelisms. The main functions of ``get_state_dict``
    are: 1.) returning a model and optimizer state_dict that can be resharded
    with a different number of trainers and/or different parallelisms.
    2.) hiding the parallelism-specific state_dict APIs. Users don't have to call
    these APIs.
    3.) sanity checking the result state_dict.

    The keys of the result state dictionary are the canonical FQNs (Fully
    Qualified Names).  A canonical FQN refers to the FQN based on a parameter's
    position in an nn.Module hierarchy. More specifically, a canonical FQN to a
    parameter is the FQN returned by ``module.named_parameters()`` or
    ``module.named_buffers()`` when the module is not distributed by any
    parallelisms. Since the optimizer internally uses parameter IDs to represent
    a parameter, there will be a conversion from the parameter IDs to the
    canonical FQNs when calling this API.

    ``get_state_dict`` can also process a module that is not parallelized. In
    such a case, ``get_state_dict`` only performs one function -- converting the
    optimizer parameter IDs to the canonical FQNs.

    Example:
        >>> # xdoctest: +SKIP
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> from torch.nn.parallel import DistributedDataParallel as DDP
        >>> from torch.distributed.checkpoint.state_dict import get_state_dict

        >>> fsdp_model = FSDP(copy.deepcopy(model))
        >>> fsdp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> ddp_model = DDP(copy.deepcopy(model))
        >>> ddp_optim = torch.optim.Adam(model.parameters(), lr=1e-3)


        >>> ddp_state_dict, ddp_optim_state_dict = get_state_dict(ddp_model, ddp_optim)
        >>> fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(
        ...     fsdp_model, fsdp_optim
        ... )

        >>> # if we simply call ddp_model.state_dict() and fsdp_model.state_dict(),
        >>> # the asserts will fail.
        >>> assert ddp_state_dict == fsdp_state_dict
        >>> assert ddp_optim_state == fsdp_optim_state_dict


    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[None, Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        submodules (deprecated): Optional[set[nn.Module]]: only return the model parameters
            that belong to the submodules.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be returned. See
            `StateDictOptions` for the details.

    Returns:
        ``Tuple`` that contain model state_dict and optimizer state_dict.

    :rtype: typing.Tuple[typing.Dict[str, ValueType], OptimizerStateType]
    """
def set_model_state_dict(model: nn.Module, model_state_dict: dict[str, ValueType], *, options: StateDictOptions | None = None) -> _IncompatibleKeys:
    """Load the model state_dict.

    The counterpart of ``get_model_state_dict`` to set the state_dict to the
    model. See ``set_state_dict`` for the detail usage.

    Args:
        model (nn.Module): the nn.Module to the model.
        model_state_dict: (Dict[str, ValueType]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys
            * **unexpected_keys** is a list of str containing the unexpected keys

    :type model_state_dict: typing.Dict[str, ValueType]
    """
def set_optimizer_state_dict(model: nn.Module, optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer], optim_state_dict: OptimizerStateType, *, options: StateDictOptions | None = None) -> None:
    """Load the optimizers state_dict.

    The counterpart of ``get_optimizer_state_dict`` to set the state_dict to the
    optimizers. See ``set_state_dict`` for the detail usage.

    WARN: ``set_optimizer_state_dict`` can only be called before ``backward()`` or after
        ``step()`` is called on the optimizers. Otherwise, the optimizer states won't be
        initialized correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        None

    :type optim_state_dict: typing.OptimizerStateType
    """
def set_state_dict(model: nn.Module, optimizers: torch.optim.Optimizer | Iterable[torch.optim.Optimizer], *, model_state_dict: dict[str, ValueType], optim_state_dict: OptimizerStateType, options: StateDictOptions | None = None) -> _IncompatibleKeys:
    """Load the model state_dict and optimizers state_dict.

    The counterpart of ``get_state_dict`` to set the state_dict to the model and
    optimizers.  The given ``model_state_dict`` and ``optim_state_dict`` do not
    have to be returned by ``get_state_dict`` but must meet the following
    requirements: 1) all FQNs are canonical FQNs as defined in ``get_state_dict``,
    2) if a tensor is sharded, it must be either a ShardedTensor or DTensor,
    3) optimizer state_dict cannot contain the parameter IDs; the keys should be
    the canonical FQNs.

    WARN: ``set_state_dict`` can only be called before ``backward()`` or after ``step()``
        is called on the optimizers. Otherwise, the optimizer states won't be initialized
        correctly.

    Args:
        model (nn.Module): the nn.Module to the model.
        optimizers (Union[Optimizer, Iterable[Optimizer]]):
            The optimizers that are used to optimize ``model``.
        model_state_dict: (Union[Dict[nn.Module, Dict[str, ValueType]], Dict[str, ValueType]]):
           the model state_dict to load. If the key of the ``model_state_dict``
           is nn.Module, the key is a submodule of ``model`` and the value should
           be the state_dict of the submodule. When loading the state_dict,
           the prefix of the submodule will be append to the state_dict.
        optim_state_dict: OptimizerStateType:
            the optimizer state_dict to load.
        options (StateDictOptions): the options to control how
            model state_dict and optimizer state_dict should be loaded. See
            `StateDictOptions` for the details.

    Returns:
        ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
            * **missing_keys** is a list of str containing the missing keys of the model state_dict.
            * **unexpected_keys** is a list of str containing the unexpected keys of the model state_dict.

    :type model_state_dict: typing.Dict[str, ValueType]
    :type optim_state_dict: typing.OptimizerStateType
    """
