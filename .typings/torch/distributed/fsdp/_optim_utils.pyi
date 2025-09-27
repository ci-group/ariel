import torch
import torch.distributed as dist
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from torch.distributed._shard.sharded_tensor import ShardedTensor as ShardedTensor
from torch.distributed._state_dict_utils import _gather_state_dict as _gather_state_dict
from torch.distributed.distributed_c10d import _get_pg_default_device as _get_pg_default_device
from torch.distributed.fsdp._common_utils import _FSDPState as _FSDPState, _apply_to_modules as _apply_to_modules, _get_module_fsdp_state_if_fully_sharded_module as _get_module_fsdp_state_if_fully_sharded_module, _get_param_to_fqns as _get_param_to_fqns, _module_handle as _module_handle, _named_parameters_with_duplicates as _named_parameters_with_duplicates, clean_tensor_name as clean_tensor_name
from torch.distributed.fsdp._debug_utils import SimpleProfiler as SimpleProfiler
from torch.distributed.fsdp._flat_param import FlatParamHandle as FlatParamHandle, FlatParameter as FlatParameter
from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_dtensor as _ext_chunk_dtensor, _ext_chunk_tensor as _ext_chunk_tensor
from torch.distributed.fsdp._runtime_utils import _lazy_init as _lazy_init, _reset_flat_param_grad_info_if_needed as _reset_flat_param_grad_info_if_needed
from torch.distributed.fsdp.api import ShardingStrategy as ShardingStrategy, StateDictSettings as StateDictSettings, StateDictType as StateDictType
from torch.distributed.tensor import DTensor as DTensor, Replicate as Replicate
from torch.utils._pytree import tree_map_only as tree_map_only
from typing import Any, NamedTuple, no_type_check

logger: Incomplete

@dataclass
class FSDPParamInfo:
    state: _FSDPState
    handle: FlatParamHandle
    param_indices: dict[str, int]
    param_requires_grad: list[bool]

def sorted_items(dictionary: dict[str, Any]) -> Iterator[tuple[str, Any]]: ...

@dataclass
class _ConsolidatedOptimState:
    """
    This holds the consolidated optimizer state on the target rank. Positive-
    dimension tensor state is communicated across ranks, while zero-dimension
    tensor state and non-tensor state is taken directly from the target rank.

    PyTorch version 1.12 moved to using zero-dimension tensors for scalar
    values, but user implemented optimizers may still use float (i.e. a
    non-tensor). Thus, we support both and handle them identically.

    Attributes:
        tensor_state (Dict[str, torch.Tensor]): Mapping from positive-dimension
            tensor state name to the unsharded flat tensor representing the
            state.
        zero_dim_tensor_state (Dict[str, torch.Tensor]): Mapping from zero-
            dimension tensor state name to its value.
        non_tensor_state (Dict[str, Any]): Mapping from non-tensor state
            name to its value.
    """
    tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    zero_dim_tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    non_tensor_state: dict[str, Any] = field(default_factory=dict)

class _PosDimTensorInfo(NamedTuple):
    """
    Metadata for positive-dimension tensors used internally for
    :meth:`scatter_full_optim_state_dict`.

    Attributes:
        shape (torch.Size): Sharded tensor shape (which is equal to the
            unsharded tensor shape if the tensor is optimizer state for a
            non-FSDP parameter and is hence not sharded).
        dtype (torch.dtype): Data type of the tensor.
    """
    shape: torch.Size
    dtype: torch.dtype

class _OptimStateKey(NamedTuple):
    """
    This represents an optimizer state key that may be used commonly across
    ranks. It is based on the unflattened parameter names rather than parameter
    IDs to make it independent of each rank's own optimizer construction.
    """
    unflat_param_names: tuple[str, ...]
    is_fsdp_managed: bool

def _unflatten_optim_state(fsdp_param_info: FSDPParamInfo, flat_param_state: dict[str, Any], to_save: bool, shard_state: bool, cpu_offload: bool) -> list[dict[str, Any]]:
    '''
    Unflattens the optimizer state, consisting of the "state" part and the
    "param_groups" part. Unflattening the "state" part involves consolidating
    the state on the target rank and remapping from flattened to unflattened
    parameter IDs, and the "param_groups" part only involves remapping from
    flattened to unflattened parameter IDs.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): Entry for the flat parameter in the
            "state" part of the optimizer state dict.
        to_save (bool): Whether to save the state on this rank.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter if on the target
        rank or an empty :class:`list` otherwise. The final optimizer state
        dict will need to map these entries using the proper unflattened
        parameter IDs.
    '''
def _is_zero_dim_tensor(x: Any) -> bool: ...
def _communicate_optim_state(fsdp_param_info: FSDPParamInfo, flat_param_state: dict[str, Any]) -> _ConsolidatedOptimState:
    '''
    Communicates the optimizer state for a flat parameter across ranks. All
    ranks will hold the entire non-sharded optimizer state on GPU.

    If ``N`` is the number of tensor optimizer states in the optimizer state
    dict, then the communication complexity is 0 if ``N = 0`` and ``N + 1``
    otherwise (where the plus 1 comes from all-gathering the padding per rank).

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        flat_param_state (Dict[str, Any]): The entry in the "state" part of the
            optimizer state dict corresponding to the flat parameter.

    Returns:
        ConsolidatedOptimState: Consolidated optimizer state for the target
        flat parameter.
    '''
def _unflatten_communicated_optim_state(fsdp_param_info: FSDPParamInfo, state: _ConsolidatedOptimState, shard_state: bool) -> list[dict[str, Any]]:
    '''
    Unflattens the communicated optimizer state (given by ``tensor_state``,
    ``non_tensor_state``, and ``zero_dim_tensor_state``) for a single flat
    parameter. This should only be called on the target rank.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        state (_ConsolidatedOptimState): Consolidated optimizer state.

    Returns:
        List[Dict[str, Any]]: A :class:`list` holding the entries in the
        "state" part of the optimizer state dict corresponding to the
        unflattened parameters comprising the flat parameter. The final
        optimizer state dict will need to map these entries using the proper
        unflattened parameter IDs.
    '''
def _broadcast_processed_state(fsdp_state: _FSDPState, optim_state: dict[str, Any], group: dist.ProcessGroup | None) -> dict[str, Any]: ...
def _broadcast_state(fsdp_state: _FSDPState, state: Any, group: dist.ProcessGroup | None) -> Any: ...
def _shard_orig_param_state(fsdp_param_info: FSDPParamInfo, fqn: str, optim_state: dict[str, Any]) -> dict[str, Any]:
    """
    Shard the optimizer state for the original parameter with the name ``fqn``.
    This API should only be used when ``use_orig_params`` is True.
    """
def _flatten_optim_state_dict(optim_state_dict: dict[str, Any], model: nn.Module, use_orig_params: bool = False, optim: torch.optim.Optimizer | None = None, rank0_only: bool = False, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
    """
    Flattens the full optimizer state dict, still keying by unflattened parameter
    names.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP know how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- it is managed by other parallelism and FSDP does not
    know ho to handle/aggregate them.

    Note that ``_flatten_tensor_optim_state`` does not need ``optim`` to
    flatten/shard the state. However, NamedOptimizer and KeyedOptimizer require
    all the states even if the corresponding parameters are empty. To this end,
    ``optim`` will be used to to get the initial state of the empty parameters.
    ``optim`` should only be non-None if the ``optim` is KeyedOptimizer or
    NamedOptimizer.

    Returns:
        Dict[str, Any]: The flattened optimizer state dict.
    """
def _flatten_optim_state(fsdp_param_info: FSDPParamInfo, unflat_osd_state: dict[str, dict[str, Any]], unflat_param_names: list[str]) -> dict[str, Any]:
    '''
    Flattens the optimizer state in ``full_optim_state_dict`` for a single
    flat parameter in ``fsdp_param_info`` corresponding to the unflattened
    parameter names in ``unflat_param_names``.

    Args:
        fsdp_param_info (FSDPParamInfo): The FSDP state, the handle, and a
            mapping from FQN to original parameter index.
        unflat_osd_state (Dict[str, Dict[str, Any]]): The "state" part of the
            optimizer state dict corresponding to the unflattened parameters.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the flat parameter ``flat_param``.

    Returns:
        Dict[str, Any]: A :class:`dict` mapping state names to their values for
        a particular flat parameter. The sharded optimizer state dict\'s "state"
        part will map a key to this returned value.
    '''
def _flatten_tensor_optim_state(state_name: str, pos_dim_tensors: list[torch.Tensor], unflat_param_names: list[str], unflat_param_shapes: Sequence[torch.Size], handle: FlatParamHandle) -> torch.Tensor:
    '''
    Flattens the positive-dimension tensor optimizer state given by the values
    ``tensors`` for the state ``state_name`` for a single flat parameter
    from ``handle`` corresponding to the unflattened parameter names
    ``unflat_param_names`` and unflatted parameter shapes
    ``unflat_param_shapes``. This flattens each unflattened parameter\'s tensor
    state into one tensor.

    NOTE: We use zero tensors for any unflattened parameters without state
    since some value is required to fill those entries. This assumes that the
    zero tensor is mathematically equivalent to having no state, which is true
    for Adam\'s "exp_avg" and "exp_avg_sq" but may not be true for all
    optimizers.

    Args:
        state_name (str): Optimizer state name.
        pos_dim_tensors (List[torch.Tensor]): Positive-dimension tensor
            optimizer state values for the unflattened parameters corresponding
            to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.
        unflat_param_shapes (List[torch.Size]): Unflattened parameter shapes
            corresponding to the single flat parameter.
        handle (FlatParamHandle): The flat parameter\'s handle.

    Returns:
        torch.Tensor: A flat tensor containing the optimizer state
        corresponding to ``state_name`` constructed by concatenating the
        unflattened parameter tensor states in ``pos_dim_tensors`` (using zero
        tensors for any unflattened parameters without the state).
    '''
def _flatten_zero_dim_tensor_optim_state(state_name: str, zero_dim_tensors: list[torch.Tensor], unflat_param_names: list[str]) -> torch.Tensor:
    '''
    Flattens the zero-dimension tensor optimizer state given by the values
    ``zero_dim_tensors`` for the state ``state_name`` for a single flat
    parameter corresponding to the unflattened parameter names
    ``unflat_param_names`` by enforcing that all tensors are the same and using
    that common value.

    NOTE: The requirement that the tensors are the same across all unflattened
    parameters comprising the flat parameter is needed to maintain the
    invariant that FSDP performs the same computation as its non-sharded
    equivalent. This means that none of the unflattened parameters can be
    missing this state since imposing a value may differ from having no value.
    For example, for Adam\'s "step", no value means maximum bias correction,
    while having some positive value means less bias correction.

    Args:
        state_name (str): Optimizer state name.
        zero_dim_tensors (List[torch.Tensor]): Zero-dimension optimizer state
            for the unflattened parameters corresponding to the single
            flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        torch.Tensor: A zero-dimensional tensor giving the value of the state
        ``state_name`` for all unflattened parameters corresponding to the
        names ``unflat_param_names``.
    '''
def _flatten_non_tensor_optim_state(state_name: str, non_tensors: list[Any], unflat_param_names: list[str]) -> Any:
    """
    Flattens the non-tensor optimizer state given by the values ``non_tensors``
    for the state ``state_name`` for a single flat parameter corresponding
    to the unflattened parameter names ``unflat_param_names`` by enforcing that
    all values are the same and using that common value.

    See the note in :func:`_flatten_zero_dim_tensor_optim_state`.

    Args:
        state_name (str): Optimizer state name.
        non_tensors (List[Any]): Non-tensor optimizer state for the unflattened
            parameters corresponding to the single flat parameter.
        unflat_param_names (List[str]): A :class:`list` of unflattened
            parameter names corresponding to the single flat parameter.

    Returns:
        Any: A non-tensor giving the value of the state ``state_name`` for all
        unflattened parameters corresponding to the names
        ``unflat_param_names``.
    """
def _rekey_sharded_optim_state_dict(sharded_osd: dict[str, Any], model: nn.Module, optim: torch.optim.Optimizer, optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None, using_optim_input: bool, is_named_optimizer: bool = False) -> dict[str, Any]:
    """
    Rekeys the optimizer state dict from unflattened parameter names to flat
    parameter IDs according to the calling rank's ``optim``, which may be
    different across ranks. In particular, the unflattened parameter names are
    represented as :class:`_OptimStateKey` s.
    """
def _get_param_id_to_param_from_optim_input(model: nn.Module, optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None = None) -> dict[int, nn.Parameter]:
    """
    Constructs a mapping from parameter IDs to parameters. This may be used
    both for models with ``FlatParameter`` s and without.

    NOTE: This method is only preserved for backward compatibility. The method
    :meth:`_get_param_key_to_param` is the preferred code path that does not
    rely on ``optim_input``.

    NOTE: We critically assume that, whether the optimizer input is a list of
    parameters or a list of parameter groups, :class:`torch.optim.Optimizer`
    enumerates the parameter IDs in order. In other words, for a parameter list
    input, the parameter IDs should be in that list order, and for a parameter
    groups input, the parameter IDs should be in order within each parameter
    group and in order across parameter groups.

    Args:
        model (nn.Module): Model whose parameters are passed into the
            optimizer.
        optim_input (Optional[Union[List[Dict[str, Any]],
        Iterable[nn.Parameter]]]): Input passed into the optimizer
            representing either a :class:`list` of parameter groups or an
            iterable of parameters; if ``None``, then this method assumes the
            input was ``model.parameters()``. (Default: ``None``)

    Returns:
        List[nn.Parameter]: Mapping from parameter IDs to parameters,
        where the parameter ID is implicitly the index in the :class:`list`.
    """
def _get_flat_param_to_fqn(model: torch.nn.Module) -> dict[FlatParameter, str]:
    '''
    Constructs a mapping from ``FlatParameter`` to a cleaned (devoid of prefixes
    from wrappers) fully qualified name (FQN). Note that this FQN is "non-canonical"
    because ``FlatParameter``  s do not come from the original module but are
    registered only after FSDP has been applied. This function returns the FSDP-given
    name for the ``FlatParameter`` (usually module._flat_param) as opposed to the
    canonical FQNs returned for ``FlatParameter`` s in ``_common_utils._get_param_to_fqns(...)``).

    Consequently, this function will only return a non-empty mapping if FSDP was
    applied with ``use_orig_params=False`` as, otherwise, the original parameters
    are used within the module and there would be no ``FlatParameter`` s in the module.

    '''
def _get_param_key_to_param(optim: torch.optim.Optimizer, model: nn.Module | None = None, is_named_optimizer: bool = False, param_to_fqns: dict[nn.Parameter, list[str]] | None = None, flat_param_to_fqn: dict[FlatParameter, str] | None = None) -> dict[int | str, nn.Parameter]:
    """
    Constructs a mapping from parameter keys to parameters. For the regular
    optimizers, the keys are parameter IDs. For NamedOptimizer, the keys
    are FQNs. This API may be used both for models with ``FlatParameter`` s and
    without.
    """
def _get_param_to_param_key(optim: torch.optim.Optimizer, model: nn.Module | None = None, is_named_optimizer: bool = False, param_to_fqns: dict[nn.Parameter, list[str]] | None = None, flat_param_to_fqn: dict[FlatParameter, str] | None = None) -> dict[nn.Parameter, int | str]:
    """
    Constructs the inverse mapping of :func:`_get_param_key_to_param`. This API
    only supports the case where `optim` is a regular optimizer, not NamedOptimizer.
    So the parameter keys will be parameter ids.
    """
def _get_param_to_param_id_from_optim_input(model: nn.Module, optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None = None) -> dict[nn.Parameter, int]:
    """Constructs the inverse mapping of :func:`_get_param_id_to_param_from_optim_input`."""
def _check_missing_keys_on_rank(r0_optim_state_keys: list[_OptimStateKey], optim_state_key_to_param_key: dict[_OptimStateKey, str | int], param_key_to_param: dict[str | int, nn.Parameter], group: dist.ProcessGroup | None) -> None: ...
def _map_param_key_to_optim_keys(optim_state_dict: dict[str, Any], group: dist.ProcessGroup | None, param_key_to_param: dict[int | str, nn.Parameter], param_to_fqns: dict[nn.Parameter, list[str]], fqn_to_fsdp_param_info: dict[str, FSDPParamInfo], merge_keys: bool = False) -> tuple[list[_OptimStateKey], dict[_OptimStateKey, int | str]]:
    """
    Construct the local mapping between the ``_OptimStateKey`` and parameter keys
    and all the ``_OptimStateKey`` across ranks. If ``merge_keys`` is False, rank0
    must contain all the ``_OptimStateKey``, an exception will be raised otherwise.
    Note that ``merge_keys`` should equal to ``use_orig_params``.
    """
def _unflatten_param_groups(state_dict: dict[str, Any], param_key_to_param: dict[int | str, nn.Parameter], param_to_fqns: dict[nn.Parameter, list[str]]) -> list[dict[str, Any]]: ...
def _is_named_optimizer(optim_state_dict: dict[str, Any]) -> bool:
    """
    Returns whether the state_dict is from a NamedOptimizer.
    This function checks that the keys in the state_dict['state'] are strings
    (which usually are FQNs) versus integers (which usually refer to param_ids
    from a vanilla torch.optim.Optimizer).
    """

@dataclass
class StateInfo:
    tensors: dict[str, _PosDimTensorInfo]
    scalar_tensors: dict[str, torch.Tensor]
    non_tensors: dict[str, Any]

def _allgather_state_info(fsdp_state: _FSDPState, input_states: dict[str, Any]) -> list[dict[str, StateInfo]]:
    """
    Given the ``input_states``, allgather StateInfo for each state. The function
    uses all_gather_object to gather StateInfo so no GPU tensors are sent.
    """
def _convert_all_state_info(fsdp_param_info: FSDPParamInfo, gathered_state_info: list[dict[str, StateInfo]], input_states: dict[str, Any], output_states: dict[str, dict[str, Any]]) -> tuple[torch.dtype | None, dict[str, list[torch.Tensor | None]]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API converted
    the StateInfo into the original state if the state is not a non-scalar
    tensor. For a multi-dimensional tensor, the local state will be stored in
    ``state_buffer`` in a correct order for later allgather purpose.
    """
def _unflatten_orig_param_states(fsdp_param_info: FSDPParamInfo, output_states: dict[str, dict[str, Any]], state_name: str, shard_state: bool, to_save: bool, cpu_offload: bool) -> None:
    """
    Given a output state dict, ``output_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parameter ID), and the values
    are gathered states, unflatten the states to the original dimensions.

    This function performs the unflattening process in-place.
    """
def _allgather_orig_param_states(fsdp_param_info: FSDPParamInfo, gathered_state_info: list[dict[str, StateInfo]], input_states: dict[str, Any], shard_state: bool, to_save: bool, cpu_offload: bool) -> dict[str, dict[str, Any]]:
    """
    Given the ``gathered_state_info`` and ``input_states``, the API allgathers
    all tensor states and restore non-tensor states from ``gathered_state_info``.
    """
def _gather_all_orig_param_state(fsdp_param_info: FSDPParamInfo, input_states: dict[str, Any], shard_state: bool, to_save: bool, cpu_offload: bool) -> dict[str, Any]:
    """
    Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
    original parameters (not FlatParameters nor parameter ID), gather all the
    states and unflatten them to the original dimensions. Note that all the
    params referred by the ``input_states`` must be managed by FSDP.
    """
def _convert_state_with_orig_params(all_optim_state_keys: list[_OptimStateKey], optim_state_key_to_param_key: dict[_OptimStateKey, int | str], fqn_to_fsdp_param_info: dict[str, FSDPParamInfo], optim_state_dict: dict[str | int, Any], to_save: bool, shard_state: bool, cpu_offload: bool = True) -> dict[str, Any]: ...
def _convert_state_with_flat_params(all_optim_state_keys: list[_OptimStateKey], optim_state_key_to_param_key: dict[_OptimStateKey, int | str], fqn_to_fsdp_param_info: dict[str, FSDPParamInfo], optim_state_dict: dict[str | int, Any], to_save: bool, shard_state: bool, cpu_offload: bool = True) -> dict[str, Any]: ...
def _optim_state_dict(model: nn.Module, optim: torch.optim.Optimizer, optim_state_dict: dict[str, Any], optim_input: list[dict[str, Any]] | Iterable[nn.Parameter] | None, rank0_only: bool, shard_state: bool, group: dist.ProcessGroup | None, using_optim_input: bool, use_orig_params: bool = False, cpu_offload: bool = True) -> dict[str, Any]:
    '''
    Consolidates the optimizer state and returns it as a :class:`dict`
    following the convention of :meth:`torch.optim.Optimizer.state_dict`,
    i.e. with keys ``"state"`` and ``"param_groups"``.
    The flat parameters in ``FSDP`` modules contained in ``model`` are mapped
    back to their unflattened parameters.

    Parameter keys are not well-defined. For a regular optimizer, the optimizer
    state_dict contains a mapping from parameter IDs to parameter states.
    Parameter IDs are the order of parameters in ``optim.param_groups()`` across
    all the groups. This API also allows user to pass ``optim_input`` for the
    mapping between parameters and parameter IDs. Using ``optim_input`` is being
    deprecated.

    If the optimizer is a ``NamedOptimizer``, the optimizer state_dict does not
    contain parameter IDs mapping but a mapping from parameter FQNs to parameter
    states. This API finds the mapping from FQNs to parameters if the optimizer
    is a ``NamedOptimizer``.

    If ``use_orig_params`` is True, each rank will have all FSDP-managed
    parameters but some of these parameters may be empty due to the sharding.
    For a regular optim.Optimizer, states for those empty parameters will
    not be initialized. So, when aggregating the FQNs across ranks, no assert
    will be raised on a rank even if it does not have all the states -- it is
    valid and FSDP knows how to aggregate them. However, FSDP has to ignore
    handling those parameters that are not managed by FSDP and do not exist on
    the local rank -- those are managed by other parallelisms and FSDP does not
    know how to handle/aggregate them.

    Args:
        model (nn.Module): Root module (which may or may not be a
            :class:`FullyShardedDataParallel` instance) whose parameters
            were passed into the optimizer ``optim``.
        optim (torch.optim.Optimizer): Optimizer for ``model`` \'s
            parameters.
        rank0_only (bool): If ``True``, saves the populated :class:`dict`
            only on rank 0; if ``False``, saves it on all ranks. (Default:
            ``True``)
        shard_state (bool): If ``True``, shard and distribute all
            non-zero-dimension states.

    Returns:
        Dict[str, Any]: A :class:`dict` containing the optimizer state for
        ``model`` \'s original unflattened parameters and including keys
        "state" and "param_groups" following the convention of
        :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=False``,
        then nonzero ranks return an empty :class:`dict`.
    '''
def _get_fqn_to_fsdp_param_info(model: nn.Module) -> dict[str, FSDPParamInfo]:
    """
    Construct the mapping from a param's fqn to its corresponding ``FSDPParamInfo``
    if the param is managed by FSDP. Shared parameters, or original parameters that
    are shared across multiple nn.Modules, are required to belong to one and only
    one FSDP instance and thus correspond to one ``FlatParameter``. Within the one
    ``FlatParameter``, ``FlatParameter._fqns`` only stores the first FQN of a shared
    parameter. Thus, the keys in the mapping are guaranteed to map to unique parameters.
    """
@no_type_check
def _set_optim_use_dtensor(fsdp_state, state_dict_settings) -> None: ...
