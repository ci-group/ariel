import contextlib
import torch
import torch.distributed as dist
import torch.nn as nn
from ._fsdp_extensions import FSDPExtensions
from _typeshed import Incomplete
from collections.abc import Generator, Iterator, Sequence
from enum import Enum
from torch import Tensor
from torch.nn.parameter import _ParameterMeta
from typing import Any, Callable, NamedTuple, no_type_check

__all__ = ['FlatParameter', 'FlatParamHandle', 'FlatParamShardMetadata', 'ParamInfo', 'SharedParamInfo', 'HandleShardingStrategy']

class HandleShardingStrategy(Enum):
    FULL_SHARD = ...
    SHARD_GRAD_OP = ...
    NO_SHARD = ...
    HYBRID_SHARD = ...
    _HYBRID_SHARD_ZERO2 = ...

class ParamInfo(NamedTuple):
    """Information for an original parameter."""
    param_name: str
    module: nn.Module
    module_name: str

class SharedParamInfo(NamedTuple):
    '''
    Additional information for a shared parameter.

    For each shared parameter, we designate one module and its parameter
    variable to be the primary owner, determined as the first one encountered
    in the parameter walk. These are prefixed with "prim". The primary module
    and parameter do not have their own :class:`SharedParamInfo` instance.
    '''
    param_name: str
    module: nn.Module
    module_name: str
    prim_param_name: str
    prim_module: nn.Module
    prim_module_name: str

class _ShardParamInfo(NamedTuple):
    """Shard-related information for an original parameter."""
    in_shard: bool
    offset_in_shard: int | None
    numel_in_shard: int | None
    intra_param_start_idx: int | None
    intra_param_end_idx: int | None

class FlatParamShardMetadata(NamedTuple):
    """
    This holds metadata specific to this rank's shard of the flat parameter.

    Attributes:
        param_names (Tuple[str, ...]): Prefixed parameter names of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_shapes (Tuple[torch.Size, ...]): Parameter shapes of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_strides (Tuple[torch.Size, ...]): Parameter strides of this rank's
            shard of the parameters; see :class:`FlatParameter`.
        param_contiguities (Tuple[bool, ...]): Parameter `.contiguous` call results
            of this rank's shard of the parameters; see :class:`FlatParameter`.
        param_numels (Tuple[int, ...]): Parameter numels of this rank's shard
            of the parameters; see :class:`FlatParameter`.
        param_offsets (Tuple[Tuple[int, int], ...]): [start, end] offsets (in
            units of numels) giving this rank's part of each flattened
            original parameter.
    """
    param_names: tuple[str, ...]
    param_shapes: tuple[torch.Size, ...]
    param_strides: tuple[tuple[int, ...], ...]
    param_contiguities: tuple[bool, ...]
    param_numels: tuple[int, ...]
    param_offsets: tuple[tuple[int, int], ...]

class _FlatParameterMeta(_ParameterMeta):
    def __instancecheck__(self, instance): ...

class FlatParameter(nn.Parameter, metaclass=_FlatParameterMeta):
    '''
    This is the flat parameter used by :class:`FullyShardedDataParallel`.

    It is comprised of one or more original parameters, which are flattened and
    concatenated to construct the flat parameter.

    Under the current design, this parameter logically represents both the
    unsharded and sharded flat parameter, and its data changes storages
    dynamically.
        - In the :class:`FullyShardedDataParallel` constructor, the parameter
        is initialized as unsharded and then sharded in-place.
        - At runtime, the parameter is lazily (re)-initialized. The sharded
        parameter data is saved in ``self._local_shard``, and a new ``Tensor``
        ``self._full_param_padded`` is created, which is the all-gather
        destination and owns the unsharded parameter storage thereafter. (See
        :meth:`FlatParamHandle.init_flat_param_attributes`.)
        - Throughout runtime, the parameter data changes storages as needed,
        e.g. to the sharded flat parameter, low precision sharded flat
        parameter, or the unsharded flat parameter.

    NOTE: Since ``use_orig_params=True`` supports intra-``FlatParameter``
    padding, we have two versions of the per-parameter numels, one that
    includes the padding (``_numels_with_padding``) and one that does not
    (``_numels``). The former may have length longer than the other data
    structures, while the latter has the same length as the number of actual
    original parameters like the other per-parameter data structures.

    NOTE: This is not a real class; instead, you will always get a Parameter
    back out if you try to create one of these.  This is similar to the trick
    we implemented for Parameter to get it to work with subclasses; this
    is primarily so that FlatParameter supports combination with FakeTensor.

    Attributes:
        _unpadded_unsharded_size (torch.Size): Unsharded flat parameter\'s size
            without right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding.
        _padded_unsharded_size (torch.Size): Unsharded flat parameter\'s size
            with right-hand-side padding for divisibility by the world size.
            For ``use_orig_params=True``, this includes alignment padding. This
            is only set for sharded strategies since they require padding for
            the all-gather.
        _sharded_size (torch.Size): Sharded flat parameter\'s size with padding.
            This is also set for ``NO_SHARD``, in which case it is the same as
            the unsharded sizes. (We omit "padded" because there is no
            analogous unpadded one.)

        _num_params (int): Number of original parameters flattened into this
            flat parameter. This is the length of the per-parameter data
            structures.
        _param_infos (Tuple[ParamInfo, ...]): Each parameter\'s parameter info
            entry; see :class:`ParamInfo` for details.
        _shapes (Tuple[torch.Size, ...]): Each parameter\'s original shape.
        _strides (Tuple[torch.Size, ...]): Each parameter\'s original stride.
        _contiguities (Tuple[bool, ...]): Each parameter\'s ``contiguous()``
            call result.
        _fqns (Tuple[str, ...]): Each parameter\'s fully-qualified name (FQN)
            prefixed from the ``_fully_sharded_module``. The names are
            guaranteed to be unique in the subtree rooted at that module.
        _param_extensions (Tuple[Optional[Any], ...]): Each parameter\'s
            extension (i.e. some per-parameter state) used to customize
            pre-flatten and post-unflatten behavior or ``None``. This is
            experimental, and users should not depend on its existence in the
            future.
        _numels_with_padding (Tuple[int, ...]): Each parameter\'s numel
            including entries for the padding. This is used to construct views
            into the flat parameter via ``torch.split()``. This may have length
            longer than ``_num_params``.
        _numels (Tuple[int, ...]): Each parameter\'s numel excluding entries for
            padding. This has length equal to ``_num_params``.
        _shard_param_infos (Tuple[_ShardParamInfo, ...]): Each parameter\'s
            shard parameter info; see :class:`_ShardParamInfo` for details.
        _shared_param_infos (Tuple[SharedParamInfo, ...]): Shared parameter
            info entries; see :class:`SharedParamInfo` for details.
        _modules (set[nn.Module]): Modules that contain some original parameter
            that is flattened into the flat parameter.

        _shard_numel_padded (int): Numel padded for this rank\'s sharded flat
            parameter.
        _local_shard (Tensor): Sharded flat parameter with padding if using a
            sharded strategy. If using ``NO_SHARD``, then this is the unpadded
            unsharded flat parameter, and there is no notion of a sharded flat
            parameter or padded unsharded flat parameter.
        _full_param_padded (Tensor): Unsharded flat parameter with padding.
            This is not defined for ``NO_SHARD``. When using mixed precision
            for parameters, this has the low precision.
        _full_prec_full_param_padded (Tensor): Full precision unsharded flat
            parameter with padding. This is used for unsharding outside of
            computation when using mixed precision for parameters. This is
            never defined for ``NO_SHARD``.
        _post_backward_hook_handle (RemovableHandle):
            Flat parameter\'s post-backward hook handle. (Compile only)
        _post_backward_hook_state (Tuple[AccumulateGrad, RemovableHandle]):
            Flat parameter\'s :class:`AccumulateGrad` object and post-backward
            hook handle. (Eager only)
        _mp_shard (Tensor): Low precision sharded flat parameter with padding.
            This is only defined when parameter mixed precision is enabled. For
            ``NO_SHARD``, this is used for computation.
        _cpu_grad (Tensor): Sharded gradient with padding stored on CPU.
            This is only defined when offloading parameters is enabled.
        _saved_grad_shard (Tensor): Sharded gradient with padding from previous
            iterations for gradient accumulation without :meth:`no_sync`.

        _params (Optional[List[nn.Parameter]]): If ``use_orig_params=True``,
            then each original parameter variable; otherwise, ``None``. This
            does not include any padding tensors.
        _shared_params (Optional[List[nn.Parameter]]): The original shared
            parameter variables if ``use_orig_params=True`` and ``None``
            otherwise.
        _tensors (Optional[List[Optional[Tensor]]]): This saves the ``Tensor``
            views created in the forward and tracked by autograd when
            ``use_orig_params=True`` and is ``None`` otherwise. This is to
            preserve those ``Tensor`` variables for the backward to ensure that
            the ``FlatParameter`` \'s ``AccumulateGrad`` object does not change
            in which case the post-backward hook does not run. This is relevant
            for cases like reentrant activation checkpointing.
        _is_grad_none_mask (Optional[List[bool]]): If ``use_orig_params=True``,
            a mask over the original parameters\' gradients indicating if it is
            logically ``None`` or not; otherwise, ``None``. This does not
            include entries for padding. This mask is needed because only some
            of the parameters may have ``None`` gradient, in which case the
            flat gradient must be non-``None`` and must use zeros to
            approximate those original ``None`` gradients. This mask informs
            FSDP to set the original parameter gradients to ``None`` (instead
            of zeros) as needed.
    '''
    _unpadded_unsharded_size: torch.Size
    _padded_unsharded_size: torch.Size
    _sharded_size: torch.Size
    _num_params: int
    _param_infos: tuple[ParamInfo, ...]
    _shapes: tuple[torch.Size, ...]
    _strides: tuple[tuple[int, ...], ...]
    _contiguities: tuple[bool, ...]
    _fqns: tuple[str, ...]
    _param_extensions: tuple[Any | None, ...]
    _numels_with_padding: tuple[int, ...]
    _numels: tuple[int, ...]
    _shard_param_infos: tuple[_ShardParamInfo, ...]
    _shared_param_infos: tuple[SharedParamInfo, ...]
    _modules: set[nn.Module]
    _shard_numel_padded: int
    _local_shard: Tensor
    _full_param_padded: Tensor
    _full_prec_full_param_padded: Tensor
    _post_backward_hook_state: tuple[Any, Any]
    _post_backward_hook_handle: Any
    _mp_shard: Tensor
    _cpu_grad: Tensor
    _saved_grad_shard: Tensor
    _params: list[nn.Parameter] | None
    _shared_params: list[nn.Parameter] | None
    _tensors: list[Tensor | None] | None
    _is_grad_none_mask: list[bool] | None
    _is_padding_mask: list[bool]
    def __new__(cls, data=None, requires_grad: bool = True): ...
    _post_backward_called: bool
    @classmethod
    def _init_metadata(cls, self, param_infos: list[ParamInfo], numels: list[int], shapes: list[torch.Size], strides: list[tuple[int, ...]], contiguities: list[bool], fqns: list[str], shared_param_infos: list[SharedParamInfo], param_extensions: list[Any | None], params: list[nn.Parameter] | None, shared_params: list[nn.Parameter] | None, is_padding_mask: list[bool]) -> None:
        """
        Initialize attributes holding metadata about the original parameters comprising the flat parameter.

        We expose this method separate from the constructor to keep the
        constructor only responsible for the flat parameter's tensor data. This
        method should only be called once per model, while the constructor may
        be called multiple times, e.g. when reloading from a checkpoint, in
        which case only the tensor data needs to be passed to the constructor.
        Since :meth:`load_state_dict` is implemented via :meth:`copy_`, the
        metadata is correctly assumed to be unchanged.

        Args:
            See the Attributes in the class docstring.
        """

class FlatParamHandle:
    """
    A handle that manages a flat parameter (:class:`FlatParameter`).

    This includes sharding and view management.

    Args:
        params (Sequence[nn.Parameter]): The parameters to flatten into the
            flat parameter.
        fully_sharded_module (nn.Module): See [Note: Fully Sharded Module].
        device (torch.device): The compute and communication device, which
            should be a non-CPU device. We refer to it as the compute device.
        sharding_strategy (ShardingStrategy): Sharding strategy to apply to
            this handle's ``FlatParameter``.
        offload_params (bool): Whether to offload the handle's
            ``FlatParameter`` to CPU.
        mp_param_dtype (Optional[torch.dtype]): Parameter mixed precision
            setting passed to the FSDP constructor.
        mp_reduce_dtype (Optional[torch.dtype]): Gradient reduction mixed
            precision setting passed to the FSDP constructor.
        keep_low_precision_grads (bool): Whether to keep gradients in low
            precision.
        use_orig_params (bool): If ``True``, then FSDP preserves the original
            parameter variables and returns them from ``named_parameters()``
            (e.g. to support different optimizer hyperparameters within one
            :class:`FlatParameter`). If ``False``, then FSDP reconstructs the
            parameters every iteration and returns the :class:`FlatParameter` s
            from ``named_parameters()``.
    """
    _skip_writeback_check: Incomplete
    _use_full_prec_in_eval: Incomplete
    _use_fake_all_gather: Incomplete
    _use_fake_reduce: Incomplete
    device: Incomplete
    _device_handle: Incomplete
    process_group: Incomplete
    _fake_process_group: Incomplete
    rank: Incomplete
    world_size: Incomplete
    _sharding_strategy: Incomplete
    _offload_params: Incomplete
    _use_orig_params: Incomplete
    _keep_low_precision_grads: Incomplete
    _training_state: Incomplete
    _debug_level: Incomplete
    _fully_sharded_module: Incomplete
    _unsharded_flat_param_for_skipped_views: Tensor | None
    _handle_index: int | None
    _pre_forward_order_index: int | None
    _post_forward_index: int | None
    _needs_pre_forward_unshard: bool
    _needs_pre_backward_unshard: bool
    _prefetched: bool
    _orig_param_dtype: Incomplete
    _aligned_numel: Incomplete
    _fsdp_extension: Incomplete
    def __init__(self, params: Sequence[nn.Parameter | Tensor], fully_sharded_module: nn.Module, device: torch.device, sharding_strategy: HandleShardingStrategy, offload_params: bool, mp_param_dtype: torch.dtype | None, mp_reduce_dtype: torch.dtype | None, keep_low_precision_grads: bool, process_group: dist.ProcessGroup, use_orig_params: bool, *, fsdp_extension: FSDPExtensions | None = None) -> None: ...
    def __repr__(self) -> str: ...
    _setattr_tensor: Callable[[nn.Module, str, Tensor], None]
    _setattr_param: Callable[[nn.Module, str, nn.Parameter], None]
    def _init_setattr_fns(self) -> None: ...
    _get_unflat_views: Incomplete
    def _init_get_unflat_views_fn(self, align_addresses: bool): ...
    flat_param: FlatParameter
    def _init_flat_param_and_metadata(self, params: list[Tensor | nn.Parameter], module: nn.Module, aligned_numel: int, use_orig_params: bool) -> None:
        """
        Initialize the ``FlatParameter`` and its metadata.

        NOTE: This should only be called once at construction time, after which
        the ``FlatParameter`` metadata is assumed to be static.

        NOTE: The elements of ``params`` should only be ``Tensor`` s when
        composing with ``DTensor`` -based tensor parallelism, in which case the
        elements may be ``DTensor`` local shards.
        """
    def _validate_tensors_to_flatten(self, tensors: list[Tensor | nn.Parameter]) -> tuple:
        """Validate the tensors to flatten and returns any necessary metadata."""
    def flatten_tensors(self, tensors: list[Tensor], aligned_numel: int) -> Tensor:
        """
        Flatten ``tensors`` into a single flat tensor.

        The flattening optionally includes
        padding if ``aligned_numel`` is greater than 0, where ``aligned_numel``
        gives the numel required to have address alignment.

        NOTE: The padding alignment algorithm must be kept in sync with
        :meth:`_init_flat_param_metadata`. We separate the two methods because
        the initialization happens once, whereas this method may be called
        multiple times throughout training (e.g. for checkpointing).
        """
    def flatten_tensors_into_flat_param(self, tensors: list[Tensor], aligned_numel: int, requires_grad: bool) -> FlatParameter: ...
    _low_prec_param_dtype_specified: Incomplete
    _low_prec_reduce_dtype_specified: Incomplete
    _fwd_bwd_param_dtype: Incomplete
    _reduce_dtype: Incomplete
    def _init_param_reduce_dtypes(self, mp_param_dtype: torch.dtype | None, mp_reduce_dtype: torch.dtype | None) -> None:
        """
        Initialize param and reduce dtypes.

        Precondition: ``self.flat_param`` is set. This ensures that this
        handle's parameters have a single dtype.

        Postcondition: This sets ``self._fwd_bwd_param_dtype`` and
        ``self._reduce_dtype``. If ``mp_param_dtype`` or ``mp_reduce_dtype``
        is ``None``, then we assume the original parameter dtype. One special
        case is if ``mp_param_dtype`` is not ``None`` and ``mp_reduce_dtype``
        is ``None``, in which case we assume the gradient reduction dtype
        matches the forward/backward parameter dtype.
        """
    def shard(self) -> None:
        """
        Shard the handle's ``FlatParameter``.

        This allocates new memory for
        the sharded flat parameter and frees the unsharded flat parameter's
        storage.

        Postcondition: ``self.flat_param`` is the sharded flat parameter. Shard
        metadata attributes are set for all sharding strategies.
        """
    def _init_shard_metadata(self, numel_padded: int, unsharded_start_idx: int, unsharded_end_idx: int) -> None:
        """
        Initialize shard-related metadata for this rank's shard of the flat parameter.

        This includes ``_sharded_size``, ``_shard_param_infos``, and ``_shard_numel_padded``.

        Args:
            numel_padded (int): Numel padded for this rank's sharded flat
                parameter.
            unsharded_start_idx (int): Start index in the unsharded flat
            parameter assigned to this rank.
            unsharded_end_idx (int): End index (inclusive) in the unsharded
                flat parameter assigned to this rank.

        Precondition: ``self.flat_param`` 's data is the sharded flat
        parameter.
        """
    def _get_shard_metadata(self, unsharded_start_idx: int, unsharded_end_idx: int) -> tuple[_ShardParamInfo, ...]:
        """
        Compute the shard metadata based on ``unsharded_start_idx`` and ``unsharded_end_idx`` (inclusive).

        ``unsharded_start_idx`` and ``unsharded_end_idx`` give the interval of the
        unsharded flat parameter specifying the shard.
        """
    @staticmethod
    def _get_unpadded_shard(tensor: Tensor, rank: int, world_size: int) -> tuple[Tensor, int]:
        """
        Return the unpadded shard of ``tensor`` for the given ``rank`` and ``world_size``.

        The returned value is a tuple of the shard of ``tensor`` without any
        padding and the numel to pad for that shard.

        If ``tensor`` is already flattened or may be viewed in the flattened
        shape (which is true in the expected usage), then this method does not
        allocate any new tensor memory.
        """
    @staticmethod
    def _get_shard(tensor: Tensor, rank: int, world_size: int) -> tuple[Tensor, int]:
        """
        Return the shard of ``tensor`` with padding for the given ``rank`` and ``world_size`` and the numel padded for that shard.

        This method allocates new memory (via :meth:`clone`) since the
        unsharded ``tensor`` may be deallocated after this method returns.
        """
    @staticmethod
    def _get_sharded_size(tensor: Tensor, rank: int, world_size: int) -> torch.Size:
        """
        Return the shape of ``tensor`` after sharding including padding.

        This requires ``tensor`` to have 1D shape and ensures that the returned
        shape is 1D.
        """
    def _get_flat_param_offsets(self) -> list[tuple[int, int]]:
        """
        Return [start, end] offsets of each original parameter's flattened data in the unsharded flat parameter (without padding).

        NOTE: The returned list includes elements for alignment padding.
        """
    @no_type_check
    def shard_metadata(self):
        """
        Return the shard-related metadata specific to this rank's shard of the flat parameter.

        NOTE: The returned tuple does not include elements for alignment
        padding but does account for the padding.
        """
    @no_type_check
    def init_flat_param_attributes(self) -> None:
        """
        This initializes some attributes on the handle's ``FlatParameter``.
        This should be called during lazy initialization since it requires the
        parameter to be on the compute device if not offloading to CPU and we
        want to give users the chance to move the parameter appropriately after
        the FSDP constructor.

        For each tensor attribute on the ``FlatParameter``, see the unshard and
        reshard methods in this class for the allocation and free pattern.
        """
    def pre_unshard(self) -> bool:
        """
        Return ``False`` if this is a no-op and ``True`` otherwise.

        Postcondition: ``self.flat_param`` 's data is on the device for
        communication and is what should be all-gathered. This means that it
        matches the dtype of the expected unsharded parameter.
        """
    def _use_low_precision_shard(self) -> None:
        """Allocate on the compute device and switch to using the low precision sharded flat parameter."""
    def unshard(self) -> None:
        """
        Run the unshard logic.

        This includes all-gathering the flat parameter
        and switching to using the unsharded flat parameter. If the handle does
        not need unsharding, then this only switches to using the unsharded
        flat parameter. For ``NO_SHARD``, this is a no-op.

        If FSDP is in :meth:`summon_full_params` and the handle uses parameter
        mixed precision, then the parameter is forced to full precision.
        """
    def needs_unshard(self) -> bool:
        """Return if the handle's flat parameter needs to be unsharded."""
    def _alloc_padded_unsharded_flat_param(self):
        """
        Allocate the *padded* unsharded flat parameter.

        The unpadded unsharded
        flat parameter is always a view into the padded one. This padded
        parameter is saved to a different attribute on the ``FlatParameter``
        depending on if we force full precision.
        """
    def _get_padded_unsharded_flat_param(self) -> torch.Tensor:
        """
        Return a reference to the padded unsharded flat parameter depending on the calling context.

        This should only be called if using a sharded strategy.
        """
    def _all_gather_flat_param(self, padded_unsharded_flat_param: Tensor) -> Tensor:
        """
        All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

        Then switch to use the all-gathered tensor.
        """
    def _use_unsharded_flat_param(self, padded_unsharded_flat_param: torch.Tensor) -> None:
        """
        Switch to use the *unpadded* unsharded flat parameter.

        This is a view into the *padded* unsharded flat parameter.
        """
    def post_unshard(self) -> None:
        """
        Run the post-unshard logic.

        This includes freeing the low precision shard if needed.
        """
    def _free_low_precision_sharded_param(self) -> None:
        """Frees the low precision sharded flat parameter."""
    def unshard_grad(self) -> None:
        """
        Unshard the handle's ``FlatParameter``'s gradient.

        If all ranks have
        ``None`` gradient, then all original parameters will as well. This
        method performs an all-reduce and an all-gather. The additional
        all-reduce is tolerable since this method is not meant to be used on
        the computation critical path.

        Postcondition: ``_saved_grad_shard`` is defined and contains the value
        to set ``flat_param.grad`` after gradients are resharded.
        """
    def reshard_grad(self) -> None: ...
    def prepare_gradient_for_backward(self) -> None:
        """
        Prepare the gradient for the backward computation.

        This is done by saving and clearing any existing sharded gradient
        in ``.grad`` to enable computing a new unsharded gradient.
        """
    def prepare_gradient_for_optim(self) -> None:
        """Prepare the gradient for optimizer computation by moving the sharded gradient to the ``.grad`` attribute."""
    @contextlib.contextmanager
    def to_cpu(self) -> Generator[None]:
        """
        Move the unpadded unsharded flat parameter to CPU while in the context and moves it back to the previous device upon exit.

        For now, this assumes the ``FlatParameter`` is the unpadded unsharded flat parameter
        since (1) there is no reason to include the padding in the copy and (2)
        there is no use case for the sharded flat parameter.

        Precondition: ``self.flat_param`` 's data is the unpadded unsharded
        flat parameter on the compute device, and the handle uses a sharded
        strategy.
        Postcondition: Same as the precondition.
        """
    def reshard(self, free_unsharded_flat_param: bool):
        """
        Run the reshard logic.

        This includes freeing the unsharded flat
        parameter if ``free_unsharded_flat_param`` and switching to using the
        sharded flat parameter. Note that this also implicitly offloads
        the sharded flat parameter (if CPU offload is enabled) by pointing
        it to the ``_local_shard`` attribute which resides on CPU.
        """
    def post_reshard(self) -> None:
        """
        Run the post-reshard logic.

        This includes freeing any memory that
        can now be freed given that the ``FlatParameter`` points to the full
        precision sharded flat parameter.

        Precondition: ``self.flat_param`` 's data points to the full precision
        sharded flat parameter.
        """
    def _free_unsharded_flat_param(self) -> None:
        """
        Free the padded unsharded flat parameter. We allow this
        function to be called even when storage is not allocated

        The tensor to free depends
        on the calling context since the unshard may have forced full
        precision, in which case a different tensor is used.
        """
    def _use_sharded_flat_param(self) -> None:
        """Switches to using the sharded flat parameter."""
    @no_type_check
    def _get_unflat_views_unaligned(self, tensor=None):
        """
        Return unflattened ``Tensor`` views into ``tensor``.

        If `tensor`` is ``None``,  ``flat_param`` is used. The unflattening is based
        on ``flat_param`` 's metadata.

        Examples for ``tensor`` include ``flat_param.grad`` or unsharded
        tensor optimizer state.
        """
    @no_type_check
    def _get_unflat_views_aligned(self, tensor=None):
        """
        Return unflattened ``Tensor`` views into ``tensor`` with handling for padding.

        This method has the same contract as :meth:`_get_unflat_views_unaligned`
        except it checks for ``None`` placeholders representing padding for
        alignment, which may incur slightly more CPU overhead.
        """
    @no_type_check
    def _use_unsharded_views(self, as_params) -> None:
        """
        Unflatten the unsharded flat parameter by setting the original parameter variables to be views into it.

        Args:
            as_params (bool): If ``True``, then registers the original
                parameters as ``nn.Parameter`` s; if ``False``, then registers
                the original parameters only as ``Tensor`` s. ``False`` should
                be used during forward/backward computation and when hiding the
                original parameters from :meth:`nn.Module.named_parameters`.

        Note:
            when prefetching for next forward, current forward may be
            annotated with `@torch.no_grad()`
            `@torch.enable_grad()` ensures non-empty `view.grad_fn`
            otherwise `_post_backward_hook` will not get called
        """
    @no_type_check
    def _use_unsharded_grad_views(self) -> None:
        """
        Unflatten the unsharded flat parameter's gradient.

        The original parameter variables' gradients are set to be views into
        the unsharded flat parameter's gradient.
        """
    @contextlib.contextmanager
    def unflatten_as_params(self) -> Generator:
        """
        Unflatten the original parameters.

        The function assumes that the flat parameter is unsharded. When in the context,
        unflattens the original parameters as ``nn.Parameter`` views into the
        flat parameter, and after the context, restores the original parameters
        as ``Tensor`` views into the flat parameter.
        """
    @no_type_check
    def _use_sharded_views(self) -> None:
        """
        Set the original parameter variables' data to be flattened views into the sharded flat parameter.

        The views are kept as flattened to simplify the case where a parameter
        is sharded across ranks. Parameters whose data is not present in the
        sharded flat parameter have their data set to a size-0 empty tensor. We
        do not delete them to ensure to preserve expected behaviors like model
        printability. Parameters whose data is present must preserve their
        variables to be passable to an optimizer.
        """
    @no_type_check
    def _use_sharded_grad_views(self) -> None:
        """
        Set the original parameter variables' gradients to be flattened views into the sharded flat parameter's gradient.

        This is a no-op if there is no gradient.

        Parameters whose data is not present in the sharded flat parameter and
        parameters with ``requires_grad=False`` have their gradients set to
        ``None``. Since the gradient variables do not need to be preserved,
        this method does not manipulate existing ``Tensor`` data directly and
        creates new ``Tensor`` variables instead.
        """
    @no_type_check
    def _writeback_orig_params(self):
        """
        Write back any parameters that changed storage to the handle's ``FlatParameter``.

        Iterates over the original parameters and writes back any parameters
        that changed storages (due to a non-inplace operator) to the handle's
        ``FlatParameter``. This method preserves the ``FlatParameter` 's
        device even if an original parameter's device changes.

        Raises:
            RuntimeError: If an original parameter or gradient changes storages
            but no longer has the expected flattened shape.
        Returns: ``True`` if some writeback happened, and ``False`` otherwise.
        """
    def _writeback_tensor(self, src_tensor: Tensor | None, dst_tensor: Tensor, tensor_index: int, expected_shape: torch.Size, offset: int, is_param: bool) -> None:
        """
        Write back ``src_tensor`` to ``dst_tensor`` at offset ``offset``, where ``src_tensor`` should have shape ``expected_shape``.

        ``is_param`` indicates if the tensor is the parameter (if ``True``) or gradient (if
        ``False``). If ``src_tensor`` is ``None``, then the effect is zeroing
        instead of copying. ``tensor_index`` gives the index of ``src_tensor``
        in the metadata structures.

        Raises:
            RuntimeError: If the ``src_tensor`` does not have the expected
            shape.
        """
    def _reset_flat_param_grad_info_if_needed(self) -> None:
        """
        Reset ``flat_param.grad`` if needed.

        When ``use_orig_params=True``:
        (1) sets the underlying ``flat_param.grad`` to ``None`` if *all* of the
        original parameters' ``.grad`` are ``None``, and
        (2) sets ``flat_param.requires_grad=False`` if *none* of the original
        parameters require gradient.
        For (1), this is targeting ``optim.zero_grad(set_to_none=True)``, in
        which case we want to free the gradients as soon after the
        ``zero_grad()`` call as possible.
        """
    def _deregister_orig_params(self) -> None: ...
    def flat_param_to(self, *args, **kwargs) -> None:
        """Wrap an in-place call to ``.to()`` for ``self.flat_param``."""
    def _get_modules(self) -> set[nn.Module]:
        """Return a :class:`set` of the modules whose parameters are included in this handle's flat parameter."""
    def is_sharded(self, tensor: Tensor) -> bool:
        """
        Return whether ``tensor`` is *currently* sharded.

        For ``NO_SHARD``, we choose to have this always return ``False`` for clarity.
        """
    def param_module_names(self) -> Iterator[tuple[str, str]]: ...
    def shared_param_module_names(self) -> Iterator[tuple[str, str]]: ...
    @property
    def _fqns_in_shard(self) -> list[str]:
        """Return the FQNs of the parameters present in this rank's shard."""
    @property
    def sharded_grad(self) -> Tensor | None:
        """Return the handle's sharded gradient."""
    def _reset_is_grad_none(self) -> None:
        """
        Reset ``_is_grad_none_mask`` as needed.

        This method should only be
        called in the post-backward after gradient computation, in which case
        if a parameter requires gradient, then it will surely receive a
        gradient and we may reset its mask entry to ``False``.
        """
    def _check_sharded_strategy(self) -> None: ...
    def _check_on_compute_device(self, tensor: Tensor): ...
    def _check_on_cpu(self, tensor: Tensor): ...
    @staticmethod
    def _check_storage_freed(tensor: Tensor): ...
    @staticmethod
    def _check_storage_allocated(tensor: Tensor): ...
    def _check_low_precision_shard(self) -> None: ...
    def _check_unsharded(self, tensor: Tensor): ...
    def _check_sharded(self, tensor: Tensor): ...
    @property
    def uses_sharded_strategy(self) -> bool: ...
    @property
    def _uses_param_mixed_precision(self) -> bool: ...
    @property
    def _uses_reduce_mixed_precision(self) -> bool: ...
    @property
    def _force_full_precision(self) -> bool: ...
    @property
    def _skipped_use_sharded_views(self) -> bool:
        """
        This property is used for sharding strategies that do not free after forward with ``use_orig_params=True``.

        This returns if this handle is
        currently in a state where it has skipped using sharded views, in which
        case it can restore view invariants via ``_use_sharded_views()``.
        """
