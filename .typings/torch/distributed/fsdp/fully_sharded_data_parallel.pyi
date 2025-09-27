import contextlib
import torch
import torch.distributed as dist
import torch.nn as nn
from ._flat_param import FlatParameter
from .wrap import CustomPolicy, ModuleWrapPolicy
from _typeshed import Incomplete
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from enum import Enum
from torch.distributed.fsdp._common_utils import HandleTrainingState, TrainingState, _FSDPState
from torch.distributed.fsdp._init_utils import ProcessGroupType
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, MixedPrecision, OptimStateDictConfig, ShardingStrategy, StateDictConfig, StateDictSettings, StateDictType
from torch.distributed.tensor import DeviceMesh
from typing import Any, Callable

__all__ = ['FullyShardedDataParallel', 'OptimStateKeyType']

class OptimStateKeyType(Enum):
    """Represents the type of key in an optimizer state-dict."""
    PARAM_NAME = ...
    PARAM_ID = ...

class FullyShardedDataParallel(nn.Module, _FSDPState):
    '''A wrapper for sharding module parameters across data parallel workers.

    This is inspired by `Xu et al. <https://arxiv.org/abs/2004.13336>`_ as
    well as the ZeRO Stage 3 from `DeepSpeed <https://www.deepspeed.ai/>`_.
    FullyShardedDataParallel is commonly shortened to FSDP.

    To understand FSDP internals, refer to the
    :ref:`fsdp_notes`.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> import torch
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> torch.cuda.set_device(device_id)
        >>> sharded_module = FSDP(my_module)
        >>> optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
        >>> x = sharded_module(x, y=3, z=torch.Tensor([1]))
        >>> loss = x.sum()
        >>> loss.backward()
        >>> optim.step()

    Using FSDP involves wrapping your module and then initializing your
    optimizer after. This is required since FSDP changes the parameter
    variables.

    When setting up FSDP, you need to consider the destination CUDA
    device. If the device has an ID (``dev_id``), you have three options:

    * Place the module on that device
    * Set the device using ``torch.cuda.set_device(dev_id)``
    * Pass ``dev_id`` into the ``device_id`` constructor argument.

    This ensures that the FSDP instance\'s compute device is the
    destination device. For option 1 and 3, the FSDP initialization
    always occurs on GPU. For option 2, the FSDP initialization
    happens on module\'s current device, which may be a CPU.

    If you\'re using the ``sync_module_states=True`` flag, you need to
    ensure that the module is on a GPU or use the ``device_id``
    argument to specify a CUDA device that FSDP will move the module
    to in the FSDP constructor. This is necessary because
    ``sync_module_states=True`` requires GPU communication.

    FSDP also takes care of moving input tensors to the forward method
    to the GPU compute device, so you don\'t need to manually move them
    from CPU.

    For ``use_orig_params=True``,
    ``ShardingStrategy.SHARD_GRAD_OP`` exposes the unsharded
    parameters, not the sharded parameters after forward, unlike
    ``ShardingStrategy.FULL_SHARD``. If you want
    to inspect the gradients, you can use the ``summon_full_params``
    method with ``with_grads=True``.

    With ``limit_all_gathers=True``, you may see a gap in the FSDP
    pre-forward where the CPU thread is not issuing any kernels. This is
    intentional and shows the rate limiter in effect. Synchronizing the CPU
    thread in that way prevents over-allocating memory for subsequent
    all-gathers, and it should not actually delay GPU kernel execution.

    FSDP replaces managed modules\' parameters with ``torch.Tensor``
    views during forward and backward computation for autograd-related
    reasons. If your module\'s forward relies on saved references to
    the parameters instead of reacquiring the references each
    iteration, then it will not see FSDP\'s newly created views,
    and autograd will not work correctly.

    Finally, when using ``sharding_strategy=ShardingStrategy.HYBRID_SHARD``
    with the sharding process group being intra-node and the
    replication process group being inter-node, setting
    ``NCCL_CROSS_NIC=1`` can help improve the all-reduce times over
    the replication process group for some cluster setups.

    **Limitations**

    There are several limitations to be aware of when using FSDP:

    * FSDP currently does not support gradient accumulation outside
      ``no_sync()`` when using CPU offloading. This is because FSDP
      uses the newly-reduced gradient instead of accumulating with any
      existing gradient, which can lead to incorrect results.

    * FSDP does not support running the forward pass of a submodule
      that is contained in an FSDP instance. This is because the
      submodule\'s parameters will be sharded, but the submodule itself
      is not an FSDP instance, so its forward pass will not all-gather
      the full parameters appropriately.

    * FSDP does not work with double backwards due to the way it
      registers backward hooks.

    * FSDP has some constraints when freezing parameters.
      For ``use_orig_params=False``, each FSDP instance must manage
      parameters that are all frozen or all non-frozen. For
      ``use_orig_params=True``, FSDP supports mixing frozen and
      non-frozen parameters, but it\'s recommended to avoid doing so to
      prevent higher than expected gradient memory usage.

    * As of PyTorch 1.12, FSDP offers limited support for shared
      parameters. If enhanced shared parameter support is needed for
      your use case, please post in
      `this issue <https://github.com/pytorch/pytorch/issues/77724>`__.

    * You should avoid modifying the parameters between forward and
      backward without using the ``summon_full_params`` context, as
      the modifications may not persist.

    Args:
        module (nn.Module):
            This is the module to be wrapped with FSDP.
        process_group (Optional[Union[ProcessGroup, Tuple[ProcessGroup, ProcessGroup]]]):
            This is the process group over which the model is sharded and thus
            the one used for FSDP\'s all-gather and reduce-scatter collective
            communications. If ``None``, then FSDP uses the default process
            group. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a tuple of
            process groups, representing the groups over which to shard and
            replicate, respectively. If ``None``, then FSDP constructs process
            groups for the user to shard intra-node and replicate inter-node.
            (Default: ``None``)
        sharding_strategy (Optional[ShardingStrategy]):
            This configures the sharding strategy, which may trade off memory
            saving and communication overhead. See :class:`ShardingStrategy`
            for details. (Default: ``FULL_SHARD``)
        cpu_offload (Optional[CPUOffload]):
            This configures CPU offloading. If this is set to ``None``, then
            no CPU offloading happens. See :class:`CPUOffload` for details.
            (Default: ``None``)
        auto_wrap_policy (Optional[Union[Callable[[nn.Module, bool, int], bool], ModuleWrapPolicy, CustomPolicy]]):
            This specifies a policy to apply FSDP to submodules of ``module``,
            which is needed for communication and computation overlap and thus
            affects performance. If ``None``, then FSDP only applies to
            ``module``, and users should manually apply FSDP to parent modules
            themselves (proceeding bottom-up). For convenience, this accepts
            ``ModuleWrapPolicy`` directly, which allows users to specify the
            module classes to wrap (e.g. the transformer block). Otherwise,
            this should be a callable that takes in three arguments
            ``module: nn.Module``, ``recurse: bool``, and
            ``nonwrapped_numel: int`` and should return a ``bool`` specifying
            whether the passed-in ``module`` should have FSDP applied if
            ``recurse=False`` or if the traversal should continue into the
            module\'s subtree if ``recurse=True``. Users may add additional
            arguments to the callable. The ``size_based_auto_wrap_policy`` in
            ``torch.distributed.fsdp.wrap.py`` gives an example callable that
            applies FSDP to a module if the parameters in its subtree exceed
            100M numel. We recommend printing the model after applying FSDP
            and adjusting as needed.

            Example::

                >>> def custom_auto_wrap_policy(
                >>>     module: nn.Module,
                >>>     recurse: bool,
                >>>     nonwrapped_numel: int,
                >>>     # Additional custom arguments
                >>>     min_num_params: int = int(1e8),
                >>> ) -> bool:
                >>>     return nonwrapped_numel >= min_num_params
                >>> # Configure a custom `min_num_params`
                >>> my_auto_wrap_policy = functools.partial(custom_auto_wrap_policy, min_num_params=int(1e5))

        backward_prefetch (Optional[BackwardPrefetch]):
            This configures explicit backward prefetching of all-gathers. If
            ``None``, then FSDP does not backward prefetch, and there is no
            communication and computation overlap in the backward pass. See
            :class:`BackwardPrefetch` for details. (Default: ``BACKWARD_PRE``)
        mixed_precision (Optional[MixedPrecision]):
            This configures native mixed precision for FSDP. If this is set to
            ``None``, then no mixed precision is used. Otherwise, parameter,
            buffer, and gradient reduction dtypes can be set. See
            :class:`MixedPrecision` for details. (Default: ``None``)
        ignored_modules (Optional[Iterable[torch.nn.Module]]): Modules whose
            own parameters and child modules\' parameters and buffers are
            ignored by this instance. None of the modules directly in
            ``ignored_modules`` should be :class:`FullyShardedDataParallel`
            instances, and any child modules that are already-constructed
            :class:`FullyShardedDataParallel` instances will not be ignored if
            they are nested under this instance. This argument may be used to
            avoid sharding specific parameters at module granularity when using an
            ``auto_wrap_policy`` or if parameters\' sharding is not managed by
            FSDP. (Default: ``None``)
        param_init_fn (Optional[Callable[[nn.Module], None]]):
            A ``Callable[torch.nn.Module] -> None`` that
            specifies how modules that are currently on the meta device should
            be initialized onto an actual device. As of v1.12, FSDP detects
            modules with parameters or buffers on meta device via ``is_meta``
            and either applies ``param_init_fn`` if specified or calls
            ``nn.Module.reset_parameters()`` otherwise. For both cases, the
            implementation should *only* initialize the parameters/buffers of
            the module, not those of its submodules. This is to avoid
            re-initialization. In addition, FSDP also supports deferred
            initialization via torchdistX\'s (https://github.com/pytorch/torchdistX)
            ``deferred_init()`` API, where the deferred modules are initialized
            by calling ``param_init_fn`` if specified or torchdistX\'s default
            ``materialize_module()`` otherwise. If ``param_init_fn`` is
            specified, then it is applied to all meta-device modules, meaning
            that it should probably case on the module type. FSDP calls the
            initialization function before parameter flattening and sharding.

            Example::

                >>> # xdoctest: +SKIP("undefined variables")
                >>> module = MyModule(device="meta")
                >>> def my_init_fn(module: nn.Module):
                >>>     # E.g. initialize depending on the module type
                >>>     ...
                >>> fsdp_model = FSDP(module, param_init_fn=my_init_fn, auto_wrap_policy=size_based_auto_wrap_policy)
                >>> print(next(fsdp_model.parameters()).device) # current CUDA device
                >>> # With torchdistX
                >>> module = deferred_init.deferred_init(MyModule, device="cuda")
                >>> # Will initialize via deferred_init.materialize_module().
                >>> fsdp_model = FSDP(module, auto_wrap_policy=size_based_auto_wrap_policy)

        device_id (Optional[Union[int, torch.device]]): An ``int`` or
            ``torch.device`` giving the CUDA device on which FSDP
            initialization takes place, including the module initialization
            if needed and the parameter sharding. This should be specified to
            improve initialization speed if ``module`` is on CPU. If the
            default CUDA device was set (e.g. via ``torch.cuda.set_device``),
            then the user may pass ``torch.cuda.current_device`` to this.
            (Default: ``None``)
        sync_module_states (bool): If ``True``, then each FSDP module will
            broadcast module parameters and buffers from rank 0 to ensure that
            they are replicated across ranks (adding communication overhead to
            this constructor). This can help load ``state_dict`` checkpoints
            via ``load_state_dict`` in a memory efficient way. See
            :class:`FullStateDictConfig` for an example of this. (Default:
            ``False``)
        forward_prefetch (bool): If ``True``, then FSDP *explicitly* prefetches
            the next forward-pass all-gather before the current forward
            computation. This is only useful for CPU-bound workloads, in which
            case issuing the next all-gather earlier may improve overlap. This
            should only be used for static-graph models since the prefetching
            follows the first iteration\'s execution order. (Default: ``False``)
        limit_all_gathers (bool): If ``True``, then FSDP explicitly
            synchronizes the CPU thread to ensure GPU memory usage from only
            *two* consecutive FSDP instances (the current instance running
            computation and the next instance whose all-gather is prefetched).
            If ``False``, then FSDP allows the CPU thread to issue all-gathers
            without any extra synchronization. (Default: ``True``) We often
            refer to this feature as the "rate limiter". This flag should only
            be set to ``False`` for specific CPU-bound workloads with low
            memory pressure in which case the CPU thread can aggressively issue
            all kernels without concern for the GPU memory usage.
        use_orig_params (bool): Setting this to ``True`` has FSDP use
            ``module`` \'s original parameters. FSDP exposes those original
            parameters to the user via :meth:`nn.Module.named_parameters`
            instead of FSDP\'s internal :class:`FlatParameter` s. This means
            that the optimizer step runs on the original parameters, enabling
            per-original-parameter hyperparameters. FSDP preserves the original
            parameter variables and manipulates their data between unsharded
            and sharded forms, where they are always views into the underlying
            unsharded or sharded :class:`FlatParameter`, respectively. With the
            current algorithm, the sharded form is always 1D, losing the
            original tensor structure. An original parameter may have all,
            some, or none of its data present for a given rank. In the none
            case, its data will be like a size-0 empty tensor. Users should not
            author programs relying on what data is present for a given
            original parameter in its sharded form. ``True`` is required to
            use ``torch.compile()``. Setting this to ``False`` exposes FSDP\'s
            internal :class:`FlatParameter` s to the user via
            :meth:`nn.Module.named_parameters`. (Default: ``False``)
        ignored_states (Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]):
            Ignored parameters or modules that will not be managed by this FSDP
            instance, meaning that the parameters are not sharded and their
            gradients are not reduced across ranks. This argument unifies with
            the existing ``ignored_modules`` argument, and we may deprecate
            ``ignored_modules`` soon. For backward compatibility, we keep both
            ``ignored_states`` and `ignored_modules``, but FSDP only allows one
            of them to be specified as not ``None``.
        device_mesh (Optional[DeviceMesh]): DeviceMesh can be used as an alternative to
            process_group. When device_mesh is passed, FSDP will use the underlying process
            groups for all-gather and reduce-scatter collective communications. Therefore,
            these two args need to be mutually exclusive. For hybrid sharding strategies such as
            ``ShardingStrategy.HYBRID_SHARD``, users can pass in a 2D DeviceMesh instead
            of a tuple of process groups. For 2D FSDP + TP, users are required to pass in
            device_mesh instead of process_group. For more DeviceMesh info, please visit:
            https://pytorch.org/tutorials/recipes/distributed_device_mesh.html
    '''
    _device_mesh: Incomplete
    _fsdp_wrapped_module: Incomplete
    _zero_scalar: Incomplete
    def __init__(self, module: nn.Module, process_group: ProcessGroupType = None, sharding_strategy: ShardingStrategy | None = None, cpu_offload: CPUOffload | None = None, auto_wrap_policy: Callable | ModuleWrapPolicy | CustomPolicy | None = None, backward_prefetch: BackwardPrefetch | None = ..., mixed_precision: MixedPrecision | None = None, ignored_modules: Iterable[torch.nn.Module] | None = None, param_init_fn: Callable[[nn.Module], None] | None = None, device_id: int | torch.device | None = None, sync_module_states: bool = False, forward_prefetch: bool = False, limit_all_gathers: bool = True, use_orig_params: bool = False, ignored_states: Iterable[torch.nn.Parameter] | None | Iterable[torch.nn.Module] | None = None, device_mesh: DeviceMesh | None = None) -> None: ...
    @property
    def module(self) -> nn.Module:
        """Return the wrapped module."""
    @property
    def _has_params(self) -> bool:
        """Returns whether this FSDP instance manages any parameters."""
    @property
    def _flat_param(self) -> FlatParameter | None: ...
    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
    def __getitem__(self, key: int) -> Any:
        """Forward indexing calls in case the module is an ``nn.Sequential``."""
    def check_is_root(self) -> bool:
        """Check if this instance is a root FSDP module."""
    @staticmethod
    def fsdp_modules(module: nn.Module, root_only: bool = False) -> list['FullyShardedDataParallel']:
        """Return all nested FSDP instances.

        This possibly includes ``module`` itself and only includes FSDP root modules if ``root_only=True``.

        Args:
            module (torch.nn.Module): Root module, which may or may not be an
                ``FSDP`` module.
            root_only (bool): Whether to return only FSDP root modules.
                (Default: ``False``)

        Returns:
            List[FullyShardedDataParallel]: FSDP modules that are nested in
            the input ``module``.
        """
    def apply(self, fn: Callable[[nn.Module], None]) -> FullyShardedDataParallel:
        """Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model (see also :ref:`nn-init-doc`).

        Compared to ``torch.nn.Module.apply``, this version additionally gathers
        the full parameters before applying ``fn``. It should not be called from
        within another ``summon_full_params`` context.

        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self
        """
    def _mixed_precision_enabled_for_buffers(self) -> bool:
        """Return whether the user explicitly enabled buffer mixed precision.

        NOTE: Unlike parameters and gradient reduction, buffer mixed precision
        is applied at the FSDP instance level, not the ``FlatParameter`` level,
        which may be different for the composable code path.
        """
    def _low_precision_hook_enabled(self) -> bool:
        """Whether a low precision hook is registered or not."""
    _is_root: bool | None
    def _reset_lazy_init(self) -> None:
        """Reset instance so :func:`_lazy_init` will run on the next forward."""
    @staticmethod
    def set_state_dict_type(module: nn.Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None) -> StateDictSettings:
        '''Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        Also takes (optional) configuration for the model\'s and optimizer\'s state dict.
        The target module does not have to be a FSDP module. If the target
        module is a FSDP module, its ``state_dict_type`` will also be changed.

        .. note:: This API should be called for only the top-level (root)
            module.

        .. note:: This API enables users to transparently use the conventional
            ``state_dict`` API to take model checkpoints in cases where the
            root FSDP module is wrapped by another ``nn.Module``. For example,
            the following will ensure ``state_dict`` is called on all non-FSDP
            instances, while dispatching into `sharded_state_dict` implementation
            for FSDP:

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>>     state_dict_config = ShardedStateDictConfig(offload_to_cpu=True),
            >>>     optim_state_dict_config = OptimStateDictConfig(offload_to_cpu=True),
            >>> )
            >>> param_state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the configuration for the
                target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the configuration
                for the optimizer state dict.

        Returns:
            A StateDictSettings that include the previous state_dict type and
            configuration for the module.
        '''
    @staticmethod
    def get_state_dict_type(module: nn.Module) -> StateDictSettings:
        """Get the state_dict_type and the corresponding configurations for the FSDP modules rooted at ``module``.

        The target module does not have to be an FSDP module.

        Returns:
            A ``StateDictSettings`` containing the state_dict_type and
            state_dict / optim_state_dict configs that are currently set.

        Raises:
            ``AssertionError`` if the ``StateDictSettings`` for different
            FSDP submodules differ.
        """
    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module: nn.Module, state_dict_type: StateDictType, state_dict_config: StateDictConfig | None = None, optim_state_dict_config: OptimStateDictConfig | None = None) -> Generator:
        '''Set the ``state_dict_type`` of all the descendant FSDP modules of the target module.

        This context manager has the same functions as :meth:`set_state_dict_type`. Read the document of
        :meth:`set_state_dict_type` for the detail.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> model = DDP(FSDP(...))
            >>> with FSDP.state_dict_type(
            >>>     model,
            >>>     StateDictType.SHARDED_STATE_DICT,
            >>> ):
            >>>     checkpoint = model.state_dict()

        Args:
            module (torch.nn.Module): Root module.
            state_dict_type (StateDictType): the desired ``state_dict_type`` to set.
            state_dict_config (Optional[StateDictConfig]): the model ``state_dict``
                configuration for the target ``state_dict_type``.
            optim_state_dict_config (Optional[OptimStateDictConfig]): the optimizer
               ``state_dict`` configuration for the target ``state_dict_type``.
        '''
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the forward pass for the wrapped module, inserting FSDP-specific pre- and post-forward sharding logic."""
    @staticmethod
    @contextlib.contextmanager
    def summon_full_params(module: nn.Module, recurse: bool = True, writeback: bool = True, rank0_only: bool = False, offload_to_cpu: bool = False, with_grads: bool = False) -> Generator:
        """Expose full params for FSDP instances with this context manager.

        Can be useful *after* forward/backward for a model to get
        the params for additional processing or checking. It can take a non-FSDP
        module and will summon full params for all contained FSDP modules as
        well as their children, depending on the ``recurse`` argument.

        .. note:: This can be used on inner FSDPs.
        .. note:: This can *not* be used within a forward or backward pass. Nor
            can forward and backward be started from within this context.
        .. note:: Parameters will revert to their local shards after the context
            manager exits, storage behavior is the same as forward.
        .. note:: The full parameters can be modified, but only the portion
            corresponding to the local param shard will persist after the
            context manager exits (unless ``writeback=False``, in which case
            changes will be discarded). In the case where FSDP does not shard
            the parameters, currently only when ``world_size == 1``, or ``NO_SHARD``
            config, the modification is persisted regardless of ``writeback``.
        .. note:: This method works on modules which are not FSDP themselves but
            may contain multiple independent FSDP units. In that case, the given
            arguments will apply to all contained FSDP units.

        .. warning:: Note that ``rank0_only=True`` in conjunction with
            ``writeback=True`` is not currently supported and will raise an
            error. This is because model parameter shapes would be different
            across ranks within the context, and writing to them can lead to
            inconsistency across ranks when the context is exited.

        .. warning:: Note that ``offload_to_cpu`` and ``rank0_only=False`` will
            result in full parameters being redundantly copied to CPU memory for
            GPUs that reside on the same machine, which may incur the risk of
            CPU OOM. It is recommended to use ``offload_to_cpu`` with
            ``rank0_only=True``.

        Args:
            recurse (bool, Optional): recursively summon all params for nested
                FSDP instances (default: True).
            writeback (bool, Optional): if ``False``, modifications to params are
                discarded after the context manager exits;
                disabling this can be slightly more efficient (default: True)
            rank0_only (bool, Optional): if ``True``, full parameters are
                materialized on only global rank 0. This means that within the
                context, only rank 0 will have full parameters and the other
                ranks will have sharded parameters. Note that setting
                ``rank0_only=True`` with ``writeback=True`` is not supported,
                as model parameter shapes will be different across ranks
                within the context, and writing to them can lead to
                inconsistency across ranks when the context is exited.
            offload_to_cpu (bool, Optional): If ``True``, full parameters are
                offloaded to CPU. Note that this offloading currently only
                occurs if the parameter is sharded (which is only not the case
                for world_size = 1 or ``NO_SHARD`` config). It is recommended
                to use ``offload_to_cpu`` with ``rank0_only=True`` to avoid
                redundant copies of model parameters being offloaded to the same CPU memory.
            with_grads (bool, Optional): If ``True``, gradients are also
                unsharded with the parameters. Currently, this is only
                supported when passing ``use_orig_params=True`` to the FSDP
                constructor and ``offload_to_cpu=False`` to this method.
                (Default: ``False``)
        """
    @contextlib.contextmanager
    def _deregister_orig_params_ctx(self) -> Generator[None]:
        """Deregister the original parameters and expose the :class:`FlatParameter`.

        If a :class:`FlatParameter` is sharded, then
        this refreshes the sharded views before exiting. This method should
        only be called when using the original parameters.
        """
    def _apply(self, *args, **kwargs):
        """Deregister the original parameters and expose the :class:`FlatParameter` s before calling ``_apply()``."""
    def named_buffers(self, *args, **kwargs) -> Iterator[tuple[str, torch.Tensor]]:
        """Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.

        Intercepts buffer names and removes all occurrences of the FSDP-specific flattened buffer prefix
        when inside the :meth:`summon_full_params` context manager.
        """
    def named_parameters(self, *args, **kwargs) -> Iterator[tuple[str, torch.nn.Parameter]]:
        """Return an iterator over module parameters, yielding both the name of the parameter and the parameter itself.

        Intercepts parameter names and removes all occurrences of the FSDP-specific flattened parameter prefix
        when inside the :meth:`summon_full_params` context manager.
        """
    def _assert_state(self, state: TrainingState | list[TrainingState]) -> None:
        """Assert we are in the given state."""
    @contextmanager
    def no_sync(self) -> Generator:
        """Disable gradient synchronizations across FSDP instances.

        Within this context, gradients will be accumulated in module
        variables, which will later be synchronized in the first
        forward-backward pass after exiting the context. This should only be
        used on the root FSDP instance and will recursively apply to all
        children FSDP instances.

        .. note:: This likely results in higher memory usage because FSDP will
            accumulate the full model gradients (instead of gradient shards)
            until the eventual sync.

        .. note:: When used with CPU offloading, the gradients will not be
            offloaded to CPU when inside the context manager. Instead, they
            will only be offloaded right after the eventual sync.
        """
    def clip_grad_norm_(self, max_norm: float | int, norm_type: float | int = 2.0) -> torch.Tensor:
        '''Clip the gradient norm of all parameters.

        The norm is computed over all parameters\' gradients as viewed as a single vector, and the
        gradients are modified in-place.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (float or int): type of the used p-norm. Can be ``\'inf\'``
                for infinity norm.

        Returns:
            Total norm of the parameters (viewed as a single vector).

        If every FSDP instance uses ``NO_SHARD``, meaning that no
        gradients are sharded across ranks, then you may directly use
        :func:`torch.nn.utils.clip_grad_norm_`.

        If at least some FSDP instance uses a sharded strategy (i.e.
        one other than ``NO_SHARD``), then you should use this method
        instead of :func:`torch.nn.utils.clip_grad_norm_` since this method
        handles the fact that gradients are sharded across ranks.

        The total norm returned will have the "largest" dtype across
        all parameters/gradients as defined by PyTorch\'s type promotion
        semantics. For example, if *all* parameters/gradients use a low
        precision dtype, then the returned norm\'s dtype will be that low
        precision dtype, but if there exists at least one parameter/
        gradient using FP32, then the returned norm\'s dtype will be FP32.

        .. warning:: This needs to be called on all ranks since it uses
            collective communications.
        '''
    @staticmethod
    def _warn_optim_input(optim_input, *, stacklevel: int = 1): ...
    @staticmethod
    def _is_using_optim_input(optim_input, optim) -> bool: ...
    @staticmethod
    def _warn_legacy_optim_state_dict(curr: str, new: str, *, stacklevel: int = 1): ...
    @staticmethod
    def _optim_state_dict_impl(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_state_dict: dict[str, Any], optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, rank0_only: bool = True, full_state_dict: bool = True, group: dist.ProcessGroup | None = None, cpu_offload: bool = True, *, _stacklevel: int = 1) -> dict[str, Any]:
        """Transform the state-dict of an optimizer corresponding to a sharded model.

        This is the internal API that is used by all the optim_state_dict implementations.
        Given model, optim, the original optim_state_dict, this API removes the
        FSDP internal information and internal sharding from the optim_state_dict.
        """
    @staticmethod
    def _optim_state_dict_to_load_impl(optim_state_dict: dict[str, Any], model: torch.nn.Module, optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, optim: torch.optim.Optimizer | None = None, full_state_dict: bool = True, rank0_only: bool = False, is_named_optimizer: bool = False, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
        """
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        This is the internal API that is used by all the load optim_state_dict implementations.
        Given model, optim, and the saved optim_state_dict, this API adds the FSDP
        internal information and internal sharding to the optim_state_dict.
        """
    @staticmethod
    def full_optim_state_dict(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, rank0_only: bool = True, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
        '''Return the full optimizer state-dict.

        Consolidates the full optimizer state on rank 0 and returns it
        as a :class:`dict` following the convention of
        :meth:`torch.optim.Optimizer.state_dict`, i.e. with keys ``"state"``
        and ``"param_groups"``. The flattened parameters in ``FSDP`` modules
        contained in ``model`` are mapped back to their unflattened parameters.

        This needs to be called on all ranks since it uses
        collective communications. However, if ``rank0_only=True``, then
        the state dict is only populated on rank 0, and all other ranks
        return an empty :class:`dict`.

        Unlike ``torch.optim.Optimizer.state_dict()``, this method
        uses full parameter names as keys instead of parameter IDs.

        Like in :meth:`torch.optim.Optimizer.state_dict`, the tensors
        contained in the optimizer state dict are not cloned, so there may
        be aliasing surprises. For best practices, consider saving the
        returned optimizer state dict immediately, e.g. using
        ``torch.save()``.

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` \'s
                parameters.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer ``optim`` representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            rank0_only (bool): If ``True``, saves the populated :class:`dict`
                only on rank 0; if ``False``, saves it on all ranks. (Default:
                ``True``)
            group (dist.ProcessGroup): Model\'s process group or ``None`` if using
                the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model`` \'s original unflattened parameters and including keys
            "state" and "param_groups" following the convention of
            :meth:`torch.optim.Optimizer.state_dict`. If ``rank0_only=True``,
            then nonzero ranks return an empty :class:`dict`.
        '''
    @staticmethod
    def sharded_optim_state_dict(model: torch.nn.Module, optim: torch.optim.Optimizer, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
        """Return the optimizer state-dict in its sharded form.

        The API is similar to :meth:`full_optim_state_dict` but this API chunks
        all non-zero-dimension states to :class:`ShardedTensor` to save memory.
        This API should only be used when the model ``state_dict`` is derived
        with the context manager ``with state_dict_type(SHARDED_STATE_DICT):``.

        For the detailed usage, refer to :meth:`full_optim_state_dict`.

        .. warning:: The returned state dict contains ``ShardedTensor`` and
            cannot be directly used by the regular ``optim.load_state_dict``.
        """
    @staticmethod
    def shard_full_optim_state_dict(full_optim_state_dict: dict[str, Any], model: torch.nn.Module, optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, optim: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        '''Shard a full optimizer state-dict.

        Remaps the state in ``full_optim_state_dict`` to flattened parameters instead of unflattened
        parameters and restricts to only this rank\'s part of the optimizer state.
        The first argument should be the return value of :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)
            >>> torch.save(full_osd, PATH)
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim = ...
            >>> full_osd = torch.load(PATH)
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(full_osd, new_model)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                full non-sharded optimizer state.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank\'s part of the optimizer state.
        '''
    @staticmethod
    def flatten_sharded_optim_state_dict(sharded_optim_state_dict: dict[str, Any], model: torch.nn.Module, optim: torch.optim.Optimizer) -> dict[str, Any]:
        """Flatten a sharded optimizer state-dict.

        The API is similar to :meth:`shard_full_optim_state_dict`. The only
        difference is that the input ``sharded_optim_state_dict`` should be
        returned from :meth:`sharded_optim_state_dict`. Therefore, there will
        be all-gather calls on each rank to gather ``ShardedTensor`` s.

        Args:
            sharded_optim_state_dict (Dict[str, Any]): Optimizer state dict
                corresponding to the unflattened parameters and holding the
                sharded optimizer state.
            model (torch.nn.Module):
                Refer to :meth:`shard_full_optim_state_dict`.
            optim (torch.optim.Optimizer): Optimizer for ``model`` 's
                parameters.

        Returns:
            Refer to :meth:`shard_full_optim_state_dict`.
        """
    @staticmethod
    def scatter_full_optim_state_dict(full_optim_state_dict: dict[str, Any] | None, model: torch.nn.Module, optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, optim: torch.optim.Optimizer | None = None, group: Any | None = None) -> dict[str, Any]:
        '''Scatter the full optimizer state dict from rank 0 to all other ranks.

        Returns the sharded optimizer state dict on each rank.
        The return value is the same as :meth:`shard_full_optim_state_dict`, and on rank
        0, the first argument should be the return value of
        :meth:`full_optim_state_dict`.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> model, optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(model, optim)  # only non-empty on rank 0
            >>> # Define new model with possibly different world size
            >>> new_model, new_optim, new_group = ...
            >>> sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, new_model, group=new_group)
            >>> new_optim.load_state_dict(sharded_osd)

        .. note:: Both :meth:`shard_full_optim_state_dict` and
            :meth:`scatter_full_optim_state_dict` may be used to get the
            sharded optimizer state dict to load. Assuming that the full
            optimizer state dict resides in CPU memory, the former requires
            each rank to have the full dict in CPU memory, where each rank
            individually shards the dict without any communication, while the
            latter requires only rank 0 to have the full dict in CPU memory,
            where rank 0 moves each shard to GPU memory (for NCCL) and
            communicates it to ranks appropriately. Hence, the former has
            higher aggregate CPU memory cost, while the latter has higher
            communication cost.

        Args:
            full_optim_state_dict (Optional[Dict[str, Any]]): Optimizer state
                dict corresponding to the unflattened parameters and holding
                the full non-sharded optimizer state if on rank 0; the argument
                is ignored on nonzero ranks.
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                correspond to the optimizer state in ``full_optim_state_dict``.
            optim_input (Optional[Union[List[Dict[str, Any]], Iterable[torch.nn.Parameter]]]):
                Input passed into the optimizer representing either a
                :class:`list` of parameter groups or an iterable of parameters;
                if ``None``, then this method assumes the input was
                ``model.parameters()``. This argument is deprecated, and there
                is no need to pass it in anymore. (Default: ``None``)
            optim (Optional[torch.optim.Optimizer]): Optimizer that will load
                the state dict returned by this method. This is the preferred
                argument to use over ``optim_input``. (Default: ``None``)
            group (dist.ProcessGroup): Model\'s process group or ``None`` if
                using the default process group. (Default: ``None``)

        Returns:
            Dict[str, Any]: The full optimizer state dict now remapped to
            flattened parameters instead of unflattened parameters and
            restricted to only include this rank\'s part of the optimizer state.
        '''
    @staticmethod
    def rekey_optim_state_dict(optim_state_dict: dict[str, Any], optim_state_key_type: OptimStateKeyType, model: torch.nn.Module, optim_input: list[dict[str, Any]] | Iterable[torch.nn.Parameter] | None = None, optim: torch.optim.Optimizer | None = None) -> dict[str, Any]:
        '''Re-keys the optimizer state dict ``optim_state_dict`` to use the key type ``optim_state_key_type``.

        This can be used to achieve compatibility between optimizer state dicts from models with FSDP
        instances and ones without.

        To re-key an FSDP full optimizer state dict (i.e. from
        :meth:`full_optim_state_dict`) to use parameter IDs and be loadable to
        a non-wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> wrapped_model, wrapped_optim = ...
            >>> full_osd = FSDP.full_optim_state_dict(wrapped_model, wrapped_optim)
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(full_osd, OptimStateKeyType.PARAM_ID, nonwrapped_model)
            >>> nonwrapped_optim.load_state_dict(rekeyed_osd)

        To re-key a normal optimizer state dict from a non-wrapped model to be
        loadable to a wrapped model::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> nonwrapped_model, nonwrapped_optim = ...
            >>> osd = nonwrapped_optim.state_dict()
            >>> rekeyed_osd = FSDP.rekey_optim_state_dict(osd, OptimStateKeyType.PARAM_NAME, nonwrapped_model)
            >>> wrapped_model, wrapped_optim = ...
            >>> sharded_osd = FSDP.shard_full_optim_state_dict(rekeyed_osd, wrapped_model)
            >>> wrapped_optim.load_state_dict(sharded_osd)

        Returns:
            Dict[str, Any]: The optimizer state dict re-keyed using the
            parameter keys specified by ``optim_state_key_type``.
        '''
    @staticmethod
    def optim_state_dict(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_state_dict: dict[str, Any] | None = None, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
        '''
        Transform the state-dict of an optimizer corresponding to a sharded model.

        The given state-dict can be transformed to one of three types:
        1) full optimizer state_dict, 2) sharded optimizer state_dict, 3) local optimizer state_dict.

        For full optimizer state_dict, all states are unflattened and not sharded.
        Rank0 only and CPU only can be specified via :meth:`state_dict_type` to
        avoid OOM.

        For sharded optimizer state_dict, all states are unflattened but sharded.
        CPU only can be specified via :meth:`state_dict_type` to further save
        memory.

        For local state_dict, no transformation will be performed. But a state
        will be converted from nn.Tensor to ShardedTensor to represent its sharding
        nature (this is not supported yet).

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(model, optim)
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     model, optim, optim_state_dict
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` \'s
                parameters.
            optim_state_dict (Dict[str, Any]): the target optimizer state_dict to
                transform. If the value is None, optim.state_dict() will be used. (
                Default: ``None``)
            group (dist.ProcessGroup): Model\'s process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)

        Returns:
            Dict[str, Any]: A :class:`dict` containing the optimizer state for
            ``model``. The sharding of the optimizer state is based on
            ``state_dict_type``.
        '''
    @staticmethod
    def optim_state_dict_to_load(model: torch.nn.Module, optim: torch.optim.Optimizer, optim_state_dict: dict[str, Any], is_named_optimizer: bool = False, load_directly: bool = False, group: dist.ProcessGroup | None = None) -> dict[str, Any]:
        '''
        Convert an optimizer state-dict so that it can be loaded into the optimizer associated with the FSDP model.

        Given a ``optim_state_dict`` that is transformed through
        :meth:`optim_state_dict`, it gets converted to the flattened optimizer
        state_dict that can be loaded to ``optim`` which is the optimizer for
        ``model``. ``model`` must be sharded by FullyShardedDataParallel.

            >>> # xdoctest: +SKIP("undefined variables")
            >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            >>> from torch.distributed.fsdp import StateDictType
            >>> from torch.distributed.fsdp import FullStateDictConfig
            >>> from torch.distributed.fsdp import FullOptimStateDictConfig
            >>> # Save a checkpoint
            >>> model, optim = ...
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> state_dict = model.state_dict()
            >>> original_osd = optim.state_dict()
            >>> optim_state_dict = FSDP.optim_state_dict(
            >>>     model,
            >>>     optim,
            >>>     optim_state_dict=original_osd
            >>> )
            >>> save_a_checkpoint(state_dict, optim_state_dict)
            >>> # Load a checkpoint
            >>> model, optim = ...
            >>> state_dict, optim_state_dict = load_a_checkpoint()
            >>> FSDP.set_state_dict_type(
            >>>     model,
            >>>     StateDictType.FULL_STATE_DICT,
            >>>     FullStateDictConfig(rank0_only=False),
            >>>     FullOptimStateDictConfig(rank0_only=False),
            >>> )
            >>> model.load_state_dict(state_dict)
            >>> optim_state_dict = FSDP.optim_state_dict_to_load(
            >>>     model, optim, optim_state_dict
            >>> )
            >>> optim.load_state_dict(optim_state_dict)

        Args:
            model (torch.nn.Module): Root module (which may or may not be a
                :class:`FullyShardedDataParallel` instance) whose parameters
                were passed into the optimizer ``optim``.
            optim (torch.optim.Optimizer): Optimizer for ``model`` \'s
                parameters.
            optim_state_dict (Dict[str, Any]): The optimizer states to be loaded.
            is_named_optimizer (bool): Is this optimizer a NamedOptimizer or
                KeyedOptimizer. Only set to True if ``optim`` is TorchRec\'s
                KeyedOptimizer or torch.distributed\'s NamedOptimizer.
            load_directly (bool): If this is set to True, this API will also
                call optim.load_state_dict(result) before returning the result.
                Otherwise, users are responsible to call ``optim.load_state_dict()``
                (Default: ``False``)
            group (dist.ProcessGroup): Model\'s process group across which parameters
                are sharded or ``None`` if using the default process group. (
                Default: ``None``)
        '''
    def register_comm_hook(self, state: object, hook: callable):
        """Register a communication hook.

        This is an enhancement that provides a flexible hook to users where they can specify how FSDP aggregates
        gradients across multiple workers.
        This hook can be used to implement several algorithms like
        `GossipGrad <https://arxiv.org/abs/1803.05880>`_ and gradient compression
        which involve different communication strategies for
        parameter syncs while training with :class:`FullyShardedDataParallel`.

        .. warning ::
            FSDP communication hook should be registered before running an initial forward pass
            and only once.

        Args:
            state (object): Passed to the hook to maintain any state information during the training process.
                            Examples include error feedback in gradient compression,
                            peers to communicate with next in `GossipGrad <https://arxiv.org/abs/1803.05880>`_, etc.
                            It is locally stored by each worker
                            and shared by all the gradient tensors on the worker.
            hook (Callable): Callable, which has one of the following signatures:
                            1) ``hook: Callable[torch.Tensor] -> None``:
                            This function takes in a Python tensor, which represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units).
                            It then performs all necessary processing and returns ``None``;
                            2) ``hook: Callable[torch.Tensor, torch.Tensor] -> None``:
                            This function takes in two Python tensors, the first one represents
                            the full, flattened, unsharded gradient with respect to all variables
                            corresponding to the model this FSDP unit is wrapping
                            (that are not wrapped by other FSDP sub-units). The latter
                            represents a pre-sized tensor to store a chunk of a sharded gradient after
                            reduction.
                            In both cases, callable performs all necessary processing and returns ``None``.
                            Callables with signature 1 are expected to handle gradient communication for a `NO_SHARD` case.
                            Callables with signature 2 are expected to handle gradient communication for sharded cases.

        """
    _flat_param_handle: Incomplete
    _unshard_event: Incomplete
    def _unshard(self, async_op: bool = False): ...
    def _wait_unshard_streams_on_current_stream(self) -> None: ...
    training_state: Incomplete
    @contextlib.contextmanager
    def _use_training_state(self, training_state: TrainingState, handle_training_state: HandleTrainingState): ...
