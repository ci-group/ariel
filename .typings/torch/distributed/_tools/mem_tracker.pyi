import torch
from _typeshed import Incomplete
from enum import Enum
from torch import nn, optim
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch.utils.weak import weakref
from typing import Any, Callable
from typing_extensions import Self

__all__ = ['MemTracker']

class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""
class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""

class _MemRefType(_RefType):
    """
    An enum to define memory reference types, categorizing tensors based on their usage within a model.

        - PARAM: Tensors registered as nn.Parameter within modules.
        - BUFFER: Tensors registered as nn.Buffer within modules.
        - GRAD: Gradients associated with parameters.
        - ACT: Tensors produced during the forward pass and recomputation in activation checkpointing.
        - TMP: Temporary memory used during the backward pass, including gradients of activations.
        - OPT: Tensors holding optimizer states.
        - OTH: Tensors registered via `track_external` that do not fit the above categories.
    """
    PARAM = 'Parameter'
    BUFFER = 'Buffer'
    GRAD = 'Gradient'
    ACT = 'Activation'
    TEMP = 'Temp'
    OPT = 'Optstate'
    OTH = 'Other'

class _ModState(_State):
    """
    An enum to define the state of a module.

        - PRE_FW: The module is about to run the forward pass.
        - POST_FW: The module has finished running the forward pass.
        - PEAK_FW: The module has reached the peak memory usage during the forward pass.
        - PRE_BW: The module is about to run the backward pass.
        - PRE_FW_AC: The module is about to run the forward pass with activation checkpointing.
        - POST_FW_AC: The module has finished running the forward pass with activation checkpointing.
        - POST_BW: The module has finished running the backward pass.
        - PEAK_BW: The module has reached the peak memory usage during the backward pass.
    """
    PRE_FW = 'Pre-Forward'
    POST_FW = 'Post-Forward'
    PEAK_FW = 'Peak-Forward'
    PRE_BW = 'Pre-Backward'
    PRE_FW_AC = 'Pre-Forward-AC'
    POST_FW_AC = 'Post-Forward-AC'
    POST_BW = 'Post-Backward'
    PEAK_BW = 'Peak-Backward'

class _ModMemStats:
    """
    A class to store the memory statistics of a module.

    Args:
        mod_fqn (str): The fully qualified name of the module.
    Attributes:
        mod_fqn (str): The fully qualified name of the module.
        parameter_mem (int): The memory usage of the parameters of the module.
        buffer_mem (int): The memory usage of the buffers of the module.
        input_mem (int): The memory usage of the inputs to the module.
        output_mem (int): The memory usage of the outputs from the module.
        snapshots (Dict[_ModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states defined by ``_ModState``.
    Note:
        The memory snapshot is stored as a dictionary - Dict[torch.device, Dict[str, int]], where each key is a device,
         and each value is another dictionary with keys as memory reference types defined by `_MemRefType` and
         values as the memory consumed in bytes.
    """
    mod_fqn: Incomplete
    parameter_mem: int
    buffer_mem: int
    input_mem: int
    output_mem: int
    local_peak: dict[torch.device, int]
    snapshots: dict[_ModState, list[dict[torch.device, dict[str, int]]]]
    def __init__(self, mod_fqn: str) -> None: ...

class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """
    size: Incomplete
    element_size: Incomplete
    reftype: Incomplete
    device: Incomplete
    mem_consumed: Incomplete
    def __init__(self, size: int, element_size: int, device: torch.device, reftype: _RefType) -> None:
        """
        Initializes the ``_WeakRefInfo`` object with tensor storage properties.

        Args:
            size (int): The number of elements in the tensor storage.
            element_size (int): The size of each element in the tensor storage.
            device (torch.device): The device on which the tensor is allocated.
            reftype (_RefType): The reference type of the tensor.
        """
    def _calculate_mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
            int: The memory consumed in bytes.
        """
    def update_mem_consumed(self, st: torch.UntypedStorage) -> int:
        """
        Updates and returns the memory consumed if the storage size has changed.

        Args:
            st (torch.UntypedStorage): The tensor storage to check for size updates.

        Returns:
            int: The updated memory consumed in bytes.
        """
    @classmethod
    def create_winfo(cls, st: torch.UntypedStorage, device: torch.device, reftype: _RefType, callback: Callable[[Self, weakref.ref], Any] | None = None) -> tuple[Self, weakref.ref]:
        """
        Creates a new ``_WeakRefInfo`` instance and a weak reference to a ``torch.UntypedStorage`` object,
        optionally attaching a callback to the weak reference.

        Args:
            st (torch.UntypedStorage): The storage object for which to create the weak reference info.
            device (torch.device): The device associated with the storage object.
            reftype (_RefType): The type of reference, used to categorize the storage.
            callback (Optional[Callable[[Self, weakref.ref]]]): A callback function that is called when
                the storage object is about to be finalized (garbage collected). The callback function
                should accept two arguments: the ``_WeakRefInfo`` instance and the weak reference to the storage.
        Returns:
            Tuple[Self, weakref.ref]: A tuple containing the newly created ``_WeakRefInfo`` instance and the
            weak reference to the storage object. The weak reference may have an attached callback if provided.
        """

class _UpdateType(Enum):
    ADD = ...
    DEL = ...
    REF = ...
    SIZE = ...

class MemTracker(TorchDispatchMode):
    '''
    A TorchDispatchMode to track, categorize and attribute the tensor memory created or accessed within its context.

    It categorizes the tracked tensors as parameters, buffers, activations, gradients, temporary memory and optimizer states
    as defined by ``_MemRefType`` within its context. It captures memory `snapshots` for the modules, called within its context,
    at various states defined by ``_ModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key
        is a reference to a module, and each value is a ``_ModMemStats`` object that stores the memory
        statistics of the module.

    Note:
        The MemTracker should be used as a context manager. The modules, optimizers, and any other tensors created within
        the context of MemTracker will be tracked by default. Any tensors or stateful objects such as modules, optimizers etc.
        that need to be tracked but are created outside the MemTracker should be registered using the `track_external` method.
        The `track_external` method should be called before the MemTracker is used. Any tensors created outside the ``MemTracker``
        and not supplied to the `track_external` method will not be tracked by the ``MemTracker``.

    Example usage:

        .. code-block:: python

            module = ...
            optimizer = ...
            inp = ...
            mem_tracker = MemTracker()
            mem_tracker.track_external(module, optimizer, inp)
            with mem_tracker as mt:
                loss = module(inp)
                print("After Forward:")
                mt.display_snapshot("current")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mt.display_snapshot("peak")
            mt.display_modulewise_snapshots(depth=3, units="MiB")

    Known Limitations:
        - The ``MemTracker`` does not track memory for tensors that bypass the ``TorchDispatchMode`` ex. under ``no_dispatch``.
        - Resizing tensor storages directly by using non-Tensor methods other than using ``torch.Untyped_Storage.resize_``
          is not tracked. File a Github issue if you have use-cases for this.
        - If the tensors are not traceable or wrappable subclasses of ``torch.Tensor``, then the tracker does not know how to
            track their storages. File a Github issue if you have use-cases for this.
        - During AC in the backward pass there might be misattribution between activation and temp memory, but the peak memory
          will be tracked accurately. This will be fixed in the next update by hooking intricately with ``torch.uitls.checkpoint``.
    '''
    memory_tracking: Incomplete
    _curr_mem_snap: dict[torch.device, dict[str, int]]
    _peak_mem: dict[torch.device, int]
    _peak_mem_snap: dict[torch.device, dict[str, int]]
    _param_to_grad_hook_handles: Incomplete
    _optimizer_hook_handles: tuple[RemovableHandle, RemovableHandle] | None
    _WINFO: Incomplete
    _mod_tracker: Incomplete
    _ref_class: type[_RefType]
    _in_opt: bool
    _in_ac: bool
    _ac_mod: weakref.ref | None
    _orig_resize: Incomplete
    _orig_dtensor_dispatch: Incomplete
    _depth: int
    def __init__(self) -> None: ...
    def _update_snap(self, u_type: _UpdateType, winfo: _WeakRefInfo, old_mem_consumed: int | None = None, old_reftype: _RefType | None = None) -> None: ...
    def _update_and_maybe_create_winfos(self, t: torch.Tensor, reftype: _RefType, update_existing: bool = False) -> set[_WeakRefInfo]: ...
    def _delete_callback(self, winfo: _WeakRefInfo, w_st: weakref.ref) -> None: ...
    def _track_resize(self) -> None: ...
    def _restore_resize(self) -> None: ...
    def _update_peak_stats(self, peak_state: _State) -> None: ...
    def _track(self, reftype: _RefType, t: torch.Tensor) -> None: ...
    def get_tracker_snapshot(self, type: str = 'current') -> dict[torch.device, dict[str, int]]:
        '''
        Capture a snapshot of the memory usage breakdown per device, based on the specified type.

        Args:
            type (str): The type of snapshot to capture. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
        Returns:
            Dict[torch.device, Dict[str, int]]: A dictionary where each key is a torch.device, and each value is another
                                                dictionary. This inner dictionary has keys representing memory reference
                                                types as defined in ``_MemRefType`` and values representing the amount of
                                                memory consumed in bytes.
        Raises:
            ValueError: If an invalid type is specified.
        '''
    def _track_module_params_and_buffers(self, module: nn.Module, install_grad_hooks: bool = True) -> tuple[int, int]: ...
    def _track_inputs_or_outputs(self, args: Any) -> int: ...
    def _pre_fw_hook(self, module: nn.Module, inputs: Any) -> None: ...
    def _post_fw_hook(self, module: nn.Module, inputs: Any, outputs: Any) -> None: ...
    def _pre_bw_hook(self, module: nn.Module, args: Any) -> None: ...
    def _post_bw_hook(self, module: nn.Module, args: Any) -> None: ...
    def _track_optimizer_states(self, reftype: _RefType, optimizer: optim.Optimizer) -> None: ...
    def _register_global_optimizer_hook(self) -> None: ...
    def _deregister_param_and_optimizer_hooks(self) -> None: ...
    def track_external(self, *external: nn.Module | optim.Optimizer | torch.Tensor) -> None:
        """
        Track tensors and stateful objects like modules, optimizers etc. that are created outside the MemTracker.

        This method should be called before the ``MemTracker`` is used. Any tensors that are not module parameters, buffers,
        gradients activations, or optimizer states will be categorized as ``Other``. If you want them categorized with a
        custom name, please file a GitHub issue. Any tensors created outside the MemTracker and not supplied to this
        method will not be be tracked by ``MemTracker``.

        Args:
            *external (Union[nn.Module, optim.Optimizer, torch.Tensor]): The external modules, optimizers, and
                                                                         tensors to be tracked.
        """
    def display_snapshot(self, type: str = 'current', units: str = 'B', tabulate: bool = False) -> None:
        '''
        Display the memory usage breakdown snapshot of the tracker based on the specified type and units.

        Keyword args:
            type (str): The type of snapshot to display. Can be "current" for the current memory usage or "peak" for the
                        peak memory usage. Defaults to "current".
            units (str): The units to use for displaying memory usage. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool): Whether to display the snapshot in a tabular format. Defaults to False.
        '''
    def display_modulewise_snapshots(self, depth: int = 2, units: str = 'B', tabulate: bool = False) -> None:
        '''
        Print per device memory breakdown snapshot for each module called within MemTracker.

        Snapshots are displayed for the states defined by ``_ModState``.
        The module hierarchy is displayed up to the specified depth.

        Keyword Args:
            depth (int, optional): The depth of the module hierarchy to display. Defaults to 2.
            units (str, optional): The units to use for memory tracking. Defaults to "B". Supports ["B", "KiB", "MiB", "GiB"].
            tabulate (bool, optional): Whether to display the snapshot in a tabular format. Defaults to False.
        '''
    def reset_mod_stats(self) -> None:
        """
        Reset all the module memory stats. Clears ``memory_tracking`` dictionary.
        """
    def _track_dtensor_dispatch(self) -> None: ...
    def _restore_dtensor_dispatch(self) -> None: ...
    def __enter__(self) -> MemTracker: ...
    def __exit__(self, *args: Any) -> None: ...
    def __torch_dispatch__(self, func, types, args=(), kwargs=None): ...
