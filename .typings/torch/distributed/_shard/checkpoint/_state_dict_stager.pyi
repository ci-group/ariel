from _typeshed import Incomplete
from torch.storage import UntypedStorage as UntypedStorage
from torch.utils.weak import WeakIdKeyDictionary as WeakIdKeyDictionary
from typing import Any

logger: Incomplete

class StateDictStager:
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager stages the state_dict to CPU DRAM while applying optimizations
    like memory sharing and pinning to improve performance. It caches storage objects
    to avoid redundant copies and can be configured to automatically share memory
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        _cached_storage_mapping (WeakIdKeyDictionary): Maps storage objects to optimized CPU storages using weak references
    """
    pin_memory: bool
    share_memory: Incomplete
    _cached_storage_mapping: Incomplete
    _deepcopy_dispatch: Incomplete
    def __init__(self, pin_memory: bool = False, share_memory: bool = False) -> None: ...
    def _stage_untyped_storage(self, storage: UntypedStorage, non_blocking: bool = False):
        """
        Called from the hooked storage_deepcopy function in torch.Tensor.__deepcopy__.

        This method handles the storage optimization logic for the StagingStateDict class.
        It checks if the storage has already been cached, and if so, reuses it.
        Otherwise, it creates a new CPU storage and applies memory optimizations.

        Args:
            storage: The storage to optimize

        Returns:
            The optimized storage
        """
    def stage(self, state_dict: dict[str, Any], non_blocking: bool = False) -> dict[str, Any]: ...
    def _offload_tensor(self, x, memo, non_blocking: bool = False):
        """
        Deep copy a PyTorch tensor with optimized storage handling.

        This method creates a CPU copy of a tensor while applying memory optimizations
        like sharing and pinning based on the StateDictStager configuration.

        Args:
            x: The tensor to copy
            memo: Memo dictionary for tracking already copied objects
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A CPU copy of the tensor with optimized storage
        """
    def deepcopy_with_tensor_offload(self, x, memo=None, _nil=[], non_blocking: bool = False):
        """Deep copy operation on arbitrary Python objects with special handling for PyTorch tensors.

        This implementation extends the standard deepcopy functionality to handle PyTorch tensors
        and their storages in a way that optimizes memory usage and performance, similar to the
        stage method. It applies memory sharing and pinning optimizations based on the StateDictStager
        configuration.

        Args:
            x: The object to deep copy
            memo: Memo dictionary for tracking already copied objects
            _nil: Sentinel value for memo dictionary
            non_blocking: Whether to perform non-blocking copies where possible

        Returns:
            A deep copy of the input object with optimized tensor storage handling
        """
    def _keep_alive(self, x, memo) -> None:
        """Keeps a reference to the object x in the memo.

        Because we remember objects by their id, we have
        to assure that possibly temporary objects are kept
        alive by referencing them.
        We store a reference at the id of the memo, which should
        normally not be used unless someone tries to deepcopy
        the memo itself...
        """
    def _reconstruct(self, x, memo, func, args, state=None, listiter=None, dictiter=None): ...
