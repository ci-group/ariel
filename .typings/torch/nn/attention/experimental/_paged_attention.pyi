import torch
from _typeshed import Incomplete
from torch.nn.attention.flex_attention import BlockMask, _mask_mod_signature, _score_mod_signature

__all__ = ['PagedAttention']

class PagedAttention:
    """
    PagedAttention supports flex attention inference with a large batch size.
    With PagedAttention, a batch of key/value tensors with varying kv length
    is splitted into tensor blocks of fixed length and cached in a compact way.
    Thus we can avoid redundant memory consumption due to varying kv length and
    support a larger batch size.
    """
    n_pages: Incomplete
    page_size: Incomplete
    page_table: Incomplete
    capacity: Incomplete
    empty_pages: Incomplete
    physical_to_logical: Incomplete
    def __init__(self, n_pages: int, page_size: int, max_batch_size: int, device: str = 'cuda') -> None: ...
    def reserve(self, batch_idx: torch.Tensor, seq_len: torch.Tensor) -> None:
        """
        Requests the capacity of a given batch to be at least enough to
        hold `seq_len` elements.

        Args:
            batch_idx (Tensor): batch index to be reserved; shape :math:`(1)`.
            seq_len (Tensor): minimum capacity for the given batch; shape :math:`(1)`.
        """
    def erase(self, batch_idx: torch.Tensor) -> None:
        """
        Removes a single batch from paged attention.

        Args:
            batch_idx (Tensor): batch index to be removed; shape :math:`(1)`.
        """
    def assign(self, batch_idx: torch.Tensor, input_pos: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor) -> None:
        """
        Assigns new contents `val` to the storage `cache` at the location
        `batch_idx` and `input_pos`.

        Args:
            batch_idx (Tensor): batch index; shape :math:`(B)`.
            input_pos (Tensor): input positions to be assigned for the given batch; shape :math:`(B, S)`.
            val (Tensor): value to be assigned; shape :math:`(B, H, S, D)`
            cache (Tensor): the cache to store the values; shape:`(1, H, MAX_S, D)`
        """
    def convert_logical_block_mask(self, block_mask: BlockMask, batch_idx: torch.Tensor | None = None) -> BlockMask:
        """
        Converts a logical block mask by mapping its logical kv indices to the corresponding
        physical kv indices.

        Args:
            block_mask (BlockMask): logical block mask;
                kv_indices shape :math:`(B, H, ROWS, MAX_BLOCKS_IN_COL)`.
            batch_idx (Tensor): batch index corresponding to the block_mask
                batch dimension. This provides flexibility to convert a
                block mask with smaller batch size than the page table;
                shape :math:`(B)`.
        """
    def get_mask_mod(self, mask_mod: _mask_mod_signature | None) -> _mask_mod_signature:
        """
        Converts a mask_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            mask_mod (_mask_mod_signature): mask_mod based on the logical block index.
        """
    def get_score_mod(self, score_mod: _score_mod_signature | None) -> _score_mod_signature:
        """
        Converts a score_mod based on mapping from the physical block index to the logical
        block index.

        Args:
            score_mod (_score_mod_signature): score_mod based on the logical block index.
        """
