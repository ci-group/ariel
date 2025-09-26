import sympy
import torch
from .ir import Pointwise as Pointwise, TensorBox as TensorBox
from .lowering import fallback_handler as fallback_handler, is_integer_type as is_integer_type, register_lowering as register_lowering
from .virtualized import ops as ops

def dense_idx_to_jagged_idx(batch_idx, seq_idx, offsets_loader, jagged_len): ...
def get_inverse_offsets(offsets: TensorBox, jagged_len: int | sympy.Expr, realize: bool = True) -> TensorBox:
    '''
    Returns "inverse_offsets" - the inverse of the offsets array.
    offsets maps batch index (dense) to jagged index (i.e. offset into jagged tensor).
    inverse_offsets maps jagged index to batch index.

    e.g. for offsets [0, 3, 4, 9, 10] this will return
    inverse_offsets = [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]

    For the given offsets, the computed inverse_offsets are cached
    on the first call and reused in the further calls.
    '''
def jagged_idx_to_dense_idx(jagged_idx, inverse_offsets_loader, offsets_loader, batch_size: int | sympy.Expr, max_seq_len: int | sympy.Expr, offsets_dtype: torch.dtype) -> tuple[sympy.Expr, sympy.Expr]: ...
def register_jagged_ops(): ...
