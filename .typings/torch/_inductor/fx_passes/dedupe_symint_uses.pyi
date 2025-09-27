import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from torch import SymBool as SymBool, SymFloat as SymFloat, SymInt as SymInt
from torch.types import py_sym_types as py_sym_types
from torch.utils._ordered_set import OrderedSet as OrderedSet

@dataclass
class _SymExprHash:
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """
    sym_obj: SymInt | SymFloat | SymBool
    def __hash__(self) -> int: ...
    def __eq__(self, value) -> bool: ...

class _SymHashingDict:
    """
    Wrapper around a dictionary that will convert sym types to hash with _SymExprHash and reuse
    existing sym proxies.

    SymPy hash is not always reliable so optimistically hash sympy expression, and if those fail,
    fallback to symnodes.
    """
    sym_hash_dict: Incomplete
    def __init__(self) -> None: ...
    def __setitem__(self, key, value) -> None: ...
    def __getitem__(self, key): ...
    def __contains__(self, key) -> bool: ...
    def get(self, key, default=None): ...
    def _wrap_to_sym_expr_hash(self, key): ...

def dedupe_symints(graph: torch.fx.Graph):
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """
