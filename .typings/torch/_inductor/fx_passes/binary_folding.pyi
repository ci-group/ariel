import functools
from .. import config as config
from ..._dynamo.utils import counters as counters
from ..pattern_matcher import Arg as Arg, CallFunction as CallFunction, KeywordArg as KeywordArg
from .freezing_patterns import register_binary_folding_pattern as register_binary_folding_pattern
from _typeshed import Incomplete

aten: Incomplete
prims: Incomplete

def mark_mixed_dtype(computation_node) -> None: ...
def mark_mixed_dtype_allowed_computation_ops(gm) -> None:
    """
    Mark convolutions/linear which we will binary fold even with mixed precision constants. We constant fold in the higher precision
    for better accuracy and then recover the original precision after.
    """
def recover_original_precision_folded_computation_ops(gm) -> None:
    """
    After binary folding conv/linear weights and biases to a higher dtype, recover the original precision they were in.
    """

_binary_ops: Incomplete

@functools.cache
def binary_folding_init(): ...
