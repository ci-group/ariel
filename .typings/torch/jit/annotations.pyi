from _typeshed import Incomplete
from torch._C import AnyType as AnyType, ComplexType as ComplexType, DictType as DictType, FloatType as FloatType, IntType as IntType, ListType as ListType, StringType as StringType, TensorType as TensorType, TupleType as TupleType
from torch._jit_internal import Any as Any, BroadcastingList1 as BroadcastingList1, BroadcastingList2 as BroadcastingList2, BroadcastingList3 as BroadcastingList3, Dict as Dict, List as List, Tuple as Tuple, is_dict as is_dict, is_list as is_list, is_optional as is_optional, is_tuple as is_tuple, is_union as is_union

__all__ = ['Any', 'List', 'BroadcastingList1', 'BroadcastingList2', 'BroadcastingList3', 'Tuple', 'is_tuple', 'is_list', 'Dict', 'is_dict', 'is_optional', 'is_union', 'TensorType', 'TupleType', 'FloatType', 'ComplexType', 'IntType', 'ListType', 'StringType', 'DictType', 'AnyType', 'Module', 'get_signature', 'check_fn', 'get_param_names', 'parse_type_line', 'get_type_line', 'split_type_line', 'try_real_annotations', 'try_ann_to_type', 'ann_to_type']

class Module:
    name: Incomplete
    members: Incomplete
    def __init__(self, name, members) -> None: ...
    def __getattr__(self, name): ...

class EvalEnv:
    env: Incomplete
    rcb: Incomplete
    def __init__(self, rcb) -> None: ...
    def __getitem__(self, name): ...

def get_signature(fn, rcb, loc, is_method): ...
def get_param_names(fn, n_args): ...
def check_fn(fn, loc) -> None: ...
def parse_type_line(type_line, rcb, loc):
    """Parse a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
def get_type_line(source):
    """Try to find the line containing a comment with the type annotation."""
def split_type_line(type_line):
    '''Split the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        ("(Tensor, torch.Tensor)", "Tuple[Tensor, Tensor]")

    '''
def try_real_annotations(fn, loc):
    """Try to use the Py3.5+ annotation syntax to get the type."""
def try_ann_to_type(ann, loc, rcb=None): ...
def ann_to_type(ann, loc, rcb=None): ...
