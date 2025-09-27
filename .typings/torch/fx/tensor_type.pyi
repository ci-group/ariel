from ._compatibility import compatibility as compatibility
from _typeshed import Incomplete
from torch.fx.experimental.unification import Var as Var

class TensorType:
    """
    TensorType defines a type for tensors, which consists of a list of dimensions.
    Example:
        class M(torch.nn.Module):
            def forward(self, x:TensorType((1,2,3, Dyn)), y:TensorType((1,2,3, Dyn))):
                return torch.add(x, y)
    """
    __origin__: Incomplete
    __args__: Incomplete
    def __init__(self, dim) -> None: ...
    def __repr__(self) -> str: ...
    def __eq__(self, other): ...
    @staticmethod
    def __class_getitem__(*args): ...

class _DynType:
    """
    _DynType defines a type which stands for the absence of type information.
    """
    __name__: str
    def __init__(self) -> None: ...
    def __eq__(self, other): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

Dyn: Incomplete

def is_consistent(t1, t2):
    """
    A binary relation denoted by ~ that determines if t1 is consistent with t2.
    The relation is reflexive, symmetric but not transitive.
    returns True if t1 and t2 are consistent and False otherwise.
    Example:
        Dyn ~ TensorType((1,2,3))
        int ~ Dyn
        int ~ int
        TensorType((1,Dyn,3)) ~ TensorType((1,2,3))
    """
def is_more_precise(t1, t2):
    """
    A binary relation denoted by <= that determines if t1 is more precise than t2.
    The relation is reflexive and transitive.
    returns True if t1 is more precise than t2 and False otherwise.
    Example:
        Dyn >= TensorType((1,2,3))
        int >= Dyn
        int >= int
        TensorType((1,Dyn,3)) <= TensorType((1,2,3))
    """
