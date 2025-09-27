import dataclasses
import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from torch.utils._python_dispatch import is_traceable_wrapper_subclass as is_traceable_wrapper_subclass
from typing import Any

@dataclasses.dataclass
class SubclassCreationMeta:
    start_idx: int
    num_tensors: int
    class_type: Any
    attrs: dict[str, 'SubclassCreationMeta']
    metadata: Any
    outer_size: Iterable[None | int | torch.SymInt]
    outer_stride: Iterable[None | int | torch.SymInt]

class UnwrapTensorSubclass(torch.nn.Module):
    def forward(self, *tensors) -> torch.Tensor: ...
    subclass_meta: Incomplete
    def right_inverse(self, tensor: torch.Tensor) -> list[torch.Tensor]: ...

def unwrap_tensor_subclass_parameters(module: torch.nn.Module) -> torch.nn.Module:
    '''
    Model transformation that replaces all the parameters that are subclasses to plain tensors.
    This reduces runtime overhead of flattening/unflattening the parameters.

    This transformation adds parametrization with `torch.nn.utils.parametrize`.
    The FQNs of the subclass parameters will be changed and state_dict will become incompatible with the original model.
    E.g.
    Original model state_dict: {"p1": torch.testing._internal.TwoTensor}
    becomes: {"parametrizations.p2.original0": torch.Tensor, "parametrizations.p2.original1": torch.Tensor}

    '''
