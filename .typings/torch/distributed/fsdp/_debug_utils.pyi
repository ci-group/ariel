import torch
from _typeshed import Incomplete
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum
from torch.distributed.fsdp._common_utils import _apply_to_modules as _apply_to_modules, _get_module_fsdp_state as _get_module_fsdp_state, clean_tensor_name as clean_tensor_name

logger: Incomplete

class SimpleProfiler:
    class Type(str, Enum):
        ALL = 'all'
        ALLGATHER = 'all_gather'
        ALLGATHER_OBJ = 'all_gather_object'
        RESHARDING = 'resharding'
        H2D = 'H2D'
        D2H = 'D2H'
    results: dict[str, float]
    profiling: set[str]
    @classmethod
    def reset(cls) -> None: ...
    @classmethod
    @contextmanager
    def profile(cls, profile_type: str) -> Iterator[None]: ...
    @classmethod
    def dump_and_reset(cls, msg: str) -> None: ...

def _get_sharded_module_tree_with_module_name_to_fqns(model: torch.nn.Module) -> tuple[str, dict[str, list[str]]]:
    """
    It is used for composable fully_shard() code path, it returns
      1. sharded module tree info: each line represents a submodule name that contains the
    submodule's FQN and its submodule class name, if the submodule is sharded by `fully_shard`,
    the submodule name will add a postfix with ' FULLY SHARDED'. Each increased tree
    level adds 4 spaces before the printed name. A printed sharded module tree info for a toy model
    is like this:
        [CompositeModel] FULLY SHARDED
            l1[Linear]
            u1[UnitModule] FULLY SHARDED
                u1.l1[Linear]
                u1.seq[Sequential]
                    u1.seq.0[ReLU]
                    u1.seq.1[Linear]
                    u1.seq.2[ReLU]
                u1.l2[Linear]
            u2[UnitModule] FULLY SHARDED
                u2.l1[Linear]
                u2.seq[Sequential]
                    u2.seq.0[ReLU]
                    u2.seq.1[Linear]
                    u2.seq.2[ReLU]
                u2.l2[Linear]
            l2[Linear]
      2. a dict mapping from the concated module FQN and class name to a list of its managed
    original parameters' FQNs. An example of the dict for the above toy sharded model is like this:
            {'[CompositeModel]': ['l1.weight', 'l1.bias', 'l2.weight', 'l2.bias'],
             'u1[UnitModule]': ['u1.l1.weight', 'u1.l1.bias', 'u1.seq.1.weight', 'u1.seq.1.bias', 'u1.l2.weight', 'u1.l2.bias'],
             'u2[UnitModule]': ['u2.l1.weight', 'u2.l1.bias', 'u2.seq.1.weight', 'u2.seq.1.bias', 'u2.l2.weight', 'u2.l2.bias']
            }
    All FQNs are prefixed starting from ``model``.

    Args:
        model (torch.nn.Module): Root module (which may or may not be passed to
                                 composable `fully_shard()`).
    """
