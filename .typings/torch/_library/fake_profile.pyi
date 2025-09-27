import contextlib
import torch
from _typeshed import Incomplete
from collections.abc import Generator
from dataclasses import dataclass
from torch._library.custom_ops import _maybe_get_opdef as _maybe_get_opdef
from torch.types import FileLike as FileLike
from typing import Any, Callable

log: Incomplete

class MissingOpProfile(RuntimeError):
    """
    This is raised when we don't have an operator profile available for the
    given inputs.
    """

@dataclass(frozen=True)
class TensorMetadata:
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    @staticmethod
    def maybe_from_tensor(t: Any) -> TensorMetadata | None: ...

@dataclass(frozen=True)
class OpProfile:
    args_profile: tuple[TensorMetadata | None]
    out_profile: TensorMetadata | tuple[TensorMetadata]

def _generate_fake_kernel(op_name: str, op_profile: set[OpProfile]) -> Callable: ...
@contextlib.contextmanager
def unsafe_generate_fake_kernels(op_profiles: dict[str, set[OpProfile]]) -> Generator:
    '''
    Registers a fake kernel based on the given operator profiles. This fake
    kernel registration will override any existing fake kernel registrations.

    The input is a dictionary mapping operator names to a set of operator
    profiles, which we will use to generate fake kernels. The operator profiles
    are a record of the input and output tensor metadata. Based on this
    information we will match a given input to the recorded profile, and return
    an output with the same metadata as in the recorded profile. If a profile
    doesn\'t exist then an exception will be thrown.

    The fake kernel generation is considerd unsafe because it relies on the
    rigid, pre-defined operator profiles that do not account for potential
    variations in output behavior. Specifically, the generated kernels assume a
    fixed relationship between input and output ranks. However, in reality, it\'s
    possible that data-dependent operations may produce outputs of different
    ranks even when given inputs of the same rank. The generated fake kernels
    are inflexible and unable to accommodate these nuances, making them
    potentially unsafe.

    Args:
        op_profiles (dict[str, set[OpProfile]]): A dictionary mapping operator
            name to a set of operator profiles from which we will generate fake
            kernels.

    Examples:

        >>> # Example: Registering an op-profile from draft-export
        >>> import torch
        >>> from torch.export._draft_export import draft_export
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args=())
        >>> def foo(x: Tensor, y: Tensor) -> Tensor:
        >>>     return x + y
        >>>
        >>> class M(torch.nn.Module):
        >>>     def forward(self, a, b):
        >>>         res = torch.ops.mylib.foo(a, b)  # no fake impl
        >>>         return res
        >>>
        >>> ep = draft_export(M(), (torch.ones(3, 4), torch.ones(3, 4))
        >>>
        >>> with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
        >>>     decomp = ep.run_decompositions()

    '''
def get_torch_version() -> str: ...
def generate_yaml_from_profiles(op_profiles: dict[str, set[OpProfile]]) -> str:
    """
    Generates a yaml string from the given operator profiles which can be saved
    to a file. The yaml string can be loaded back into an operator profile
    structure using `read_profiles_from_yaml`.
    """
def save_op_profiles(op_profiles: dict[str, set[OpProfile]], f: FileLike) -> None:
    """
    Serializes the given operator profiles into a yaml format and saves it to
    the given file. The operator profile can be loaded back using `load_op_profiles`.
    """
def read_profiles_from_yaml(yaml_str: str) -> dict[str, set[OpProfile]]:
    """
    Reads the yaml saved by `save_op_profiles` and returns the operator profiles.
    """
def load_op_profiles(f: FileLike) -> dict[str, set[OpProfile]]:
    """
    Loads the saved operator profiles from `save_op_profiles`.
    """
