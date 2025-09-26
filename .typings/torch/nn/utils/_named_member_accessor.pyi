import torch
from _typeshed import Incomplete
from collections.abc import Iterable

_MISSING: torch.Tensor

def set_tensor(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None: ...
def swap_tensor(module: torch.nn.Module, name: str, tensor: torch.Tensor, allow_missing: bool = False) -> torch.Tensor: ...
def swap_submodule(module: torch.nn.Module, name: str, submodule: torch.nn.Module) -> torch.nn.Module: ...

class NamedMemberAccessor:
    """
    A class that provides a way to access the submodules and parameters/buffers of a module.

    It provides caching mechanism to speed up submodule lookups.
    This is useful for functional programming to manipulate the module state.
    """
    module: Incomplete
    memo: dict[str, torch.nn.Module]
    def __init__(self, module: torch.nn.Module) -> None: ...
    def get_submodule(self, name: str) -> torch.nn.Module:
        '''
        Return the submodule specified by the given path.

        For example, to get the submodule mod.layer1.conv1,
        use accessor.get_submodule("layer1.conv1")

        Compare to mod.get_submodule("layer1.conv1"), this method will cache the
        intermediate submodule access to speed up future lookups.
        '''
    def swap_submodule(self, path: str, value: torch.nn.Module) -> torch.nn.Module:
        '''
        Swap the submodule specified by the given ``path`` to ``value``.

        For example, to swap the attribute mod.layer1.conv1 use
        ``accessor.swap_submodule("layer1.conv1", conv2)``.
        '''
    def get_tensor(self, name: str) -> torch.Tensor:
        '''
        Get the tensor specified by the given path to value.

        For example, to get the attribute mod.layer1.conv1.weight,
        use accessor.get_tensor(\'layer1.conv1.weight\')

        Compare to mod.get_parameter("layer1.conv1.weight"), this method will
        cache the intermediate submodule access to speed up future lookups.
        '''
    def set_tensor(self, name: str, value: torch.Tensor) -> None:
        '''
        Set the attribute specified by the given path to value.

        For example, to set the attribute mod.layer1.conv1.weight,
        use accessor.set_tensor("layer1.conv1.weight", value)
        '''
    def del_tensor(self, name: str) -> None:
        '''
        Delete the attribute specified by the given path.

        For example, to delete the attribute mod.layer1.conv1.weight,
        use accessor.del_tensor("layer1.conv1.weight")
        '''
    def swap_tensor(self, name: str, value: torch.Tensor, allow_missing: bool = False) -> torch.Tensor:
        '''
        Swap the attribute specified by the given path to value.

        For example, to swap the attribute mod.layer1.conv1.weight,
        use accessor.swap_tensor("layer1.conv1.weight", value)
        '''
    def get_tensors(self, names: Iterable[str]) -> list[torch.Tensor]:
        '''
        Get the tensors specified by the given paths.

        For example, to get the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.get_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        '''
    def set_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor]) -> None:
        '''
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        '''
    def set_tensors_dict(self, named_tensors: dict[str, torch.Tensor]) -> None:
        '''
        Set the attributes specified by the given paths to values.

        For example, to set the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.set_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        '''
    def del_tensors(self, names: Iterable[str]) -> None:
        '''
        Delete the attributes specified by the given paths.

        For example, to delete the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.del_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"])
        '''
    def swap_tensors(self, names: Iterable[str], values: Iterable[torch.Tensor], allow_missing: bool = False) -> list[torch.Tensor]:
        '''
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors(["layer1.conv1.weight",
        "layer1.conv1.bias"], [weight, bias])
        '''
    def swap_tensors_dict(self, named_tensors: dict[str, torch.Tensor], allow_missing: bool = False) -> tuple[dict[str, torch.Tensor], list[str]]:
        '''
        Swap the attributes specified by the given paths to values.

        For example, to swap the attributes mod.layer1.conv1.weight and
        mod.layer1.conv1.bias, use accessor.swap_tensors_dict({
            "layer1.conv1.weight": weight,
            "layer1.conv1.bias": bias,
        })
        '''
    def check_keys(self, keys: Iterable[str]) -> tuple[list[str], list[str]]:
        """Check that the given keys are valid."""
    def named_parameters(self, remove_duplicate: bool = True) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the parameters in the module."""
    def named_buffers(self, remove_duplicate: bool = True) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the buffers in the module."""
    def named_tensors(self, remove_duplicate: bool = True) -> Iterable[tuple[str, torch.Tensor]]:
        """Iterate over all the tensors in the module."""
    def named_modules(self, remove_duplicate: bool = True) -> Iterable[tuple[str, 'torch.nn.Module']]:
        """Iterate over all the modules in the module."""
