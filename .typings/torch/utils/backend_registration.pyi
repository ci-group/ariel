import torch

__all__ = ['rename_privateuse1_backend', 'generate_methods_for_privateuse1_backend']

def rename_privateuse1_backend(backend_name: str) -> None:
    '''
    Rename the privateuse1 backend device to make it more convenient to use as a device name within PyTorch APIs.

    The steps are:

    (1) (In C++) implement kernels for various torch operations, and register them
        to the PrivateUse1 dispatch key.
    (2) (In python) call torch.utils.rename_privateuse1_backend("foo")

    You can now use "foo" as an ordinary device string in python.

    Note: this API can only be called once per process. Attempting to change
    the external backend after it\'s already been set will result in an error.

    Note(AMP): If you want to support AMP on your device, you can register a custom backend module.
    The backend must register a custom backend module with ``torch._register_device_module("foo", BackendModule)``.
    BackendModule needs to have the following API\'s:

    (1) ``get_amp_supported_dtype() -> List[torch.dtype]``
        get the supported dtypes on your "foo" device in AMP, maybe the "foo" device supports one more dtype.

    Note(random): If you want to support to set seed for your device, BackendModule needs to have the following API\'s:

    (1) ``_is_in_bad_fork() -> bool``
        Return ``True`` if now it is in bad_fork, else return ``False``.

    (2) ``manual_seed_all(seed int) -> None``
        Sets the seed for generating random numbers for your devices.

    (3) ``device_count() -> int``
        Returns the number of "foo"s available.

    (4) ``get_rng_state(device: Union[int, str, torch.device] = \'foo\') -> Tensor``
        Returns a list of ByteTensor representing the random number states of all devices.

    (5) ``set_rng_state(new_state: Tensor, device: Union[int, str, torch.device] = \'foo\') -> None``
        Sets the random number generator state of the specified "foo" device.

    And there are some common funcs:

    (1) ``is_available() -> bool``
        Returns a bool indicating if "foo" is currently available.

    (2) ``current_device() -> int``
        Returns the index of a currently selected device.

    For more details, see https://pytorch.org/tutorials/advanced/extend_dispatcher.html#get-a-dispatch-key-for-your-backend
    For an existing example, see https://github.com/bdhirsh/pytorch_open_registration_example

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        # This will work, assuming that you\'ve implemented the right C++ kernels
        # to implement torch.ones.
        >>> a = torch.ones(2, device="foo")

    '''
def generate_methods_for_privateuse1_backend(for_tensor: bool = True, for_module: bool = True, for_packed_sequence: bool = True, for_storage: bool = False, unsupported_dtype: list[torch.dtype] | None = None) -> None:
    '''
    Automatically generate attributes and methods for the custom backend after rename privateuse1 backend.

    In the default scenario, storage-related methods will not be generated automatically.

    When you implement kernels for various torch operations, and register them to the PrivateUse1 dispatch key.
    And call the function torch.rename_privateuse1_backend("foo") to rename your backend name.
    At this point, you can easily register specific methods and attributes by calling this function.
    Just like torch.Tensor.foo(), torch.Tensor.is_foo, torch.Storage.foo(), torch.Storage.is_foo.

    Note: We recommend you use generic functions (check devices are equal or to(device=)).
    We provide these methods for convenience only and they will be "monkey patched" onto the objects
    and so will not be properly typed. For Storage methods generate, if you need to support sparse data storage,
    you need to extend the implementation yourself.

    Args:
        for_tensor (bool): whether register related methods for torch.Tensor class.
        for_module (bool): whether register related methods for torch.nn.Module class.
        for_storage (bool): whether register related methods for torch.Storage class.
        unsupported_dtype (List[torch.dtype]): takes effect only when the storage method needs to be generated,
            indicating that the storage does not support the torch.dtype type.

    Example::

        >>> # xdoctest: +SKIP("failing")
        >>> torch.utils.rename_privateuse1_backend("foo")
        >>> torch.utils.generate_methods_for_privateuse1_backend()
        # Then automatically generate backend-related attributes and methods.
        >>> a = torch.tensor(2).foo()
        >>> a.is_foo
        >>> hasattr(torch.nn.Module, \'foo\')
    '''
