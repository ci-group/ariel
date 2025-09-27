import torch
from _typeshed import Incomplete
from collections.abc import Generator
from types import ModuleType
from typing import Any, Callable, Generic
from typing_extensions import ParamSpec

def _type(self, dtype=None, non_blocking: bool = False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
def _to(self, device, non_blocking: bool = False):
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    """Return the non-blocking flag given the function name and kwargs.

    Args:
        function_name (str): the name of the function being used.
        non_blocking (bool): the default value.
        **kwargs (dict): the kwargs passed to the function.
    """
def _get_restore_location(device):
    """Return the map_location location.

    Used for rebuild functions where the tensor device is distinct from the storage
    """
def _rebuild_tensor(storage, storage_offset, size, stride): ...
def get_tensor_metadata(tensor): ...
def set_tensor_metadata(tensor, metadata) -> None: ...
def _restore_device_fake_mode(tensor): ...
def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None): ...
def _rebuild_tensor_v3(storage, storage_offset, size, stride, requires_grad, backward_hooks, dtype, metadata=None): ...

_sparse_tensors_to_validate: list['torch.Tensor']

def _validate_loaded_sparse_tensors() -> None: ...
def _rebuild_sparse_tensor(layout, data):
    """
    Rebuilds a sparse tensor from its sparse storage representation.

    Args:
        layout (str): The sparse storage layout of the tensor.
        data (tuple): The tensor's sparse storage representation.
    """
def _rebuild_nested_tensor(buffer, sizes, strides, storage_offsets): ...
def _rebuild_device_tensor_from_cpu_tensor(data, dtype, device, requires_grad): ...
def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad): ...
_rebuild_xla_tensor = _rebuild_device_tensor_from_numpy

def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad): ...
def _rebuild_wrapper_subclass(cls, dtype, size, stride, storage_offset, layout, device, requires_grad): ...
def _rebuild_qtensor(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks): ...
def _rebuild_parameter(data, requires_grad, backward_hooks): ...
def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state): ...
def _get_obj_state(obj): ...
def _set_obj_state(obj, state): ...
def _import_dotted_name(name): ...
def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
def _flatten_sparse_tensors(tensors):
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Args:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Args:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
def _take_tensors(tensors, size_limit) -> Generator[Incomplete, None, Incomplete]:
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
def annotate(ret, **kwargs): ...
def render_call(fn, args, kwargs): ...

class KeyErrorMessage(str):
    """str subclass that returns itself in repr"""
    __slots__: Incomplete
    def __repr__(self) -> str: ...

class ExceptionWrapper:
    """Wraps an exception plus traceback to communicate across threads"""
    exc_type: Incomplete
    exc_msg: Incomplete
    where: Incomplete
    def __init__(self, exc_info=None, where: str = 'in background') -> None: ...
    def reraise(self) -> None:
        """Reraises the wrapped exception in the current thread"""

def _get_available_device_type(): ...
def _get_device_attr(get_member): ...
def _get_current_device_index(): ...
def _get_all_device_indices(): ...
def _get_devices_properties(device_ids): ...
def get_current_device_index() -> int:
    """Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.
    Arguments: ``None``
    """
def _get_device_index(device: Any, optional: bool = False, allow_cpu: bool = False) -> int:
    """Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    has index. Note that for a device without a specified index,
    i.e., ``torch.device('xxx')``, this will return the current default
    device of that type if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default
    device of the supported runtime platform if :attr:`optional` is ``True``.
    i.e., the current default CUDA device will be returned if CUDA runtime is supported.
    """
def _handle_complex(tensor):
    """
    Returns a real view of a tensor if complex dtype else just the tensor
    need to check if a UninitializedParameter because otherwise checking is_complex is an error for a LazyModule
    """
def _element_size(dtype):
    """
    Returns the element size for a dtype, in bytes
    """

class _ClassPropertyDescriptor:
    fget: Incomplete
    def __init__(self, fget, fset=None) -> None: ...
    def __get__(self, instance, owner=None): ...

def classproperty(func): ...
def is_compiling() -> bool: ...
def _functionalize_sync(t) -> None: ...
def _get_device_module(device_type: str): ...
def _dummy_type(name: str) -> type: ...

class _LazySeedTracker:
    manual_seed_all_cb: Incomplete
    manual_seed_cb: Incomplete
    call_order: Incomplete
    def __init__(self) -> None: ...
    def queue_seed_all(self, cb, traceback) -> None: ...
    def queue_seed(self, cb, traceback) -> None: ...
    def get_calls(self) -> list: ...

logger: Incomplete
P = ParamSpec('P')

class CallbackRegistry(Generic[P]):
    name: Incomplete
    callback_list: list[Callable[P, None]]
    def __init__(self, name: str) -> None: ...
    def add_callback(self, cb: Callable[P, None]) -> None: ...
    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None: ...

def try_import(module_name: str) -> ModuleType | None: ...

IMPORT_MAPPING: Incomplete
NAME_MAPPING: Incomplete
