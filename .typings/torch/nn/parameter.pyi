import torch
from _typeshed import Incomplete
from torch._C import _disabled_torch_function_impl as _disabled_torch_function_impl

class _ParameterMeta(torch._C._TensorMeta):
    def __instancecheck__(self, instance): ...

class Parameter(torch.Tensor, metaclass=_ParameterMeta):
    """A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad: bool = True): ...
    def __deepcopy__(self, memo): ...
    def __repr__(self) -> str: ...
    def __reduce_ex__(self, proto): ...
    __torch_function__ = _disabled_torch_function_impl

class UninitializedTensorMixin:
    _allowed_methods: Incomplete
    data: Incomplete
    __class__: Incomplete
    def materialize(self, shape, device=None, dtype=None) -> None:
        """Create a Parameter or Tensor with the same properties of the uninitialized one.

        Given a shape, it materializes a parameter in the same device
        and with the same `dtype` as the current one or the specified ones in the
        arguments.

        Args:
            shape : (tuple): the shape for the materialized tensor.
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module. Optional.
            dtype (:class:`torch.dtype`): the desired floating point type of
                the floating point parameters and buffers in this module. Optional.
        """
    @property
    def shape(self) -> None: ...
    def share_memory_(self) -> None: ...
    def __repr__(self) -> str: ...
    def __reduce_ex__(self, proto): ...
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None): ...

def is_lazy(param): ...

class UninitializedParameter(UninitializedTensorMixin, Parameter):
    """A parameter that is not initialized.

    Uninitialized Parameters are a special case of :class:`torch.nn.Parameter`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.nn.Parameter`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.nn.Parameter`.

    The default device or dtype to use when the parameter is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    cls_to_become = Parameter
    def __new__(cls, requires_grad: bool = True, device=None, dtype=None) -> None: ...
    def __deepcopy__(self, memo): ...

class _BufferMeta(torch._C._TensorMeta):
    def __instancecheck__(self, instance): ...

class Buffer(torch.Tensor, metaclass=_BufferMeta):
    """A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the module's state.

    Buffers are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s -- when they're
    assigned as Module attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~torch.nn.Module.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~torch.nn.Module.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the module's
            :attr:`state_dict`. Default: ``True``
    """
    def __new__(cls, data=None, *, persistent: bool = True): ...
    __torch_function__ = _disabled_torch_function_impl

class UninitializedBuffer(UninitializedTensorMixin, torch.Tensor):
    """A buffer that is not initialized.

    Uninitialized Buffer is a a special case of :class:`torch.Tensor`
    where the shape of the data is still unknown.

    Unlike a :class:`torch.Tensor`, uninitialized parameters
    hold no data and attempting to access some properties, like their shape,
    will throw a runtime error. The only operations that can be performed on a uninitialized
    parameter are changing its datatype, moving it to a different device and
    converting it to a regular :class:`torch.Tensor`.

    The default device or dtype to use when the buffer is materialized can be set
    during construction using e.g. ``device='cuda'``.
    """
    cls_to_become = torch.Tensor
    def __new__(cls, requires_grad: bool = False, device=None, dtype=None, persistent: bool = True) -> None: ...
