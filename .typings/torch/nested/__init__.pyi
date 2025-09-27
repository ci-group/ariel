from _typeshed import Incomplete
from torch import Tensor
from torch.types import _device as Device, _dtype as DType

__all__ = ['to_padded_tensor', 'as_nested_tensor', 'nested_tensor', 'nested_tensor_from_jagged', 'narrow', 'masked_select']

def as_nested_tensor(ts: Tensor | list[Tensor] | tuple[Tensor, ...], dtype: DType | None = None, device: Device | None = None, layout=None) -> Tensor:
    """
    Constructs a nested tensor preserving autograd history from a tensor or a list / tuple of
    tensors.

    If a nested tensor is passed, it will be returned directly unless the device / dtype / layout
    differ. Note that converting device / dtype will result in a copy, while converting layout
    is not currently supported by this function.

    If a non-nested tensor is passed, it is treated as a batch of constituents of consistent size.
    A copy will be incurred if the passed device / dtype differ from those of the input OR if
    the input is non-contiguous. Otherwise, the input's storage will be used directly.

    If a tensor list is provided, tensors in the list are always copied during construction of
    the nested tensor.

    Args:
        ts (Tensor or List[Tensor] or Tuple[Tensor]): a tensor to treat as a nested tensor OR a
            list / tuple of tensors with the same ndim

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.as_nested_tensor([a, b])
        >>> nt.is_leaf
        False
        >>> fake_grad = torch.nested.nested_tensor([torch.ones_like(a), torch.zeros_like(b)])
        >>> nt.backward(fake_grad)
        >>> a.grad
        tensor([1., 1., 1.])
        >>> b.grad
        tensor([0., 0., 0., 0., 0.])
        >>> c = torch.randn(3, 5, requires_grad=True)
        >>> nt2 = torch.nested.as_nested_tensor(c)
    """

to_padded_tensor: Incomplete

def nested_tensor(tensor_list, *, dtype=None, layout=None, device=None, requires_grad: bool = False, pin_memory: bool = False) -> Tensor:
    '''
    Constructs a nested tensor with no autograd history (also known as a "leaf tensor", see
    :ref:`Autograd mechanics <autograd-mechanics>`) from :attr:`tensor_list` a list of tensors.

    Args:
        tensor_list (List[array_like]): a list of tensors, or anything that can be passed to torch.tensor,
        where each element of the list has the same dimensionality.

    Keyword arguments:
        dtype (:class:`torch.dtype`, optional): the desired type of returned nested tensor.
            Default: if None, same :class:`torch.dtype` as leftmost tensor in the list.
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.
        device (:class:`torch.device`, optional): the desired device of returned nested tensor.
            Default: if None, same :class:`torch.device` as leftmost tensor in the list
        requires_grad (bool, optional): If autograd should record operations on the
            returned nested tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned nested tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.

    Example::

        >>> a = torch.arange(3, dtype=torch.float, requires_grad=True)
        >>> b = torch.arange(5, dtype=torch.float, requires_grad=True)
        >>> nt = torch.nested.nested_tensor([a, b], requires_grad=True)
        >>> nt.is_leaf
        True
    '''
def narrow(tensor: Tensor, dim: int, start: int | Tensor, length: int | Tensor, layout=...) -> Tensor:
    """
    Constructs a nested tensor (which might be a view) from :attr:`tensor`, a strided tensor. This follows
    similar semantics to torch.Tensor.narrow, where in the :attr:`dim`-th dimension the new nested tensor
    shows only the elements in the interval `[start, start+length)`. As nested representations
    allow for a different `start` and `length` at each 'row' of that dimension, :attr:`start` and :attr:`length`
    can also be tensors of shape `tensor.shape[0]`.

    There's some differences depending on the layout you use for the nested tensor. If using strided layout,
    torch.narrow will do a copy of the narrowed data into a contiguous NT with strided layout, while
    jagged layout narrow() will create a non-contiguous view of your original strided tensor. This particular
    representation is really useful for representing kv-caches in Transformer models, as specialized
    SDPA kernels can deal with format easily, resulting in performance improvements.


    Args:
        tensor (:class:`torch.Tensor`): a strided tensor, which will be used as the underlying data
            for the nested tensor if using the jagged layout or will be copied for the strided layout.
        dim (int): the dimension where narrow will be applied. Only `dim=1` is supported for the
            jagged layout, while strided supports all dim
        start (Union[int, :class:`torch.Tensor`]): starting element for the narrow operation
        length (Union[int, :class:`torch.Tensor`]): number of elements taken during the narrow op

    Keyword arguments:
        layout (:class:`torch.layout`, optional): the desired layout of returned nested tensor.
            Only strided and jagged layouts are supported. Default: if None, the strided layout.

    Example::

        >>> starts = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        >>> lengths = torch.tensor([3, 2, 2, 1, 5], dtype=torch.int64)
        >>> narrow_base = torch.randn(5, 10, 20)
        >>> nt_narrowed = torch.nested.narrow(narrow_base, 1, starts, lengths, layout=torch.jagged)
        >>> nt_narrowed.is_contiguous()
        False
    """
def nested_tensor_from_jagged(values: Tensor, offsets: Tensor | None = None, lengths: Tensor | None = None, jagged_dim: int | None = None, min_seqlen: int | None = None, max_seqlen: int | None = None) -> Tensor:
    '''
    Constructs a jagged layout nested tensor from the given jagged components. The jagged layout
    consists of a required values buffer with the jagged dimension packed into a single dimension.
    The offsets / lengths metadata determines how this dimension is split into batch elements
    and are expected to be allocated on the same device as the values buffer.

    Expected metadata formats:
        * offsets: Indices within the packed dimension splitting it into heterogeneously-sized
          batch elements. Example: [0, 2, 3, 6] indicates that a packed jagged dim of size 6
          should be conceptually split into batch elements of length [2, 1, 3]. Note that both the
          beginning and ending offsets are required for kernel convenience (i.e. shape batch_size + 1).
        * lengths: Lengths of the individual batch elements; shape == batch_size. Example: [2, 1, 3]
          indicates that a packed jagged dim of size 6 should be conceptually split into batch
          elements of length [2, 1, 3].

    Note that it can be useful to provide both offsets and lengths. This describes a nested tensor
    with "holes", where the offsets indicate the start position of each batch item and the length
    specifies the total number of elements (see example below).

    The returned jagged layout nested tensor will be a view of the input values tensor.

    Args:
        values (:class:`torch.Tensor`): The underlying buffer in the shape of
            (sum_B(*), D_1, ..., D_N). The jagged dimension is packed into a single dimension,
            with the offsets / lengths metadata used to distinguish batch elements.
        offsets (optional :class:`torch.Tensor`): Offsets into the jagged dimension of shape B + 1.
        lengths (optional :class:`torch.Tensor`): Lengths of the batch elements of shape B.
        jagged_dim (optional int): Indicates which dimension in values is the packed jagged
            dimension. If None, this is set to dim=1 (i.e. the dimension immediately following
            the batch dimension). Default: None
        min_seqlen (optional int): If set, uses the specified value as the cached minimum sequence
            length for the returned nested tensor. This can be a useful alternative to computing
            this value on-demand, possibly avoiding a GPU -> CPU sync. Default: None
        max_seqlen (optional int): If set, uses the specified value as the cached maximum sequence
            length for the returned nested tensor. This can be a useful alternative to computing
            this value on-demand, possibly avoiding a GPU -> CPU sync. Default: None

    Example::

        >>> values = torch.randn(12, 5)
        >>> offsets = torch.tensor([0, 3, 5, 6, 10, 12])
        >>> nt = nested_tensor_from_jagged(values, offsets)
        >>> # 3D shape with the middle dimension jagged
        >>> nt.shape
        torch.Size([5, j2, 5])
        >>> # Length of each item in the batch:
        >>> offsets.diff()
        tensor([3, 2, 1, 4, 2])

        >>> values = torch.randn(6, 5)
        >>> offsets = torch.tensor([0, 2, 3, 6])
        >>> lengths = torch.tensor([1, 1, 2])
        >>> # NT with holes
        >>> nt = nested_tensor_from_jagged(values, offsets, lengths)
        >>> a, b, c = nt.unbind()
        >>> # Batch item 1 consists of indices [0, 1)
        >>> torch.equal(a, values[0:1, :])
        True
        >>> # Batch item 2 consists of indices [2, 3)
        >>> torch.equal(b, values[2:3, :])
        True
        >>> # Batch item 3 consists of indices [3, 5)
        >>> torch.equal(c, values[3:5, :])
        True
    '''
def masked_select(tensor: Tensor, mask: Tensor) -> Tensor:
    """
    Constructs a nested tensor given a strided tensor input and a strided mask, the resulting jagged layout nested tensor
    will have values retain values where the mask is equal to True. The dimensionality of the mask is preserved and is
    represented with the offsets, this is unlike :func:`masked_select` where the output is collapsed to a 1D tensor.

    Args:
    tensor (:class:`torch.Tensor`): a strided tensor from which the jagged layout nested tensor is constructed from.
    mask (:class:`torch.Tensor`): a strided mask tensor which is applied to the tensor input

    Example::

        >>> tensor = torch.randn(3, 3)
        >>> mask = torch.tensor([[False, False, True], [True, False, True], [False, False, True]])
        >>> nt = torch.nested.masked_select(tensor, mask)
        >>> nt.shape
        torch.Size([3, j4])
        >>> # Length of each item in the batch:
        >>> nt.offsets().diff()
        tensor([1, 2, 1])

        >>> tensor = torch.randn(6, 5)
        >>> mask = torch.tensor([False])
        >>> nt = torch.nested.masked_select(tensor, mask)
        >>> nt.shape
        torch.Size([6, j5])
        >>> # Length of each item in the batch:
        >>> nt.offsets().diff()
        tensor([0, 0, 0, 0, 0, 0])
    """
