from _typeshed import Incomplete
from typing import Callable

np_str_obj_array_pattern: Incomplete

def default_convert(data):
    """
    Convert each NumPy array element into a :class:`torch.Tensor`.

    If the input is a `Sequence`, `Collection`, or `Mapping`, it tries to convert each element inside to a :class:`torch.Tensor`.
    If the input is not an NumPy array, it is left unchanged.
    This is used as the default function for collation when both `batch_sampler` and `batch_size`
    are NOT defined in :class:`~torch.utils.data.DataLoader`.

    The general input type to output type mapping is similar to that
    of :func:`~torch.utils.data.default_collate`. See the description there for more details.

    Args:
        data: a single data point to be converted

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with `int`
        >>> default_convert(0)
        0
        >>> # Example with NumPy array
        >>> default_convert(np.array([0, 1]))
        tensor([0, 1])
        >>> # Example with NamedTuple
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_convert(Point(0, 0))
        Point(x=0, y=0)
        >>> default_convert(Point(np.array(0), np.array(0)))
        Point(x=tensor(0), y=tensor(0))
        >>> # Example with List
        >>> default_convert([np.array([0, 1]), np.array([2, 3])])
        [tensor([0, 1]), tensor([2, 3])]
    """

default_collate_err_msg_format: str

def collate(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None):
    """
    General collate function that handles collection type of element within each batch.

    The function also opens function registry to deal with specific element types. `default_collate_fn_map`
    provides default collate functions for tensors, numpy arrays, numbers and strings.

    Args:
        batch: a single batch to be collated
        collate_fn_map: Optional dictionary mapping from element type to the corresponding collate function.
            If the element type isn't present in this dictionary,
            this function will go through each key of the dictionary in the insertion order to
            invoke the corresponding collate function if the element type is a subclass of the key.

    Examples:
        >>> def collate_tensor_fn(batch, *, collate_fn_map):
        ...     # Extend this function to handle batch of tensors
        ...     return torch.stack(batch, 0)
        >>> def custom_collate(batch):
        ...     collate_map = {torch.Tensor: collate_tensor_fn}
        ...     return collate(batch, collate_fn_map=collate_map)
        >>> # Extend `default_collate` by in-place modifying `default_collate_fn_map`
        >>> default_collate_fn_map.update({torch.Tensor: collate_tensor_fn})

    Note:
        Each collate function requires a positional argument for batch and a keyword argument
        for the dictionary of collate functions as `collate_fn_map`.
    """
def collate_tensor_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...
def collate_numpy_array_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...
def collate_numpy_scalar_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...
def collate_float_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...
def collate_int_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...
def collate_str_fn(batch, *, collate_fn_map: dict[type | tuple[type, ...], Callable] | None = None): ...

default_collate_fn_map: dict[type | tuple[type, ...], Callable]

def default_collate(batch):
    """
    Take in a batch of data and put the elements within the batch into a tensor with an additional outer dimension - batch size.

    The exact output type can be a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.

    Here is the general input type (based on the type of the element within the batch) to output type mapping:

        * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
        * NumPy Arrays -> :class:`torch.Tensor`
        * `float` -> :class:`torch.Tensor`
        * `int` -> :class:`torch.Tensor`
        * `str` -> `str` (unchanged)
        * `bytes` -> `bytes` (unchanged)
        * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
        * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`
        * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]),
          default_collate([V2_1, V2_2, ...]), ...]`

    Args:
        batch: a single batch to be collated

    Examples:
        >>> # xdoctest: +SKIP
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1], [2, 3]])
        [tensor([0, 2]), tensor([1, 3])]
        >>> # Two options to extend `default_collate` to handle specific type
        >>> # Option 1: Write custom collate function and invoke `default_collate`
        >>> def custom_collate(batch):
        ...     elem = batch[0]
        ...     if isinstance(elem, CustomType):  # Some custom condition
        ...         return ...
        ...     else:  # Fall back to `default_collate`
        ...         return default_collate(batch)
        >>> # Option 2: In-place modify `default_collate_fn_map`
        >>> def collate_customtype_fn(batch, *, collate_fn_map=None):
        ...     return ...
        >>> default_collate_fn_map.update(CustomType, collate_customtype_fn)
        >>> default_collate(batch)  # Handle `CustomType` automatically
    """
