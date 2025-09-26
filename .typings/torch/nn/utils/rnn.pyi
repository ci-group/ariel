import torch
from torch import Tensor
from typing import Any, NamedTuple, TypeVar, overload
from typing_extensions import Self

__all__ = ['PackedSequence', 'invert_permutation', 'pack_padded_sequence', 'pad_packed_sequence', 'pad_sequence', 'unpad_sequence', 'pack_sequence', 'unpack_sequence']

_T = TypeVar('_T')
_R = TypeVar('_R')

class PackedSequence_(NamedTuple):
    data: torch.Tensor
    batch_sizes: torch.Tensor
    sorted_indices: torch.Tensor | None
    unsorted_indices: torch.Tensor | None

class PackedSequence(PackedSequence_):
    """Holds the data and list of :attr:`batch_sizes` of a packed sequence.

    All RNN modules accept packed sequences as inputs.

    Note:
        Instances of this class should never be created manually. They are meant
        to be instantiated by functions like :func:`pack_padded_sequence`.

        Batch sizes represent the number elements at each sequence step in
        the batch, not the varying sequence lengths passed to
        :func:`pack_padded_sequence`.  For instance, given data ``abc`` and ``x``
        the :class:`PackedSequence` would contain data ``axbc`` with
        ``batch_sizes=[2,1,1]``.

    Attributes:
        data (Tensor): Tensor containing packed sequence
        batch_sizes (Tensor): Tensor of integers holding
            information about the batch size at each sequence step
        sorted_indices (Tensor, optional): Tensor of integers holding how this
            :class:`PackedSequence` is constructed from sequences.
        unsorted_indices (Tensor, optional): Tensor of integers holding how this
            to recover the original sequences with correct order.

    .. note::
        :attr:`data` can be on arbitrary device and of arbitrary dtype.
        :attr:`sorted_indices` and :attr:`unsorted_indices` must be ``torch.int64``
        tensors on the same device as :attr:`data`.

        However, :attr:`batch_sizes` should always be a CPU ``torch.int64`` tensor.

        This invariant is maintained throughout :class:`PackedSequence` class,
        and all functions that construct a :class:`PackedSequence` in PyTorch
        (i.e., they only pass in tensors conforming to this constraint).
    """
    def __new__(cls, data: Tensor, batch_sizes: Tensor | None = None, sorted_indices: Tensor | None = None, unsorted_indices: Tensor | None = None) -> Self: ...
    def pin_memory(self) -> Self: ...
    @overload
    def to(self, dtype: torch.dtype, non_blocking: bool = ..., copy: bool = ...) -> Self: ...
    @overload
    def to(self, device: str | torch.device | int | None = ..., dtype: torch.dtype | None = ..., non_blocking: bool = ..., copy: bool = ...) -> Self: ...
    @overload
    def to(self, other: Tensor, non_blocking: bool = ..., copy: bool = ...) -> Self: ...
    def cuda(self, *args: Any, **kwargs: Any) -> Self: ...
    def cpu(self, *args: Any, **kwargs: Any) -> Self: ...
    def double(self) -> Self: ...
    def float(self) -> Self: ...
    def half(self) -> Self: ...
    def long(self) -> Self: ...
    def int(self) -> Self: ...
    def short(self) -> Self: ...
    def char(self) -> Self: ...
    def byte(self) -> Self: ...
    @property
    def is_cuda(self) -> bool:
        """Return true if `self.data` stored on a gpu."""
    def is_pinned(self) -> bool:
        """Return true if `self.data` stored on in pinned memory."""

def invert_permutation(permutation: Tensor | None) -> Tensor | None: ...
def pack_padded_sequence(input: Tensor, lengths: Tensor | list[int], batch_first: bool = False, enforce_sorted: bool = True) -> PackedSequence:
    """Packs a Tensor containing padded sequences of variable length.

    :attr:`input` can be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) where ``T`` is the length
    of the longest sequence, ``B`` is the batch size, and ``*`` is any number of dimensions
    (including 0).

    For unsorted sequences, use `enforce_sorted = False`. If :attr:`enforce_sorted` is
    ``True``, the sequences should be sorted by length in a decreasing order, i.e.
    ``input[:,0]`` should be the longest sequence, and ``input[:,B-1]`` the shortest
    one. `enforce_sorted = True` is only necessary for ONNX export.

    It is an inverse operation to :func:`pad_packed_sequence`, and hence :func:`pad_packed_sequence`
    can be used to recover the underlying tensor packed in :class:`PackedSequence`.

    Note:
        This function accepts any input that has at least two dimensions. You
        can apply it to pack the labels, and use the output of the RNN with
        them to compute the loss directly. A Tensor can be retrieved from
        a :class:`PackedSequence` object by accessing its ``.data`` attribute.

    Args:
        input (Tensor): padded batch of variable length sequences.
        lengths (Tensor or list(int)): list of sequence lengths of each batch
            element (must be on the CPU if provided as a tensor).
        batch_first (bool, optional): if ``True``, the input is expected in ``B x T x *``
            format, ``T x B x *`` otherwise. Default: ``False``.
        enforce_sorted (bool, optional): if ``True``, the input is expected to
            contain sequences sorted by length in a decreasing order. If
            ``False``, the input will get sorted unconditionally. Default: ``True``.

    .. warning::
        The dim of ``input`` tensor will be truncated if its length larger than
        correspond value in ``length``.

    Returns:
        a :class:`PackedSequence` object
    """
def pad_packed_sequence(sequence: PackedSequence, batch_first: bool = False, padding_value: float = 0.0, total_length: int | None = None) -> tuple[Tensor, Tensor]:
    """Pad a packed batch of variable length sequences.

    It is an inverse operation to :func:`pack_padded_sequence`.

    The returned Tensor's data will be of size ``T x B x *`` (if :attr:`batch_first` is ``False``)
    or ``B x T x *`` (if :attr:`batch_first` is ``True``) , where ``T`` is the length of the longest
    sequence and ``B`` is the batch size.

    Example:
        >>> from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        >>> seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
        >>> lens = [2, 1, 3]
        >>> packed = pack_padded_sequence(
        ...     seq, lens, batch_first=True, enforce_sorted=False
        ... )
        >>> packed
        PackedSequence(data=tensor([4, 1, 3, 5, 2, 6]), batch_sizes=tensor([3, 2, 1]),
                       sorted_indices=tensor([2, 0, 1]), unsorted_indices=tensor([1, 2, 0]))
        >>> seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
        >>> seq_unpacked
        tensor([[1, 2, 0],
                [3, 0, 0],
                [4, 5, 6]])
        >>> lens_unpacked
        tensor([2, 1, 3])

    .. note::
        :attr:`total_length` is useful to implement the
        ``pack sequence -> recurrent network -> unpack sequence`` pattern in a
        :class:`~torch.nn.Module` wrapped in :class:`~torch.nn.DataParallel`.
        See :ref:`this FAQ section <pack-rnn-unpack-with-data-parallelism>` for
        details.

    Args:
        sequence (PackedSequence): batch to pad
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): values for padded elements.
        total_length (int, optional): if not ``None``, the output will be padded to
            have length :attr:`total_length`. This method will throw :class:`ValueError`
            if :attr:`total_length` is less than the max sequence length in
            :attr:`sequence`.

    Returns:
        Tuple of Tensor containing the padded sequence, and a Tensor
        containing the list of lengths of each sequence in the batch.
        Batch elements will be re-ordered as they were ordered originally when
        the batch was passed to ``pack_padded_sequence`` or ``pack_sequence``.
    """
def pad_sequence(sequences: Tensor | list[Tensor], batch_first: bool = False, padding_value: float = 0.0, padding_side: str = 'right') -> Tensor:
    """Pad a list of variable length Tensors with :attr:`padding_value`.

    ``pad_sequence`` stacks a list of Tensors along a new dimension, and pads them
    to equal length. :attr:`sequences` can be list of sequences with size ``L x *``,
    where `L` is length of the sequence and ``*`` is any number of dimensions
    (including ``0``). If :attr:`batch_first` is ``False``, the output is of size
    ``T x B x *``, and ``B x T x *`` otherwise, where ``B`` is the batch size
    (the number of elements in :attr:`sequences`), ``T`` is the length of the longest
    sequence.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> pad_sequence([a, b, c]).size()
        torch.Size([25, 3, 300])

    Note:
        This function returns a Tensor of size ``T x B x *`` or ``B x T x *``
        where `T` is the length of the longest sequence. This function assumes
        trailing dimensions and type of all the Tensors in sequences are same.

    Args:
        sequences (list[Tensor]): list of variable length sequences.
        batch_first (bool, optional): if ``True``, the output will be in ``B x T x *``
            format, ``T x B x *`` otherwise.
        padding_value (float, optional): value for padded elements. Default: ``0``.
        padding_side (str, optional): the side to pad the sequences on.
            Default: ``'right'``.

    Returns:
        Tensor of size ``T x B x *`` if :attr:`batch_first` is ``False``.
        Tensor of size ``B x T x *`` otherwise
    """
def unpad_sequence(padded_sequences: Tensor, lengths: Tensor, batch_first: bool = False) -> list[Tensor]:
    """Unpad padded Tensor into a list of variable length Tensors.

    ``unpad_sequence`` unstacks padded Tensor into a list of variable length Tensors.

    Example:
        >>> from torch.nn.utils.rnn import pad_sequence, unpad_sequence
        >>> a = torch.ones(25, 300)
        >>> b = torch.ones(22, 300)
        >>> c = torch.ones(15, 300)
        >>> sequences = [a, b, c]
        >>> padded_sequences = pad_sequence(sequences)
        >>> lengths = torch.as_tensor([v.size(0) for v in sequences])
        >>> unpadded_sequences = unpad_sequence(padded_sequences, lengths)
        >>> torch.allclose(sequences[0], unpadded_sequences[0])
        True
        >>> torch.allclose(sequences[1], unpadded_sequences[1])
        True
        >>> torch.allclose(sequences[2], unpadded_sequences[2])
        True

    Args:
        padded_sequences (Tensor): padded sequences.
        lengths (Tensor): length of original (unpadded) sequences.
        batch_first (bool, optional): whether batch dimension first or not. Default: ``False``.

    Returns:
        a list of :class:`Tensor` objects
    """
def pack_sequence(sequences: list[Tensor], enforce_sorted: bool = True) -> PackedSequence:
    """Packs a list of variable length Tensors.

    Consecutive call of the next functions: ``pad_sequence``, ``pack_padded_sequence``.

    ``sequences`` should be a list of Tensors of size ``L x *``, where `L` is
    the length of a sequence and `*` is any number of trailing dimensions,
    including ``0``.

    For unsorted sequences, use `enforce_sorted = False`. If ``enforce_sorted``
    is ``True``, the sequences should be sorted in the order of decreasing length.
    ``enforce_sorted = True`` is only necessary for ONNX export.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> pack_sequence([a, b, c])
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)

    Args:
        sequences (list[Tensor]): A list of sequences of decreasing length.
        enforce_sorted (bool, optional): if ``True``, checks that the input
            contains sequences sorted by length in a decreasing order. If
            ``False``, this condition is not checked. Default: ``True``.

    Returns:
        a :class:`PackedSequence` object
    """
def unpack_sequence(packed_sequences: PackedSequence) -> list[Tensor]:
    """Unpack PackedSequence into a list of variable length Tensors.

    ``packed_sequences`` should be a PackedSequence object.

    Example:
        >>> from torch.nn.utils.rnn import pack_sequence, unpack_sequence
        >>> a = torch.tensor([1, 2, 3])
        >>> b = torch.tensor([4, 5])
        >>> c = torch.tensor([6])
        >>> sequences = [a, b, c]
        >>> print(sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]
        >>> packed_sequences = pack_sequence(sequences)
        >>> print(packed_sequences)
        PackedSequence(data=tensor([1, 4, 6, 2, 5, 3]), batch_sizes=tensor([3, 2, 1]), sorted_indices=None, unsorted_indices=None)
        >>> unpacked_sequences = unpack_sequence(packed_sequences)
        >>> print(unpacked_sequences)
        [tensor([1, 2, 3]), tensor([4, 5]), tensor([6])]

    Args:
        packed_sequences (PackedSequence): A PackedSequence object.

    Returns:
        a list of :class:`Tensor` objects
    """
