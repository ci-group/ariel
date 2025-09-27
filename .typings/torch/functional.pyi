import torch
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor
from torch._lowrank import pca_lowrank as pca_lowrank, svd_lowrank as svd_lowrank
from typing import Any

__all__ = ['atleast_1d', 'atleast_2d', 'atleast_3d', 'align_tensors', 'broadcast_shapes', 'broadcast_tensors', 'cartesian_prod', 'block_diag', 'cdist', 'chain_matmul', 'einsum', 'istft', 'lu', 'norm', 'meshgrid', 'pca_lowrank', 'split', 'stft', 'svd_lowrank', 'tensordot', 'unique', 'unique_consecutive', 'unravel_index']

def broadcast_tensors(*tensors):
    """broadcast_tensors(*tensors) -> List of Tensors

    Broadcasts the given tensors according to :ref:`broadcasting-semantics`.

    Args:
        *tensors: any number of tensors of the same type

    .. warning::

        More than one element of a broadcasted tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensors, please clone them first.

    Example::

        >>> x = torch.arange(3).view(1, 3)
        >>> y = torch.arange(2).view(2, 1)
        >>> a, b = torch.broadcast_tensors(x, y)
        >>> a.size()
        torch.Size([2, 3])
        >>> a
        tensor([[0, 1, 2],
                [0, 1, 2]])
    """
def broadcast_shapes(*shapes):
    """broadcast_shapes(*shapes) -> Size

    Similar to :func:`broadcast_tensors` but for shapes.

    This is equivalent to
    ``torch.broadcast_tensors(*map(torch.empty, shapes))[0].shape``
    but avoids the need create to intermediate tensors. This is useful for
    broadcasting tensors of common batch shape but different rightmost shape,
    e.g. to broadcast mean vectors with covariance matrices.

    Example::

        >>> torch.broadcast_shapes((2,), (3, 1), (1, 1, 1))
        torch.Size([1, 3, 2])

    Args:
        \\*shapes (torch.Size): Shapes of tensors.

    Returns:
        shape (torch.Size): A shape compatible with all input shapes.

    Raises:
        RuntimeError: If shapes are incompatible.
    """
def split(tensor: Tensor, split_size_or_sections: int | list[int], dim: int = 0) -> tuple[Tensor, ...]:
    """Splits the tensor into chunks. Each chunk is a view of the original tensor.

    If :attr:`split_size_or_sections` is an integer type, then :attr:`tensor` will
    be split into equally sized chunks (if possible). Last chunk will be smaller if
    the tensor size along the given dimension :attr:`dim` is not divisible by
    :attr:`split_size`.

    If :attr:`split_size_or_sections` is a list, then :attr:`tensor` will be split
    into ``len(split_size_or_sections)`` chunks with sizes in :attr:`dim` according
    to :attr:`split_size_or_sections`.

    Args:
        tensor (Tensor): tensor to split.
        split_size_or_sections (int) or (list(int)): size of a single chunk or
            list of sizes for each chunk
        dim (int): dimension along which to split the tensor.

    Example::

        >>> a = torch.arange(10).reshape(5, 2)
        >>> a
        tensor([[0, 1],
                [2, 3],
                [4, 5],
                [6, 7],
                [8, 9]])
        >>> torch.split(a, 2)
        (tensor([[0, 1],
                 [2, 3]]),
         tensor([[4, 5],
                 [6, 7]]),
         tensor([[8, 9]]))
        >>> torch.split(a, [1, 4])
        (tensor([[0, 1]]),
         tensor([[2, 3],
                 [4, 5],
                 [6, 7],
                 [8, 9]]))
    """
def einsum(*args: Any) -> Tensor:
    '''einsum(equation, *operands) -> Tensor

    Sums the product of the elements of the input :attr:`operands` along dimensions specified using a notation
    based on the Einstein summation convention.

    Einsum allows computing many common multi-dimensional linear algebraic array operations by representing them
    in a short-hand format based on the Einstein summation convention, given by :attr:`equation`. The details of
    this format are described below, but the general idea is to label every dimension of the input :attr:`operands`
    with some subscript and define which subscripts are part of the output. The output is then computed by summing
    the product of the elements of the :attr:`operands` along the dimensions whose subscripts are not part of the
    output. For example, matrix multiplication can be computed using einsum as `torch.einsum("ij,jk->ik", A, B)`.
    Here, j is the summation subscript and i and k the output subscripts (see section below for more details on why).

    Equation:

        The :attr:`equation` string specifies the subscripts (letters in `[a-zA-Z]`) for each dimension of
        the input :attr:`operands` in the same order as the dimensions, separating subscripts for each operand by a
        comma (\',\'), e.g. `\'ij,jk\'` specify subscripts for two 2D operands. The dimensions labeled with the same subscript
        must be broadcastable, that is, their size must either match or be `1`. The exception is if a subscript is
        repeated for the same input operand, in which case the dimensions labeled with this subscript for this operand
        must match in size and the operand will be replaced by its diagonal along these dimensions. The subscripts that
        appear exactly once in the :attr:`equation` will be part of the output, sorted in increasing alphabetical order.
        The output is computed by multiplying the input :attr:`operands` element-wise, with their dimensions aligned based
        on the subscripts, and then summing out the dimensions whose subscripts are not part of the output.

        Optionally, the output subscripts can be explicitly defined by adding an arrow (\'->\') at the end of the equation
        followed by the subscripts for the output. For instance, the following equation computes the transpose of a
        matrix multiplication: \'ij,jk->ki\'. The output subscripts must appear at least once for some input operand and
        at most once for the output.

        Ellipsis (\'...\') can be used in place of subscripts to broadcast the dimensions covered by the ellipsis.
        Each input operand may contain at most one ellipsis which will cover the dimensions not covered by subscripts,
        e.g. for an input operand with 5 dimensions, the ellipsis in the equation `\'ab...c\'` cover the third and fourth
        dimensions. The ellipsis does not need to cover the same number of dimensions across the :attr:`operands` but the
        \'shape\' of the ellipsis (the size of the dimensions covered by them) must broadcast together. If the output is not
        explicitly defined with the arrow (\'->\') notation, the ellipsis will come first in the output (left-most dimensions),
        before the subscript labels that appear exactly once for the input operands. e.g. the following equation implements
        batch matrix multiplication `\'...ij,...jk\'`.

        A few final notes: the equation may contain whitespaces between the different elements (subscripts, ellipsis,
        arrow and comma) but something like `\'. . .\'` is not valid. An empty string `\'\'` is valid for scalar operands.

    .. note::

        ``torch.einsum`` handles ellipsis (\'...\') differently from NumPy in that it allows dimensions
        covered by the ellipsis to be summed over, that is, ellipsis are not required to be part of the output.

    .. note::

        Please install opt-einsum (https://optimized-einsum.readthedocs.io/en/stable/) in order to enroll into a more
        performant einsum. You can install when installing torch like so: `pip install torch[opt-einsum]` or by itself
        with `pip install opt-einsum`.

        If opt-einsum is available, this function will automatically speed up computation and/or consume less memory
        by optimizing contraction order through our opt_einsum backend :mod:`torch.backends.opt_einsum` (The _ vs - is
        confusing, I know). This optimization occurs when there are at least three inputs, since the order does not matter
        otherwise. Note that finding `the` optimal path is an NP-hard problem, thus, opt-einsum relies on different
        heuristics to achieve near-optimal results. If opt-einsum is not available, the default order is to contract
        from left to right.

        To bypass this default behavior, add the following to disable opt_einsum and skip path calculation:
        ``torch.backends.opt_einsum.enabled = False``

        To specify which strategy you\'d like for opt_einsum to compute the contraction path, add the following line:
        ``torch.backends.opt_einsum.strategy = \'auto\'``. The default strategy is \'auto\', and we also support \'greedy\' and
        \'optimal\'. Disclaimer that the runtime of \'optimal\' is factorial in the number of inputs! See more details in
        the opt_einsum documentation (https://optimized-einsum.readthedocs.io/en/stable/path_finding.html).

    .. note::

        As of PyTorch 1.10 :func:`torch.einsum` also supports the sublist format (see examples below). In this format,
        subscripts for each operand are specified by sublists, list of integers in the range [0, 52). These sublists
        follow their operands, and an extra sublist can appear at the end of the input to specify the output\'s
        subscripts., e.g. `torch.einsum(op1, sublist1, op2, sublist2, ..., [subslist_out])`. Python\'s `Ellipsis` object
        may be provided in a sublist to enable broadcasting as described in the Equation section above.

    Args:
        equation (str): The subscripts for the Einstein summation.
        operands (List[Tensor]): The tensors to compute the Einstein summation of.

    Examples::

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # trace
        >>> torch.einsum(\'ii\', torch.randn(4, 4))
        tensor(-1.2104)

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # diagonal
        >>> torch.einsum(\'ii->i\', torch.randn(4, 4))
        tensor([-0.1034,  0.7952, -0.2433,  0.4545])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # outer product
        >>> x = torch.randn(5)
        >>> y = torch.randn(4)
        >>> torch.einsum(\'i,j->ij\', x, y)
        tensor([[ 0.1156, -0.2897, -0.3918,  0.4963],
                [-0.3744,  0.9381,  1.2685, -1.6070],
                [ 0.7208, -1.8058, -2.4419,  3.0936],
                [ 0.1713, -0.4291, -0.5802,  0.7350],
                [ 0.5704, -1.4290, -1.9323,  2.4480]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # batch matrix multiplication
        >>> As = torch.randn(3, 2, 5)
        >>> Bs = torch.randn(3, 5, 4)
        >>> torch.einsum(\'bij,bjk->bik\', As, Bs)
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> # with sublist format and ellipsis
        >>> torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])
        tensor([[[-1.0564, -1.5904,  3.2023,  3.1271],
                [-1.6706, -0.8097, -0.8025, -2.1183]],

                [[ 4.2239,  0.3107, -0.5756, -0.2354],
                [-1.4558, -0.3460,  1.5087, -0.8530]],

                [[ 2.8153,  1.8787, -4.3839, -1.2112],
                [ 0.3728, -2.1131,  0.0921,  0.8305]]])

        >>> # batch permute
        >>> A = torch.randn(2, 3, 4, 5)
        >>> torch.einsum(\'...ij->...ji\', A).shape
        torch.Size([2, 3, 5, 4])

        >>> # equivalent to torch.nn.functional.bilinear
        >>> A = torch.randn(3, 5, 4)
        >>> l = torch.randn(2, 5)
        >>> r = torch.randn(2, 4)
        >>> torch.einsum(\'bn,anm,bm->ba\', l, A, r)
        tensor([[-0.3430, -5.2405,  0.4494],
                [ 0.3311,  5.5201, -3.0356]])
    '''
def meshgrid(*tensors: Tensor | list[Tensor], indexing: str | None = None) -> tuple[Tensor, ...]: ...
def stft(input: Tensor, n_fft: int, hop_length: int | None = None, win_length: int | None = None, window: Tensor | None = None, center: bool = True, pad_mode: str = 'reflect', normalized: bool = False, onesided: bool | None = None, return_complex: bool | None = None, align_to_window: bool | None = None) -> Tensor:
    '''Short-time Fourier transform (STFT).

    .. warning::
        From version 1.8.0, :attr:`return_complex` must always be given
        explicitly for real inputs and `return_complex=False` has been
        deprecated. Strongly prefer `return_complex=True` as in a future
        pytorch release, this function will only return complex tensors.

        Note that :func:`torch.view_as_real` can be used to recover a real
        tensor with an extra last dimension for real and imaginary components.

    .. warning::
        From version 2.1, a warning will be provided if a :attr:`window` is
        not specified. In a future release, this attribute will be required.
        Not providing a window currently defaults to using a rectangular window,
        which may result in undesirable artifacts. Consider using tapered windows,
        such as :func:`torch.hann_window`.

    The STFT computes the Fourier transform of short overlapping windows of the
    input. This giving frequency components of the signal as they change over
    time. The interface of this function is modeled after (but *not* a drop-in
    replacement for) librosa_ stft function.

    .. _librosa: https://librosa.org/doc/latest/generated/librosa.stft.html

    Ignoring the optional batch dimension, this method computes the following
    expression:

    .. math::
        X[\\omega, m] = \\sum_{k = 0}^{\\text{win\\_length-1}}%
                            \\text{window}[k]\\ \\text{input}[m \\times \\text{hop\\_length} + k]\\ %
                            \\exp\\left(- j \\frac{2 \\pi \\cdot \\omega k}{\\text{n\\_fft}}\\right),

    where :math:`m` is the index of the sliding window, and :math:`\\omega` is
    the frequency :math:`0 \\leq \\omega < \\text{n\\_fft}` for ``onesided=False``,
    or :math:`0 \\leq \\omega < \\lfloor \\text{n\\_fft} / 2 \\rfloor + 1` for ``onesided=True``.

    * :attr:`input` must be either a 1-D time sequence or a 2-D batch of time
      sequences.

    * If :attr:`hop_length` is ``None`` (default), it is treated as equal to
      ``floor(n_fft / 4)``.

    * If :attr:`win_length` is ``None`` (default), it is treated as equal to
      :attr:`n_fft`.

    * :attr:`window` can be a 1-D tensor of size :attr:`win_length`, e.g., from
      :meth:`torch.hann_window`. If :attr:`window` is ``None`` (default), it is
      treated as if having :math:`1` everywhere in the window. If
      :math:`\\text{win\\_length} < \\text{n\\_fft}`, :attr:`window` will be padded on
      both sides to length :attr:`n_fft` before being applied.

    * If :attr:`center` is ``True`` (default), :attr:`input` will be padded on
      both sides so that the :math:`t`-th frame is centered at time
      :math:`t \\times \\text{hop\\_length}`. Otherwise, the :math:`t`-th frame
      begins at time  :math:`t \\times \\text{hop\\_length}`.

    * :attr:`pad_mode` determines the padding method used on :attr:`input` when
      :attr:`center` is ``True``. See :meth:`torch.nn.functional.pad` for
      all available options. Default is ``"reflect"``.

    * If :attr:`onesided` is ``True`` (default for real input), only values for
      :math:`\\omega` in :math:`\\left[0, 1, 2, \\dots, \\left\\lfloor
      \\frac{\\text{n\\_fft}}{2} \\right\\rfloor + 1\\right]` are returned because
      the real-to-complex Fourier transform satisfies the conjugate symmetry,
      i.e., :math:`X[m, \\omega] = X[m, \\text{n\\_fft} - \\omega]^*`.
      Note if the input or window tensors are complex, then :attr:`onesided`
      output is not possible.

    * If :attr:`normalized` is ``True`` (default is ``False``), the function
      returns the normalized STFT results, i.e., multiplied by :math:`(\\text{frame\\_length})^{-0.5}`.

    * If :attr:`return_complex` is ``True`` (default if input is complex), the
      return is a ``input.dim() + 1`` dimensional complex tensor. If ``False``,
      the output is a ``input.dim() + 2`` dimensional real tensor where the last
      dimension represents the real and imaginary components.

    Returns either a complex tensor of size :math:`(* \\times N \\times T)` if
    :attr:`return_complex` is true, or a real tensor of size :math:`(* \\times N
    \\times T \\times 2)`. Where :math:`*` is the optional batch size of
    :attr:`input`, :math:`N` is the number of frequencies where STFT is applied
    and :math:`T` is the total number of frames used.

    .. warning::
      This function changed signature at version 0.4.1. Calling with the
      previous signature may cause error or return incorrect result.

    Args:
        input (Tensor): the input tensor of shape `(B?, L)` where `B?` is an optional
            batch dimension
        n_fft (int): size of Fourier transform
        hop_length (int, optional): the distance between neighboring sliding window
            frames. Default: ``None`` (treated as equal to ``floor(n_fft / 4)``)
        win_length (int, optional): the size of window frame and STFT filter.
            Default: ``None``  (treated as equal to :attr:`n_fft`)
        window (Tensor, optional): the optional window function.
            Shape must be 1d and `<= n_fft`
            Default: ``None`` (treated as window of all :math:`1` s)
        center (bool, optional): whether to pad :attr:`input` on both sides so
            that the :math:`t`-th frame is centered at time :math:`t \\times \\text{hop\\_length}`.
            Default: ``True``
        pad_mode (str, optional): controls the padding method used when
            :attr:`center` is ``True``. Default: ``"reflect"``
        normalized (bool, optional): controls whether to return the normalized STFT results
             Default: ``False``
        onesided (bool, optional): controls whether to return half of results to
            avoid redundancy for real inputs.
            Default: ``True`` for real :attr:`input` and :attr:`window`, ``False`` otherwise.
        return_complex (bool, optional): whether to return a complex tensor, or
            a real tensor with an extra last dimension for the real and
            imaginary components.

            .. versionchanged:: 2.0
               ``return_complex`` is now a required argument for real inputs,
               as the default is being transitioned to ``True``.

            .. deprecated:: 2.0
               ``return_complex=False`` is deprecated, instead use ``return_complex=True``
               Note that calling :func:`torch.view_as_real` on the output will
               recover the deprecated output format.

    Returns:
        Tensor: A tensor containing the STFT result with shape `(B?, N, T, C?)` where
           - `B?` is an optional batch dimension from the input.
           - `N` is the number of frequency samples, `(n_fft // 2) + 1` for
             `onesided=True`, or otherwise `n_fft`.
           - `T` is the number of frames, `1 + L // hop_length`
             for `center=True`, or `1 + (L - n_fft) // hop_length` otherwise.
           - `C?` is an optional length-2 dimension of real and imaginary
             components, present when `return_complex=False`.

    '''

istft: Incomplete
_unique_impl_out = Any
unique: Incomplete
unique_consecutive: Incomplete

def tensordot(a, b, dims: int = 2, out: torch.Tensor | None = None):
    """Returns a contraction of a and b over multiple dimensions.

    :attr:`tensordot` implements a generalized matrix product.

    Args:
      a (Tensor): Left tensor to contract
      b (Tensor): Right tensor to contract
      dims (int or Tuple[List[int], List[int]] or List[List[int]] containing two lists or Tensor): number of dimensions to
         contract or explicit lists of dimensions for :attr:`a` and
         :attr:`b` respectively

    When called with a non-negative integer argument :attr:`dims` = :math:`d`, and
    the number of dimensions of :attr:`a` and :attr:`b` is :math:`m` and :math:`n`,
    respectively, :func:`~torch.tensordot` computes

    .. math::
        r_{i_0,...,i_{m-d}, i_d,...,i_n}
          = \\sum_{k_0,...,k_{d-1}} a_{i_0,...,i_{m-d},k_0,...,k_{d-1}} \\times b_{k_0,...,k_{d-1}, i_d,...,i_n}.

    When called with :attr:`dims` of the list form, the given dimensions will be contracted
    in place of the last :math:`d` of :attr:`a` and the first :math:`d` of :math:`b`. The sizes
    in these dimensions must match, but :func:`~torch.tensordot` will deal with broadcasted
    dimensions.

    Examples::

        >>> a = torch.arange(60.).reshape(3, 4, 5)
        >>> b = torch.arange(24.).reshape(4, 3, 2)
        >>> torch.tensordot(a, b, dims=([1, 0], [0, 1]))
        tensor([[4400., 4730.],
                [4532., 4874.],
                [4664., 5018.],
                [4796., 5162.],
                [4928., 5306.]])

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> a = torch.randn(3, 4, 5, device='cuda')
        >>> b = torch.randn(4, 5, 6, device='cuda')
        >>> c = torch.tensordot(a, b, dims=2).cpu()
        tensor([[ 8.3504, -2.5436,  6.2922,  2.7556, -1.0732,  3.2741],
                [ 3.3161,  0.0704,  5.0187, -0.4079, -4.3126,  4.8744],
                [ 0.8223,  3.9445,  3.2168, -0.2400,  3.4117,  1.7780]])

        >>> a = torch.randn(3, 5, 4, 6)
        >>> b = torch.randn(6, 4, 5, 3)
        >>> torch.tensordot(a, b, dims=([2, 1, 3], [1, 2, 0]))
        tensor([[  7.7193,  -2.4867, -10.3204],
                [  1.5513, -14.4737,  -6.5113],
                [ -0.2850,   4.2573,  -3.5997]])
    """
def cartesian_prod(*tensors: Tensor) -> Tensor:
    """Do cartesian product of the given sequence of tensors. The behavior is similar to
    python's `itertools.product`.

    Args:
        *tensors: any number of 1 dimensional tensors.

    Returns:
        Tensor: A tensor equivalent to converting all the input tensors into lists,
        do `itertools.product` on these lists, and finally convert the resulting list
        into tensor.

    Example::

        >>> import itertools
        >>> a = [1, 2, 3]
        >>> b = [4, 5]
        >>> list(itertools.product(a, b))
        [(1, 4), (1, 5), (2, 4), (2, 5), (3, 4), (3, 5)]
        >>> tensor_a = torch.tensor(a)
        >>> tensor_b = torch.tensor(b)
        >>> torch.cartesian_prod(tensor_a, tensor_b)
        tensor([[1, 4],
                [1, 5],
                [2, 4],
                [2, 5],
                [3, 4],
                [3, 5]])
    """
def block_diag(*tensors):
    """Create a block diagonal matrix from provided tensors.

    Args:
        *tensors: One or more tensors with 0, 1, or 2 dimensions.

    Returns:
        Tensor: A 2 dimensional tensor with all the input tensors arranged in
        order such that their upper left and lower right corners are
        diagonally adjacent. All other elements are set to 0.

    Example::

        >>> import torch
        >>> A = torch.tensor([[0, 1], [1, 0]])
        >>> B = torch.tensor([[3, 4, 5], [6, 7, 8]])
        >>> C = torch.tensor(7)
        >>> D = torch.tensor([1, 2, 3])
        >>> E = torch.tensor([[4], [5], [6]])
        >>> torch.block_diag(A, B, C, D, E)
        tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 4, 5, 0, 0, 0, 0, 0],
                [0, 0, 6, 7, 8, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 7, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 2, 3, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 6]])
    """
def cdist(x1: Tensor, x2: Tensor, p: float = 2.0, compute_mode: str = 'use_mm_for_euclid_dist_if_necessary') -> Tensor:
    """Computes batched the p-norm distance between each pair of the two collections of row vectors.

    Args:
        x1 (Tensor): input tensor where the last two dimensions represent the points and the feature dimension respectively.
            The shape can be :math:`D_1 \\times D_2 \\times \\cdots \\times D_n \\times P \\times M`,
            where :math:`P` is the number of points and :math:`M` is the feature dimension.
        x2 (Tensor): input tensor where the last two dimensions also represent the points and the feature dimension respectively.
            The shape can be :math:`D_1' \\times D_2' \\times \\cdots \\times D_m' \\times R \\times M`,
            where :math:`R` is the number of points and :math:`M` is the feature dimension,
            which should match the feature dimension of `x1`.
        p: p value for the p-norm distance to calculate between each vector pair
            :math:`\\in [0, \\infty]`.
        compute_mode:
            'use_mm_for_euclid_dist_if_necessary' - will use matrix multiplication approach to calculate
            euclidean distance (p = 2) if P > 25 or R > 25
            'use_mm_for_euclid_dist' - will always use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            'donot_use_mm_for_euclid_dist' - will never use matrix multiplication approach to calculate
            euclidean distance (p = 2)
            Default: use_mm_for_euclid_dist_if_necessary.

    If x1 has shape :math:`B \\times P \\times M` and x2 has shape :math:`B \\times R \\times M` then the
    output will have shape :math:`B \\times P \\times R`.

    This function is equivalent to `scipy.spatial.distance.cdist(input,'minkowski', p=p)`
    if :math:`p \\in (0, \\infty)`. When :math:`p = 0` it is equivalent to
    `scipy.spatial.distance.cdist(input, 'hamming') * M`. When :math:`p = \\infty`, the closest
    scipy function is `scipy.spatial.distance.cdist(xn, lambda x, y: np.abs(x - y).max())`.

    Example:

        >>> a = torch.tensor([[0.9041, 0.0196], [-0.3108, -2.4423], [-0.4821, 1.059]])
        >>> a
        tensor([[ 0.9041,  0.0196],
                [-0.3108, -2.4423],
                [-0.4821,  1.0590]])
        >>> b = torch.tensor([[-2.1763, -0.4713], [-0.6986, 1.3702]])
        >>> b
        tensor([[-2.1763, -0.4713],
                [-0.6986,  1.3702]])
        >>> torch.cdist(a, b, p=2)
        tensor([[3.1193, 2.0959],
                [2.7138, 3.8322],
                [2.2830, 0.3791]])
    """
def atleast_1d(*tensors):
    """
    Returns a 1-dimensional view of each input tensor with zero dimensions.
    Input tensors with one or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.arange(2)
        >>> x
        tensor([0, 1])
        >>> torch.atleast_1d(x)
        tensor([0, 1])
        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_1d(x)
        tensor([1.])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_1d((x, y))
        (tensor([0.5000]), tensor([1.]))
    """
def atleast_2d(*tensors):
    """
    Returns a 2-dimensional view of each input tensor with zero dimensions.
    Input tensors with two or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example::

        >>> x = torch.tensor(1.)
        >>> x
        tensor(1.)
        >>> torch.atleast_2d(x)
        tensor([[1.]])
        >>> x = torch.arange(4).view(2, 2)
        >>> x
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_2d(x)
        tensor([[0, 1],
                [2, 3]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.)
        >>> torch.atleast_2d((x, y))
        (tensor([[0.5000]]), tensor([[1.]]))
    """
def atleast_3d(*tensors):
    """
    Returns a 3-dimensional view of each input tensor with zero dimensions.
    Input tensors with three or more dimensions are returned as-is.

    Args:
        input (Tensor or list of Tensors)

    Returns:
        output (Tensor or tuple of Tensors)

    Example:

        >>> x = torch.tensor(0.5)
        >>> x
        tensor(0.5000)
        >>> torch.atleast_3d(x)
        tensor([[[0.5000]]])
        >>> y = torch.arange(4).view(2, 2)
        >>> y
        tensor([[0, 1],
                [2, 3]])
        >>> torch.atleast_3d(y)
        tensor([[[0],
                 [1]],
                <BLANKLINE>
                [[2],
                 [3]]])
        >>> x = torch.tensor(1).view(1, 1, 1)
        >>> x
        tensor([[[1]]])
        >>> torch.atleast_3d(x)
        tensor([[[1]]])
        >>> x = torch.tensor(0.5)
        >>> y = torch.tensor(1.0)
        >>> torch.atleast_3d((x, y))
        (tensor([[[0.5000]]]), tensor([[[1.]]]))
    """
def norm(input, p: float | str | None = 'fro', dim=None, keepdim: bool = False, out=None, dtype=None):
    """Returns the matrix norm or vector norm of a given tensor.

    .. warning::

        torch.norm is deprecated and may be removed in a future PyTorch release.
        Its documentation and behavior may be incorrect, and it is no longer
        actively maintained.

        Use :func:`torch.linalg.vector_norm` when computing vector norms and
        :func:`torch.linalg.matrix_norm` when computing matrix norms.
        For a function with a similar behavior as this one see :func:`torch.linalg.norm`.
        Note, however, the signature for these functions is slightly different than the
        signature for ``torch.norm``.

    Args:
        input (Tensor): The input tensor. Its data type must be either a floating
            point or complex type. For complex inputs, the norm is calculated using the
            absolute value of each element. If the input is complex and neither
            :attr:`dtype` nor :attr:`out` is specified, the result's data type will
            be the corresponding floating point type (e.g. float if :attr:`input` is
            complexfloat).

        p (int, float, inf, -inf, 'fro', 'nuc', optional): the order of norm. Default: ``'fro'``
            The following norms can be calculated:

            ======  ==============  ==========================
            ord     matrix norm     vector norm
            ======  ==============  ==========================
            'fro'   Frobenius norm  --
            'nuc'   nuclear norm    --
            Number  --              sum(abs(x)**ord)**(1./ord)
            ======  ==============  ==========================

            The vector norm can be calculated across any number of dimensions.
            The corresponding dimensions of :attr:`input` are flattened into
            one dimension, and the norm is calculated on the flattened
            dimension.

            Frobenius norm produces the same result as ``p=2`` in all cases
            except when :attr:`dim` is a list of three or more dims, in which
            case Frobenius norm throws an error.

            Nuclear norm can only be calculated across exactly two dimensions.

        dim (int, tuple of ints, list of ints, optional):
            Specifies which dimension or dimensions of :attr:`input` to
            calculate the norm across. If :attr:`dim` is ``None``, the norm will
            be calculated across all dimensions of :attr:`input`. If the norm
            type indicated by :attr:`p` does not support the specified number of
            dimensions, an error will occur.
        keepdim (bool, optional): whether the output tensors have :attr:`dim`
            retained or not. Ignored if :attr:`dim` = ``None`` and
            :attr:`out` = ``None``. Default: ``False``
        out (Tensor, optional): the output tensor. Ignored if
            :attr:`dim` = ``None`` and :attr:`out` = ``None``.
        dtype (:class:`torch.dtype`, optional): the desired data type of
            returned tensor. If specified, the input tensor is casted to
            :attr:`dtype` while performing the operation. Default: None.

    .. note::
        Even though ``p='fro'`` supports any number of dimensions, the true
        mathematical definition of Frobenius norm only applies to tensors with
        exactly two dimensions. :func:`torch.linalg.matrix_norm` with ``ord='fro'``
        aligns with the mathematical definition, since it can only be applied across
        exactly two dimensions.

    Example::

        >>> import torch
        >>> a = torch.arange(9, dtype= torch.float) - 4
        >>> b = a.reshape((3, 3))
        >>> torch.norm(a)
        tensor(7.7460)
        >>> torch.norm(b)
        tensor(7.7460)
        >>> torch.norm(a, float('inf'))
        tensor(4.)
        >>> torch.norm(b, float('inf'))
        tensor(4.)
        >>> c = torch.tensor([[ 1, 2, 3], [-1, 1, 4]] , dtype=torch.float)
        >>> torch.norm(c, dim=0)
        tensor([1.4142, 2.2361, 5.0000])
        >>> torch.norm(c, dim=1)
        tensor([3.7417, 4.2426])
        >>> torch.norm(c, p=1, dim=1)
        tensor([6., 6.])
        >>> d = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
        >>> torch.norm(d, dim=(1, 2))
        tensor([ 3.7417, 11.2250])
        >>> torch.norm(d[0, :, :]), torch.norm(d[1, :, :])
        (tensor(3.7417), tensor(11.2250))
    """
def unravel_index(indices: Tensor, shape: int | Sequence[int] | torch.Size) -> tuple[Tensor, ...]:
    """Converts a tensor of flat indices into a tuple of coordinate tensors that
    index into an arbitrary tensor of the specified shape.

    Args:
        indices (Tensor): An integer tensor containing indices into the
            flattened version of an arbitrary tensor of shape :attr:`shape`.
            All elements must be in the range ``[0, prod(shape) - 1]``.

        shape (int, sequence of ints, or torch.Size): The shape of the arbitrary
            tensor. All elements must be non-negative.

    Returns:
        tuple of Tensors: Each ``i``-th tensor in the output corresponds with
        dimension ``i`` of :attr:`shape`. Each tensor has the same shape as
        ``indices`` and contains one index into dimension ``i`` for each of the
        flat indices given by ``indices``.

    Example::

        >>> import torch
        >>> torch.unravel_index(torch.tensor(4), (3, 2))
        (tensor(2),
         tensor(0))

        >>> torch.unravel_index(torch.tensor([4, 1]), (3, 2))
        (tensor([2, 0]),
         tensor([0, 1]))

        >>> torch.unravel_index(torch.tensor([0, 1, 2, 3, 4, 5]), (3, 2))
        (tensor([0, 0, 1, 1, 2, 2]),
         tensor([0, 1, 0, 1, 0, 1]))

        >>> torch.unravel_index(torch.tensor([1234, 5678]), (10, 10, 10, 10))
        (tensor([1, 5]),
         tensor([2, 6]),
         tensor([3, 7]),
         tensor([4, 8]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (10, 10, 10, 10))
        (tensor([[1], [5]]),
         tensor([[2], [6]]),
         tensor([[3], [7]]),
         tensor([[4], [8]]))

        >>> torch.unravel_index(torch.tensor([[1234], [5678]]), (100, 100))
        (tensor([[12], [56]]),
         tensor([[34], [78]]))
    """
def chain_matmul(*matrices, out=None):
    '''Returns the matrix product of the :math:`N` 2-D tensors. This product is efficiently computed
    using the matrix chain order algorithm which selects the order in which incurs the lowest cost in terms
    of arithmetic operations (`[CLRS]`_). Note that since this is a function to compute the product, :math:`N`
    needs to be greater than or equal to 2; if equal to 2 then a trivial matrix-matrix product is returned.
    If :math:`N` is 1, then this is a no-op - the original matrix is returned as is.

    .. warning::

        :func:`torch.chain_matmul` is deprecated and will be removed in a future PyTorch release.
        Use :func:`torch.linalg.multi_dot` instead, which accepts a list of two or more tensors
        rather than multiple arguments.

    Args:
        matrices (Tensors...): a sequence of 2 or more 2-D tensors whose product is to be determined.
        out (Tensor, optional): the output tensor. Ignored if :attr:`out` = ``None``.

    Returns:
        Tensor: if the :math:`i^{th}` tensor was of dimensions :math:`p_{i} \\times p_{i + 1}`, then the product
        would be of dimensions :math:`p_{1} \\times p_{N + 1}`.

    Example::

        >>> # xdoctest: +SKIP
        >>> # xdoctest: +IGNORE_WANT("non-deterministic")
        >>> a = torch.randn(3, 4)
        >>> b = torch.randn(4, 5)
        >>> c = torch.randn(5, 6)
        >>> d = torch.randn(6, 7)
        >>> # will raise a deprecation warning
        >>> torch.chain_matmul(a, b, c, d)
        tensor([[ -2.3375,  -3.9790,  -4.1119,  -6.6577,   9.5609, -11.5095,  -3.2614],
                [ 21.4038,   3.3378,  -8.4982,  -5.2457, -10.2561,  -2.4684,   2.7163],
                [ -0.9647,  -5.8917,  -2.3213,  -5.2284,  12.8615, -12.2816,  -2.5095]])

    .. _`[CLRS]`: https://mitpress.mit.edu/books/introduction-algorithms-third-edition
    '''
_ListOrSeq = Sequence[Tensor]
lu: Incomplete

def align_tensors(*tensors) -> None: ...
