from .container import ModuleList
from .linear import Linear
from .module import Module
from _typeshed import Incomplete
from collections.abc import Sequence
from torch import Tensor
from typing import NamedTuple

__all__ = ['AdaptiveLogSoftmaxWithLoss']

class _ASMoutput(NamedTuple):
    output: Incomplete
    loss: Incomplete

class AdaptiveLogSoftmaxWithLoss(Module):
    """Efficient softmax approximation.

    As described in
    `Efficient softmax approximation for GPUs by Edouard Grave, Armand Joulin,
    Moustapha Cissé, David Grangier, and Hervé Jégou
    <https://arxiv.org/abs/1609.04309>`__.

    Adaptive softmax is an approximate strategy for training models with large
    output spaces. It is most effective when the label distribution is highly
    imbalanced, for example in natural language modelling, where the word
    frequency distribution approximately follows the `Zipf's law`_.

    Adaptive softmax partitions the labels into several clusters, according to
    their frequency. These clusters may contain different number of targets
    each.
    Additionally, clusters containing less frequent labels assign lower
    dimensional embeddings to those labels, which speeds up the computation.
    For each minibatch, only clusters for which at least one target is
    present are evaluated.

    The idea is that the clusters which are accessed frequently
    (like the first one, containing most frequent labels), should also be cheap
    to compute -- that is, contain a small number of assigned labels.

    We highly recommend taking a look at the original paper for more details.

    * :attr:`cutoffs` should be an ordered Sequence of integers sorted
      in the increasing order.
      It controls number of clusters and the partitioning of targets into
      clusters. For example setting ``cutoffs = [10, 100, 1000]``
      means that first `10` targets will be assigned
      to the 'head' of the adaptive softmax, targets `11, 12, ..., 100` will be
      assigned to the first cluster, and targets `101, 102, ..., 1000` will be
      assigned to the second cluster, while targets
      `1001, 1002, ..., n_classes - 1` will be assigned
      to the last, third cluster.

    * :attr:`div_value` is used to compute the size of each additional cluster,
      which is given as
      :math:`\\left\\lfloor\\frac{\\texttt{in\\_features}}{\\texttt{div\\_value}^{idx}}\\right\\rfloor`,
      where :math:`idx` is the cluster index (with clusters
      for less frequent words having larger indices,
      and indices starting from :math:`1`).

    * :attr:`head_bias` if set to True, adds a bias term to the 'head' of the
      adaptive softmax. See paper for details. Set to False in the official
      implementation.

    .. warning::
        Labels passed as inputs to this module should be sorted according to
        their frequency. This means that the most frequent label should be
        represented by the index `0`, and the least frequent
        label should be represented by the index `n_classes - 1`.

    .. note::
        This module returns a ``NamedTuple`` with ``output``
        and ``loss`` fields. See further documentation for details.

    .. note::
        To compute log-probabilities for all classes, the ``log_prob``
        method can be used.

    Args:
        in_features (int): Number of features in the input tensor
        n_classes (int): Number of classes in the dataset
        cutoffs (Sequence): Cutoffs used to assign targets to their buckets
        div_value (float, optional): value used as an exponent to compute sizes
            of the clusters. Default: 4.0
        head_bias (bool, optional): If ``True``, adds a bias term to the 'head' of the
            adaptive softmax. Default: ``False``

    Returns:
        ``NamedTuple`` with ``output`` and ``loss`` fields:
            * **output** is a Tensor of size ``N`` containing computed target
              log probabilities for each example
            * **loss** is a Scalar representing the computed negative
              log likelihood loss

    Shape:
        - input: :math:`(N, \\texttt{in\\_features})` or :math:`(\\texttt{in\\_features})`
        - target: :math:`(N)` or :math:`()` where each value satisfies :math:`0 <= \\texttt{target[i]} <= \\texttt{n\\_classes}`
        - output1: :math:`(N)` or :math:`()`
        - output2: ``Scalar``

    .. _Zipf's law: https://en.wikipedia.org/wiki/Zipf%27s_law
    """
    in_features: int
    n_classes: int
    cutoffs: list[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList
    shortlist_size: Incomplete
    n_clusters: Incomplete
    head_size: Incomplete
    def __init__(self, in_features: int, n_classes: int, cutoffs: Sequence[int], div_value: float = 4.0, head_bias: bool = False, device=None, dtype=None) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, input_: Tensor, target_: Tensor) -> _ASMoutput: ...
    def _get_full_log_prob(self, input, head_output):
        """Given input tensor, and output of ``self.head``, compute the log of the full distribution."""
    def log_prob(self, input: Tensor) -> Tensor:
        """Compute log probabilities for all :math:`\\texttt{n\\_classes}`.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            log-probabilities of for each class :math:`c`
            in range :math:`0 <= c <= \\texttt{n\\_classes}`, where :math:`\\texttt{n\\_classes}` is a
            parameter passed to ``AdaptiveLogSoftmaxWithLoss`` constructor.

        Shape:
            - Input: :math:`(N, \\texttt{in\\_features})`
            - Output: :math:`(N, \\texttt{n\\_classes})`

        """
    def predict(self, input: Tensor) -> Tensor:
        """Return the class with the highest probability for each example in the input minibatch.

        This is equivalent to ``self.log_prob(input).argmax(dim=1)``, but is more efficient in some cases.

        Args:
            input (Tensor): a minibatch of examples

        Returns:
            output (Tensor): a class with the highest probability for each example

        Shape:
            - Input: :math:`(N, \\texttt{in\\_features})`
            - Output: :math:`(N)`
        """
