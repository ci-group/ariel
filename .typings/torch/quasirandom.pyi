import torch
from _typeshed import Incomplete

class SobolEngine:
    '''
    The :class:`torch.quasirandom.SobolEngine` is an engine for generating
    (scrambled) Sobol sequences. Sobol sequences are an example of low
    discrepancy quasi-random sequences.

    This implementation of an engine for Sobol sequences is capable of
    sampling sequences up to a maximum dimension of 21201. It uses direction
    numbers from https://web.maths.unsw.edu.au/~fkuo/sobol/ obtained using the
    search criterion D(6) up to the dimension 21201. This is the recommended
    choice by the authors.

    References:
      - Art B. Owen. Scrambling Sobol and Niederreiter-Xing points.
        Journal of Complexity, 14(4):466-489, December 1998.

      - I. M. Sobol. The distribution of points in a cube and the accurate
        evaluation of integrals.
        Zh. Vychisl. Mat. i Mat. Phys., 7:784-802, 1967.

    Args:
        dimension (Int): The dimensionality of the sequence to be drawn
        scramble (bool, optional): Setting this to ``True`` will produce
                                   scrambled Sobol sequences. Scrambling is
                                   capable of producing better Sobol
                                   sequences. Default: ``False``.
        seed (Int, optional): This is the seed for the scrambling. The seed
                              of the random number generator is set to this,
                              if specified. Otherwise, it uses a random seed.
                              Default: ``None``

    Examples::

        >>> # xdoctest: +SKIP("unseeded random state")
        >>> soboleng = torch.quasirandom.SobolEngine(dimension=5)
        >>> soboleng.draw(3)
        tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                [0.5000, 0.5000, 0.5000, 0.5000, 0.5000],
                [0.7500, 0.2500, 0.2500, 0.2500, 0.7500]])
    '''
    MAXBIT: int
    MAXDIM: int
    seed: Incomplete
    scramble: Incomplete
    dimension: Incomplete
    sobolstate: Incomplete
    shift: Incomplete
    quasi: Incomplete
    _first_point: Incomplete
    num_generated: int
    def __init__(self, dimension, scramble: bool = False, seed=None) -> None: ...
    def draw(self, n: int = 1, out: torch.Tensor | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Function to draw a sequence of :attr:`n` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(n, dimension)`.

        Args:
            n (Int, optional): The length of sequence of points to draw.
                               Default: 1
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
    def draw_base2(self, m: int, out: torch.Tensor | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
        """
        Function to draw a sequence of :attr:`2**m` points from a Sobol sequence.
        Note that the samples are dependent on the previous samples. The size
        of the result is :math:`(2**m, dimension)`.

        Args:
            m (Int): The (base2) exponent of the number of points to draw.
            out (Tensor, optional): The output tensor
            dtype (:class:`torch.dtype`, optional): the desired data type of the
                                                    returned tensor.
                                                    Default: ``None``
        """
    def reset(self):
        """
        Function to reset the ``SobolEngine`` to base state.
        """
    def fast_forward(self, n):
        """
        Function to fast-forward the state of the ``SobolEngine`` by
        :attr:`n` steps. This is equivalent to drawing :attr:`n` samples
        without using the samples.

        Args:
            n (Int): The number of steps to fast-forward by.
        """
    def _scramble(self) -> None: ...
    def __repr__(self) -> str: ...
